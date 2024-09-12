""" Hamiltonian ODE-based physics-augmented VAE model.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad
from torchdiffeq import odeint
from .. import utils
from ..mlp import MLP
from ..attention import SelfAttention
# import sys; sys.path.append('../'); import utils; from mlp import MLP


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_y = config['dim_y']
        dim_x = config['dim_x']
        dim_t = config['dim_t']
        dim_z_phy = config['dim_z_phy']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        x_lnvar = config['x_lnvar']
        
        # x_lnvar
        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        self.func_aux2_map = nn.Linear(dim_y, dim_x)
        if dim_z_aux2 >= 0:
            hidlayers_aux2 = config['hidlayers_aux2_dec']
            self.func_aux2_res = MLP([dim_z_aux2,]+hidlayers_aux2+[dim_x*dim_t,], activation)


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        dim_y = config['dim_y']
        dim_x = config['dim_x']
        dim_t = config['dim_t']
        dim_z_phy = config['dim_z_phy']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        num_units_feat = config['num_units_feat']
        hidlayers_init_yy = config['hidlayers_init_yy']
        batch_size = config['batch_size']
        # x --> feature
        self.func_feat = FeatureExtractor(config)
        
        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['hidlayers_aux2_enc']
            
            # x --> feature_aux2
            self.func_feat_aux2 = FeatureExtractor(config)

            # feature --> z_aux2
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if ((not no_phy) and (dim_z_phy > 0)):
            hidlayers_unmixer = config['hidlayers_unmixer']
            hidlayers_z_phy = config['hidlayers_z_phy']

            # x, z_aux2 --> unmixed - x
            self.func_unmixer_res = MLP([dim_y*dim_t+max(dim_z_aux2,0),]+hidlayers_unmixer+[dim_y*dim_t,], activation)

            # unmixed --> feature_phy
            self.func_feat_phy = FeatureExtractor(config)

            # features --> z_phy
            self.func_z_phy_mean = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)

        # feature --> init_yy
        self.func_init_yy = MLP([num_units_feat,]+hidlayers_init_yy+[2*dim_y,], activation)



class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        dim_x = config['dim_x']
        dim_t = config['dim_t']
        activation = config['activation']
        arch_feat = config['arch_feat']
        num_units_feat = config['num_units_feat']
        self.dim_x = dim_x
        self.dim_t = dim_t
        self.arch_feat = arch_feat
        self.num_units_feat = num_units_feat

        if arch_feat=='mlp':
            hidlayers_feat = config['hidlayers_feat']

            self.func= MLP([dim_x*dim_t,]+hidlayers_feat+[num_units_feat,], activation, actfun_output=True)
        elif arch_feat=='rnn':
            num_rnns_feat = config['num_rnns_feat']

            self.num_rnns_feat = num_rnns_feat
            self.func = nn.GRU(dim_x, num_units_feat, num_layers=num_rnns_feat, bidirectional=False)
        else:
            raise ValueError('unknown feature type')

    
    def forward(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_x, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        if self.arch_feat=='mlp':
            feat = self.func(x_.view(n,-1))
        elif self.arch_feat=='rnn':
            h_0 = torch.zeros(self.num_rnns_feat, n, self.num_units_feat, device=device)
            out, h_n = self.func(x_.permute(2, 0, 1), h_0)
            feat = out[-1]

        return feat

class Physics(nn.Module):
    def __init__(self, config:dict):
        super(Physics, self).__init__()

        activation = config['activation']
        dim_y = config['dim_y']
        dim_z_phy = config['dim_z_phy']
        hidlayers_H = config['hidlayers_H']

        self.dim_y = dim_y
        self.H = MLP([2*dim_y+dim_z_phy,]+hidlayers_H+[1,], activation)


    def forward(self, z_phy:torch.Tensor, yy:torch.Tensor):
        """
        given parameter and yy, return dyy/dt
        [state]
            yy: shape <n x 2dim_y>; the first half should be q (generalized position), the latter half should be p (generalized momentum)
        [physics parameter]
            z_phy: shape <n x dim_z_phy>
        """
        yy_new= yy.detach()
        yy_new.requires_grad = True
        z_phy_new=z_phy.detach()
        z_phy_new.requires_grad= True
        
        # yy = [q, p]
        
        H_val = self.H(torch.cat((yy_new, z_phy_new), dim=1))
        H_grad = grad([h for h in H_val], [yy_new], create_graph = self.training, only_inputs=True)[0]
        dHdq = H_grad[:, 0:self.dim_y]
        dHdp = H_grad[:, self.dim_y:]
        return torch.cat([dHdp, -dHdq], dim=1)
        

class VAE(nn.Module):
    def __init__(self, config:dict):
        super(VAE, self).__init__()

        self.dim_y = config['dim_y']
        self.dim_x = config['dim_x']
        self.dim_t = config['dim_t']
        self.dim_z_phy = config['dim_z_phy']
        self.dim_z_aux2 = config['dim_z_aux2']
        self.activation = config['activation']
        self.dt = config['dt']
        self.intg_lev = config['intg_lev']
        self.ode_solver = config['ode_solver']
        self.no_phy = config['no_phy']
        self.drop_feat = config['drop_feat']
        self.sample_drop_perc = config['sample_drop_perc']
        self.feat_drop_perc = config['feat_drop_perc']
        
            
        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        self.physics_model = Physics(config)

        # set time indices for integration
        self.dt_intg = self.dt / float(self.intg_lev)
        self.len_intg = (self.dim_t - 1) * self.intg_lev + 1
        self.register_buffer('t_intg', torch.linspace(0.0, self.dt_intg*(self.len_intg-1), self.len_intg))


    def priors(self, n:int, device:torch.device):
        prior_z_phy_stat = {'mean': torch.zeros(n, max(0,self.dim_z_phy), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_phy), device=device)}
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}
        return prior_z_phy_stat, prior_z_aux2_stat


    def generate_physonly(self, z_phy:torch.Tensor, init_yy:torch.Tensor):
        n = z_phy.shape[0]
        device = z_phy.device

        # define ODE
        def ODEfunc(t:torch.Tensor, yy:torch.Tensor):
            return self.physics_model(z_phy, yy)

        # solve ODE
        yy_seq = odeint(ODEfunc, init_yy, self.t_intg, method=self.ode_solver) # <len_intg x n x 2dim_y>
        y_seq = yy_seq[range(0, self.len_intg, self.intg_lev), :, 0:self.dim_y].permute(1,2,0).contiguous() # subsample, extract and reshape to <n x dim_y x dim_t>
        return y_seq


    def decode(self, z_phy:torch.Tensor, z_aux2:torch.Tensor, init_yy:torch.Tensor, full:bool=False):
        n = z_phy.shape[0]
        device = z_phy.device

        # physics part
        if not self.no_phy:
            y_seq_P = self.generate_physonly(z_phy, init_yy)
        else:
            y_seq_P = init_yy[:, 0:self.dim_y].unsqueeze(2).repeat(1, 1, self.dim_t) # (n, dim_y, dim_t)
        
        #x_P = self.dec.func_aux2_map(y_seq_P.permute(0,2,1)).permute(0,2,1).contiguous()
        x_P = y_seq_P
        # out-ODE auxiliary part (y_seq, z_aux2 --> x)
        if self.dim_z_aux2 >= 0:
            x_PB = x_P + self.dec.func_aux2_res(z_aux2).reshape(-1, self.dim_x, self.dim_t)
        else:
            x_PB = x_P.clone()

        if full:
            return x_PB, x_P, self.dec.param_x_lnvar, y_seq_P
        else:
            return x_PB, self.dec.param_x_lnvar


    def encode(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_x, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        feature = self.enc.func_feat(x_)
        
        # infer z_aux2
        if self.dim_z_aux2 > 0:
            feature_aux2 = self.enc.func_feat_aux2(x_)
            if (self.drop_feat and self.train()):
                sam= torch.randperm(feature_aux2.shape[0])[:int(self.sample_drop_perc*feature_aux2.shape[0])]
                fea= torch.randperm(feature_aux2.shape[1])[:int(self.feat_drop_perc*feature_aux2.shape[1])]
                for i in sam:
                    for j in fea:
                        feature_aux2[i][j] = 0


            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature_aux2), 'lnvar':self.enc.func_z_aux2_lnvar(feature_aux2)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}
            
        # infer z_phy
        if ((not self.no_phy) and (self.dim_z_phy > 0)):
            # unmixing
            x_unmixer =torch.cat((x_.view(n,self.dim_y*self.dim_t), z_aux2_stat['mean']), 1)           
            unmixed = x_.view(n, self.dim_y*self.dim_t).clone() + self.enc.func_unmixer_res(x_unmixer) 

            # after unmixing
            feature_phy = self.enc.func_feat_phy(unmixed)
            if (self.drop_feat and self.train()):
                sam= torch.randperm(feature_phy.shape[0])[:int(self.sample_drop_perc*feature_phy.shape[0])]
                fea= torch.randperm(feature_phy.shape[1])[:int(self.feat_drop_perc*feature_phy.shape[1])]
                for i in sam:
                    for j in fea:
                        feature_phy[i][j] = 0

            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature_phy), 'lnvar': self.enc.func_z_phy_lnvar(feature_phy)}
        else:
            unmixed = torch.zeros(n, self.dim_t, device=device)
            z_phy_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        init_yy = self.enc.func_init_yy(feature)
        return z_phy_stat, z_aux2_stat, init_yy, unmixed


    def draw(self, z_phy_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        if not hard_z:
            z_phy = utils.draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
            z_aux2 = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
            
        else:
            z_phy = z_phy_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        return z_phy, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        z_phy_stat, z_aux2_stat, init_yy, _ = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux2_stat

        # draw & reconstruction
        x_mean, x_lnvar = self.decode(*self.draw(z_phy_stat, z_aux2_stat, hard_z=hard_z), init_yy, full=False)

        return z_phy_stat, z_aux2_stat, x_mean, x_lnvar


####

class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x)**2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        zk = zk.unsqueeze(2)
        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w**2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)
        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        return z, log_det_jacobian


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, config:dict):
        super(PlanarVAE, self).__init__(config)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = Planar
        self.num_units_feat = config['num_units_feat']
        self.attention_aux = config['attention_aux']
        self.attention_phy = config['attention_phy']
        self.num_flows_aux = config['num_flows_aux']
        self.num_flows_phy= config['num_flows_phy']
        self.nf_aux = config['nf_aux']
        self.nf_phy = config['nf_phy']
        # Initialize attention layer
        if (self.dim_z_phy >0):
          self.attention_layer_z_p= SelfAttention(self.dim_z_phy)
        if (self.dim_z_aux2 > 0):
          self.attention_layer_z_aux2= SelfAttention(self.dim_z_aux2)

        if ((self.dim_z_aux2 > 0) and self.nf_aux):
            # Amortized flow parameters for z_aux2
            self.amor_u = nn.Linear(self.num_units_feat, self.num_flows_aux * self.dim_z_aux2)
            self.amor_w = nn.Linear(self.num_units_feat, self.num_flows_aux * self.dim_z_aux2)
            self.amor_b = nn.Linear(self.num_units_feat, self.num_flows_aux)
        else:
            self.amor_u = nn.Linear(self.num_units_feat, self.num_flows_aux * 0)
            self.amor_w = nn.Linear(self.num_units_feat, self.num_flows_aux * 0)
            self.amor_b = nn.Linear(self.num_units_feat, self.num_flows_aux)

        if ((self.dim_z_phy >0) and self.nf_phy): 
            # Amortized flow parameters for z_phy
            self.amor_u_p = nn.Linear(self.num_units_feat, self.num_flows_phy * self.dim_z_phy)
            self.amor_w_p = nn.Linear(self.num_units_feat, self.num_flows_phy * self.dim_z_phy)
            self.amor_b_p = nn.Linear(self.num_units_feat, self.num_flows_phy)
        else:
            self.amor_u_p = nn.Linear(self.num_units_feat, self.num_flows_phy * 0)
            self.amor_w_p = nn.Linear(self.num_units_feat, self.num_flows_phy * 0)
            self.amor_b_p = nn.Linear(self.num_units_feat, self.num_flows_phy)
        
        # Normalizing flow layers for z_aux2
        for k in range(self.num_flows_aux):
            flow_k = flow()
            self.add_module('flow_z_' + str(k), flow_k)

        # Normalizing flow layers for z_phy
        for p in range(self.num_flows_phy):
            flow_p = flow()
            self.add_module('flow_p_' + str(p), flow_p)    


    
    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        x_ = x.view(-1, self.dim_x, self.dim_t)
        n = x_.shape[0]
        device = x_.device
        
        feature = self.enc.func_feat(x_)

        if (self.dim_z_aux2 > 0):
            feature_z_aux2 = self.enc.func_feat_aux2(x_)

            if (self.drop_feat and self.train()):
                sam= torch.randperm(feature_z_aux2.shape[0])[:int(self.sample_drop_perc*feature_z_aux2.shape[0])]
                fea= torch.randperm(feature_z_aux2.shape[1])[:int(self.feat_drop_perc*feature_z_aux2.shape[1])]
                for i in sam:
                    for j in fea:
                        feature_z_aux2[i][j] = 0

            """
            if (self.attention_aux):
                attentive_feature_z_aux2 = self.attention_layer(feature_z_aux2)
                feature_z_aux2 = torch.add(feature,torch.mul(feature,attentive_feature_z_aux2))
                #feature_z_aux2 = torch.add(feature,attentive_feature_z_aux2)
                #feature_z_aux2 = torch.mul(feature,attentive_feature_z_aux2)
            """
        # infer z_aux2
            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature_z_aux2), 'lnvar':self.enc.func_z_aux2_lnvar(feature_z_aux2)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer z_phy
        if ((not self.no_phy) and (self.dim_z_phy > 0)):
            # unmixing
            x_unmixer =torch.cat((x_.view(n,self.dim_y*self.dim_t), z_aux2_stat['mean']), 1)           
            unmixed = x_.view(n, self.dim_y*self.dim_t).clone() + self.enc.func_unmixer_res(x_unmixer) 

            # after unmixing
            feature_phy = self.enc.func_feat_phy(unmixed)
            if (self.drop_feat and self.train()):
                sam= torch.randperm(feature_phy.shape[0])[:int(self.sample_drop_perc*feature_phy.shape[0])]
                fea= torch.randperm(feature_phy.shape[1])[:int(self.feat_drop_perc*feature_phy.shape[1])]
                for i in sam:
                    for j in fea:
                        feature_phy[i][j] = 0

            """
            if (self.attention_phy):
                attentive_feature_phy = self.attention_layer(feature_phy)
                feature_phy = torch.add(feature_phy,torch.mul(feature_phy,attentive_feature_phy))
                #feature_phy = torch.add(feature_phy, attentive_feature_phy)
                #feature_phy = torch.mul(feature_phy,attentive_feature_phy)
            """
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature_phy), 'lnvar': self.enc.func_z_phy_lnvar(feature_phy)}
        else:
            unmixed = torch.zeros(n, self.dim_t, device=device)
            z_phy_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        init_yy = self.enc.func_init_yy(feature)

        if ((self.dim_z_aux2 > 0) and self.nf_aux):
            # return amortized u an w for all flows of z_aux2
            u = self.amor_u(feature_z_aux2).view(n, self.num_flows_aux, self.dim_z_aux2, 1)
            w = self.amor_w(feature_z_aux2).view(n, self.num_flows_aux, 1, self.dim_z_aux2)
            b = self.amor_b(feature_z_aux2).view(n, self.num_flows_aux, 1, 1)
        else:
            u = torch.empty(n, 0, device=device)
            w = torch.empty(n, 0, device=device)
            b = torch.empty(n, 0, device=device)
        
        if ((not self.no_phy) and (self.dim_z_phy > 0) and self.nf_phy):
            # return amortized u an w for all flows of z_phy
            u_p = self.amor_u_p(feature_phy).view(n, self.num_flows_phy, self.dim_z_phy, 1)
            w_p = self.amor_w_p(feature_phy).view(n, self.num_flows_phy, 1, self.dim_z_phy)
            b_p = self.amor_b_p(feature_phy).view(n, self.num_flows_phy, 1, 1)
        else:
            u_p = torch.empty(n, 0, device=device)
            w_p = torch.empty(n, 0, device=device)
            b_p = torch.empty(n, 0, device=device)
        return z_phy_stat, z_aux2_stat, init_yy, unmixed, u, w, b, u_p, w_p, b_p


    def draw(self, u, w, b, u_p, w_p, b_p, z_phy_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        
        if not hard_z:
            if ((not self.no_phy) and (self.dim_z_phy > 0) and self.nf_phy):
                z_p = utils.draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
                z_p = z_p.unsqueeze(0)
                for k in range(self.num_flows_phy):
                    flow_p_k = getattr(self, 'flow_p_' + str(k))
                    z_k, log_det_jacobian = flow_p_k(z_p[k], u_p[:, k, :, :], w_p[:, k, :, :], b_p[:, k, :, :])
                    z_k = z_k.unsqueeze(0)
                    z_p = torch.cat((z_p,z_k),0)
                    self.log_det_j += log_det_jacobian

                z_phy = z_p[-1]
                if (self.attention_phy):
                    attentive_z_phy = self.attention_layer_z_p(z_phy)
                    z_phy = torch.add(z_phy,torch.mul(z_phy,attentive_z_phy))
            else:
                z_phy = utils.draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])

            if ((self.dim_z_aux2 > 0) and self.nf_aux):
                z = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
                z = z.unsqueeze(0)
                for k in range(self.num_flows_aux):
                    flow_z_k = getattr(self, 'flow_z_' + str(k))
                    z_k, log_det_jacobian = flow_z_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                    z_k = z_k.unsqueeze(0)
                    z = torch.cat((z,z_k),0)
                    self.log_det_j += log_det_jacobian

                z_aux2 = z[-1]
                if (self.attention_aux):
                    attentive_z_aux2 = self.attention_layer_z_aux2(z_aux2)
                    z_aux2 = torch.add(z_aux2,torch.mul(z_aux2,attentive_z_aux2))
            else:
                z_aux2 = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])    
        else:
            z_phy = z_phy_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        return z_phy, z_aux2
        

    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        z_phy_stat, z_aux2_stat, init_yy, unmixed, u, w, b, u_p, w_p, b_p = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux2_stat

        # draw & reconstruction
        
        x_mean, x_lnvar = self.decode(*self.draw(u, w, b, u_p, w_p, b_p, z_phy_stat, z_aux2_stat, hard_z=hard_z), init_yy, full=False)

        return z_phy_stat, z_aux2_stat, x_mean, x_lnvar
    

    

