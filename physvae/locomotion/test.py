import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from .model import VAE,PlanarVAE
from .. import utils
import argparse
from torchdiffeq import odeint

def test_parser():
   parser = argparse.ArgumentParser(description='')
   # testmodel
   parser.add_argument('--test_model', type=str, choices=['nnonly', 'physonly', 'physnn', 'nf', 'attnf'] ,required=True)
   return parser


parser = test_parser()
args = parser.parse_args()
method = args.test_model

# load data
datadir = './data'
dataname = 'test'
data_test = sio.loadmat('{}/data_{}.mat'.format(datadir, dataname))['data'].astype(np.float32)
_, dim_x, dim_t = data_test.shape


# load training args as dict
dim_y = 3
modeldir = './out_locomotion'

with open('{}/args_{}.json'.format(modeldir,method), 'r') as f:
    args_tr_dict = json.load(f)

# set model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args_tr_dict['flow']=='no_flow':
  model = VAE(args_tr_dict).to(device)
elif args_tr_dict['flow']=='planar':  
  model = PlanarVAE(args_tr_dict).to(device)

# load model
model.load_state_dict(torch.load('{}/model_{}.pt'.format(modeldir, method), map_location=device))
print('model loaded')

# inference & reconstruction on test data, and compute full trajectories (including generalized momenta)
model.eval()
data_test_tensor = torch.Tensor(data_test).to(device).contiguous()
z_phy_stat, z_aux2_stat, x, _ = model(data_test_tensor)


# Loss Function
def loss_function(args_tr_dict, data, z_phy_stat, z_aux2_stat, x):
    n = data.shape[0]
    device = data.device

    recerr_sq = torch.sum((x - data).pow(2), dim=[1,2]).mean()

    prior_z_phy_stat, prior_z_aux2_stat = model.priors(n, device)

    KL_z_aux2 = utils.kldiv_normal_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'],
        prior_z_aux2_stat['mean'], prior_z_aux2_stat['lnvar']) if args_tr_dict['dim_z_aux2'] > 0 else torch.zeros(1, device=device)
    KL_z_phy = utils.kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
        prior_z_phy_stat['mean'], prior_z_phy_stat['lnvar']) if not args_tr_dict['no_phy'] else torch.zeros(1, device=device)

    kldiv = (KL_z_aux2 + KL_z_phy).mean()

    return recerr_sq, kldiv

#Loss Computation and logging
log_test = {'recerr_sq':.0, 'kldiv':0.}
recerr_sq, kldiv = loss_function(args_tr_dict, data_test_tensor, z_phy_stat, z_aux2_stat, x)
log_test['recerr_sq']= recerr_sq.detach()
log_test['kldiv'] += kldiv.detach()
print('====> Test (rec. err.)^2: {:.4f}  kldiv: {:.4f}'.format(log_test['recerr_sq'], log_test['kldiv']))
with open('{}/log_test.txt'.format(args_tr_dict['outdir']), 'w') as f:
    print('Test_recerr_sq: {:.7e}  Test_kl_div: {:.7e}'.format(log_test['recerr_sq'], log_test['kldiv']), file=f)


# plot
idx=0
dat = data_test[idx].T
# reg
plt.figure()
plt.plot(dat)
plt.plot(x[idx].detach().cpu().numpy().T, 'k--')
plt.savefig('{}/recon.png'.format(args_tr_dict['outdir']))
plt.show()    