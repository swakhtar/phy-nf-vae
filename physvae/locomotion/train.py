import argparse
import os
import json
import time
import numpy as np
import scipy.io as sio

import torch
from torch import optim
import torch.utils.data
import sys
from .model import VAE,PlanarVAE
from .. import utils
# from model import VAE
# import sys; sys.path.append('../'); import utils


def set_parser():
    parser = argparse.ArgumentParser(description='')

    # input/output setting
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--dataname-train', type=str, default='train')
    parser.add_argument('--dataname-valid', type=str, default='valid')

    # model (general)
    parser.add_argument('--hidlayers-H', type=int, nargs='+', default=[128,])
    parser.add_argument('--dim-y', type=int, required=True, help="must be positive")
    parser.add_argument('--dim-z-aux2', type=int, required=True, help="if 0, aux2 is still alive without latent variable; set -1 to deactivate")
    parser.add_argument('--dim-z-phy', type=int, default=0)
    parser.add_argument('--activation', type=str, default='elu') #choices=['relu','leakyrelu','elu','softplus','prelu'],
    parser.add_argument('--ode-solver', type=str, default='euler')
    parser.add_argument('--intg-lev', type=int, default=1)
    parser.add_argument('--no_phy', action='store_true', default=False)
    parser.add_argument('--flow', type=str, default='no_flow', choices=['planar', 'no_flow'], help="""Type of flows to use, no flows can also be selected""")
    parser.add_argument('--num_flows_aux', type=int, default=4)
    parser.add_argument('--num_flows_phy', type=int, default=4)
    parser.add_argument('--attention_aux', action='store_true', default=False)
    parser.add_argument('--attention_phy', action='store_true', default=False)
    parser.add_argument('--nf_aux', action='store_true', default=False)
    parser.add_argument('--nf_phy', action='store_true', default=False)

    # model (decoder)
    parser.add_argument('--x-lnvar', type=float, default=-10.0)
    parser.add_argument('--hidlayers-aux2-dec', type=int, nargs='+', default=[128,])

    # model (encoder)
    parser.add_argument('--hidlayers-init-yy', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-aux2-enc', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-unmixer', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-z-phy', type=int, nargs='+', default=[128])
    parser.add_argument('--arch-feat', type=str, default='mlp')
    parser.add_argument('--num-units-feat', type=int, default=256)
    parser.add_argument('--hidlayers-feat', type=int, nargs='+', default=[256,])
    parser.add_argument('--num-rnns-feat', type=int, default=1)

    # optimization (base)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--adam-eps', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--balance-kld', type=float, default=1.0)
    parser.add_argument('--balance-unmix', type=float, default=0.0)
    parser.add_argument('--balance-dataug', type=float, default=0.0)
    parser.add_argument('--balance-lact-dec', type=float, default=0.0)
    parser.add_argument('--balance-lact-enc', type=float, default=0.0)

    # dropout (noise)
    parser.add_argument('--drop-feat', action='store_true', default=False)
    parser.add_argument('--sample-drop-perc', type=float, default = 0.0)
    parser.add_argument('--feat-drop-perc', type= float, default=0.0)


    # others
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--train-size', type=int, default=-1)
    parser.add_argument('--save-interval', type=int, default=999999999)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234567890)

    return parser


def loss_function(args, data, z_phy_stat, z_aux2_stat, x):
    n = data.shape[0]
    device = data.device

    recerr_sq = torch.sum((x - data).pow(2), dim=[1,2]).mean()

    prior_z_phy_stat, prior_z_aux2_stat = model.priors(n, device)

    KL_z_aux2 = utils.kldiv_normal_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'],
        prior_z_aux2_stat['mean'], prior_z_aux2_stat['lnvar']) if args.dim_z_aux2 > 0 else torch.zeros(1, device=device)
    KL_z_phy = utils.kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
        prior_z_phy_stat['mean'], prior_z_phy_stat['lnvar']) if not args.no_phy else torch.zeros(1, device=device)

    kldiv = (KL_z_aux2 + KL_z_phy).mean()

    return recerr_sq, kldiv


def train(epoch, args, device, loader, model, optimizer):
    model.train()
    logs = {'recerr_sq':.0, 'kldiv':.0, 'unmix':.0, 'dataug':.0, 'lact_dec':.0}

    for batch_idx, (data,) in enumerate(loader):
        data = data.to(device)
        batch_size = len(data)
        optimizer.zero_grad()
        if args.flow == 'no_flow':
        # inference & reconstruction on original data
            z_phy_stat, z_aux2_stat, init_yy, unmixed = model.encode(data)
            z_phy, z_aux2 = model.draw(z_phy_stat, z_aux2_stat, hard_z=False)
            x_PB, x_P, x_lnvar, y_seq_P = model.decode(z_phy, z_aux2, init_yy, full=True)
            x_var = torch.exp(x_lnvar)
        elif args.flow == 'planar':
            z_phy_stat, z_aux2_stat, init_yy, unmixed, u, w, b, u_p, w_p, b_p = model.encode(data)
            # draw & reconstruction
            z_phy, z_aux2 = model.draw(u, w, b, u_p, w_p, b_p, z_phy_stat, z_aux2_stat, hard_z=False)
            x_PB, x_P, x_lnvar, y_seq_P = model.decode(z_phy, z_aux2, init_yy, full=True)
            x_var = torch.exp(x_lnvar)
        # ELBO
        recerr_sq, kldiv = loss_function(args, data, z_phy_stat, z_aux2_stat, x_PB)

        # unmixing regularization
        if not args.no_phy:
            reg_unmix = torch.sum((unmixed - x_P.detach().view(batch_size,-1)).pow(2), dim=1).mean()
        else:
            reg_unmix = torch.zeros(1, device=device).squeeze()

        
        # data augmentation regularization
        if not args.no_phy:
            model.eval()
            #with torch.no_grad():
            #   aug_dcoeff = torch.rand(batch_size, args.dim_z_phy, requires_grad=True, device=device)
            #   aug_x_P = model.generate_physonly(aug_dcoeff, init_yy.detach())
            
            aug_dcoeff = torch.rand(batch_size, args.dim_z_phy, requires_grad=True, device=device)
            aug_x_P = model.generate_physonly(aug_dcoeff, init_yy.detach())
            model.train()
            aug_feature_phy = model.enc.func_feat_phy(aug_x_P.detach())
            aug_infer = model.enc.func_z_phy_mean(aug_feature_phy)
            reg_dataug = (aug_infer - aug_dcoeff ).pow(2).mean()
        else:
            reg_dataug = torch.zeros(1, device=device).squeeze()
        
        # least action principle
        if not args.no_phy:
            reg_lact_dec = torch.sum((x_PB - x_P).pow(2), dim=[1,2]).mean()
        else:
            reg_lact_dec = torch.zeros(1, device=device).squeeze()
        
        # loss function
        kldiv_balanced = (args.balance_kld + args.balance_lact_enc) * kldiv * x_var.detach()
        loss = recerr_sq + kldiv_balanced \
           + args.balance_unmix*reg_unmix + args.balance_dataug*reg_dataug + args.balance_lact_dec*reg_lact_dec


        # update model parameters
        loss.backward()
        if args.grad_clip>0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        # log
        logs['recerr_sq'] += recerr_sq.detach()*batch_size
        logs['kldiv'] += kldiv.detach()*batch_size
        logs['unmix'] += reg_unmix.detach()*batch_size
        logs['dataug'] += reg_dataug.detach()*batch_size
        logs['lact_dec'] += reg_lact_dec.detach()*batch_size


    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Training (rec. err.)^2: {:.4f}  kldiv: {:.4f}  unmix: {:4f}  dataug: {:4f}  lact_dec: {:4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv'], logs['unmix'], logs['dataug'], logs['lact_dec']))
    return logs


def valid(epoch, args, device, loader, model):
    model.eval()
    logs = {'recerr_sq':.0, 'kldiv':.0}
    #with torch.no_grad():
    for i, (data,) in enumerate(loader):
        data = data.to(device)
        batch_size = len(data)
        z_phy_stat, z_aux2_stat, x, _ = model(data)
        recerr_sq, kldiv = loss_function(args, data, z_phy_stat, z_aux2_stat, x)

        logs['recerr_sq'] += recerr_sq.detach()*batch_size
        logs['kldiv'] += kldiv.detach()*batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Validation (rec. err.)^2: {:.4f}  kldiv: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv']))
    return logs


if __name__ == '__main__':

    parser = set_parser()
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")


    # set random seed
    torch.manual_seed(args.seed)

    # load training/validation data
    data_train = sio.loadmat('{}/data_{}.mat'.format(args.datadir, args.dataname_train))['data'].astype(np.float32)
    data_valid = sio.loadmat('{}/data_{}.mat'.format(args.datadir, args.dataname_valid))['data'].astype(np.float32)

    args.dim_x = data_train.shape[1]
    args.dim_t = data_train.shape[2]

    if args.train_size > 0:
        if args.train_size > data_train.shape[0]:
            raise ValueError('train_size must be <= {}'.format(data_train.shape[0]))
        idx = torch.randperm(data_train.shape[0]).numpy()[0:args.train_size]
        data_train = data_train[idx]
    print(data_train.shape)

    # set data loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    loader_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(data_train).float()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    loader_valid = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(data_valid).float()),
        batch_size=args.batch_size, shuffle=False, **kwargs)


    # set model
    if args.flow == 'no_flow':
        model = VAE(vars(args)).to(device)
    elif args.flow == 'planar':
        model = PlanarVAE(vars(args)).to(device)

    # set optimizer
    kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay, 'eps': args.adam_eps}
    optimizer = optim.Adam(model.parameters(), **kwargs)

    if(args.flow == 'no_flow' and (args.dim_z_aux2 < 0)):
        method = 'physonly'
    elif(args.flow == 'no_flow' and (args.no_phy)):
        method = 'nnonly'
    elif(args.flow == 'no_flow' and (args.dim_z_aux2 > 0) and (args.dim_z_phy >0)):
        method = 'physnn'    
    elif(args.flow == 'planar' and args.nf_phy and args.nf_phy and (not args.attention_aux) and (not args.attention_phy)):
        method = 'nf'
    elif(args.flow == 'planar' and args.nf_phy and args.nf_phy and args.attention_aux and args.attention_phy):
        method = 'att-nf' 

    print('start training with device', device)
    #print(vars(args))
    print()


    # save args
    with open('{}/args_{}.json'.format(args.outdir, method), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    # create log files
    with open('{}/log_{}.txt'.format(args.outdir, method), 'w') as f:
        print('# epoch recerr_sq kldiv unmix dataug lact_dec valid_recerr_sq valid_kldiv duration', file=f)


    # main iteration
    info = {'bestvalid_epoch':0, 'bestvalid_recerr':1e10}
    dur_total = .0
    for epoch in range(1, args.epochs + 1):
        # training
        start_time = time.time()
        logs_train = train(epoch, args, device, loader_train, model, optimizer)
        dur_total += time.time() - start_time

        # validation
        logs_valid = valid(epoch, args, device, loader_valid, model)   
              
        # save loss information
        with open('{}/log_{}.txt'.format(args.outdir, method), 'a') as f:
            print('{} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e}'.format(epoch,
                logs_train['recerr_sq'], logs_train['kldiv'], logs_train['lact_dec'],
                logs_valid['recerr_sq'], logs_valid['kldiv'], dur_total), file=f)
        
        # save model if best validation loss is achieved
        if logs_valid['recerr_sq'] < info['bestvalid_recerr']:
            info['bestvalid_epoch'] = epoch
            info['bestvalid_recerr'] = logs_valid['recerr_sq']
            torch.save(model.state_dict(), '{}/model_{}.pt'.format(args.outdir, method))
            print('best model saved')

        # save model at interval
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), '{}/model_e{}.pt'.format(args.outdir, epoch))

        print()

    print()
    print('end training')




