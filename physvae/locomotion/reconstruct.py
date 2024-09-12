import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torchdiffeq import odeint
from physvae import utils
from physvae.locomotion.model import VAE, PlanarVAE

# load data
datadir = './data/locomotion/'
dataname = 'test'
data_test = sio.loadmat('{}/data_{}.mat'.format(datadir, dataname))['data'].astype(np.float32)
print(data_test.shape)
_, dim_x, dim_t = data_test.shape

index=0
method = [ 'physnn', 'att-nf', 'physonly','nnonly', 'nf']
ls=[(0, (3, 5, 1, 5, 1, 5)), (0, (1, 1)),'--', '-.' , ':',]
idx=1
dat = data_test[idx].T
plt.figure()
plt.figure(figsize=(10,8))
ax1=plt.subplot(3,1,1)
plt.plot(dat[:,0], 'k--', label = 'True')
ax2=plt.subplot(3,1,2)
plt.plot(dat[:,1], 'k--', label = 'True')
ax3= plt.subplot(3,1,3)
plt.plot(dat[:,2], 'k--', label= 'True')

for m in method:
    # load training args as dict
    dim_y = 3
    modeldir = './out_locomotion/'

    with open('{}/args_{}.json'.format(modeldir,m), 'r') as f:
        args_tr_dict = json.load(f)

    # set model
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model
    if args_tr_dict['flow']=='no_flow':
      model = VAE(args_tr_dict).to(device)
    elif args_tr_dict['flow']=='planar':  
      model = PlanarVAE(args_tr_dict).to(device)

    # load model
    model.load_state_dict(torch.load('{}/model_{}.pt'.format(modeldir,m), map_location=device))
    print('model {} loaded'.format(m))
    model.eval()
    data_test_tensor = torch.Tensor(data_test).to(device).contiguous()
    z_phy_stat, z_aux2_stat, x, _ = model(data_test_tensor)

    # plot
    y = x[idx].T
    ax1=plt.subplot(3,1,1)
    plt.plot(y[:,0].detach().cpu().numpy(), label=m, linestyle=ls[index])
    
    ax2= plt.subplot(3,1,2)
    plt.plot(y[:,1].detach().cpu().numpy(), label=m, linestyle = ls[index])
    
    ax3= plt.subplot(3,1,3)
    plt.plot(y[:,2].detach().cpu().numpy(), label=m, linestyle = ls[index])
    index +=1;

ax1.set_ylabel('Angle of hip')
ax2.set_ylabel('Angle of knee')
ax3.set_ylabel('Angle of ankle')
ax3.set_xlabel('Normalized stride duration')
plt.legend(loc='best', mode="expand", ncol = 6, bbox_to_anchor = (0, 0, 1, 1))
plt.savefig('{}/recon.png'.format(args_tr_dict['outdir']))
plt.show()
