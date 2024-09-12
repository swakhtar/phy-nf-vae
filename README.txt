## Note
This code is an implementation of the following paper:
S.W. Akhtar. Physics-integrated generative modeling using attentive planar normalizing flow based variational autoencoder. arXiv:2404.12267 [cs.LG], 2024.

The Code is built on top of https://github.com/n-takeishi/phys-vae which implements: 
Naoya Takeishi and Alexandros Kalousis. Physics-integrated variational autoencoders for robust and interpretable generative modeling. In Advances in Neural Information Processing Systems 34 (NeurIPS), 2021.

If you use this code, please cite both of the above papers.

## Prerequisite

- Python 3.8.3
- NumPy 1.19.2
- SciPy 1.5.2
- PyTorch 1.7.0
- torchdiffeq 

## Usage

->First download original locomotion dataset (see corresponding readme.txt in "data" folder)
->Run 'makedata.m' in data directory using MATLAB. This will create "data_test.mat", "data_train.mat" and "data_valid.mat" in the same folder.  

-> Change the working directory to the folder
cd /path/to/phy-nf-vae  

->You can train & test benchmark models using script locomotion.sh, for example by 

. locomotion.sh physnn
. locomotion.sh nnonly
. locomotion.sh physonly

and the proposed models by

. locomotion.sh nf
. locomotion.sh attnf

->After you have trained all of the above five models, you can run locomotion_reconstruct.ipynb file to perform reconstruction experiments to compare their performance, as shown in the paper.
