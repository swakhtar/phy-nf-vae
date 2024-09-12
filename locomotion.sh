#!/bin/bash

# architecture
phy="--dt 0.02 --hidlayers-H 128 128"
dec="--hidlayers-aux2-dec 512 512 --x-lnvar -13.8"
feat="--arch-feat mlp --hidlayers-feat 512 512 --num-units-feat 512"
unmix="--hidlayers-unmixer 512 512"
enc="--hidlayers-init-yy 64 32 --hidlayers-aux2-enc 64 32"

# optimization
optim="--learning-rate 1e-3 --train-size 400 --batch-size 100 --epochs 2 --grad-clip 5.0 --intg-lev 1 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 1.0"

# other options
others="--save-interval 9999999 --num-workers 0 --activation elu" #--cuda


# ------------------------------------------------
#feature dropout (to incorporate noise in the model)
drop="--drop-feat --sample-drop-perc 0.1 --feat-drop-perc 0.1"
flow="--flow planar"
att_flow_aux_params="--num_flows_aux 12 --attention_aux --nf_aux"
att_flow_phy_params="--num_flows_phy 10 --attention_phy --nf_phy" 
flow_aux_params="--num_flows_aux 12 --nf_aux"    
flow_phy_params="--num_flows_phy 5 --nf_phy"

outdir="./out_locomotion/"
datadir="./data/"
options1="--datadir ${datadir} --outdir ${outdir} ${phy} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others}"
options2="--datadir ${datadir} --outdir ${outdir} ${phy} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others} ${flow} ${flow_aux_params} ${flow_phy_params} ${drop}"
options3="--datadir ${datadir} --outdir ${outdir} ${phy} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others} ${flow} ${att_flow_aux_params} ${att_flow_phy_params} ${drop}"

dimz_ph=2; dimy=3; dimz=15
#alpha=1e-2; beta=1e-1; gamma=1e-3
alpha=1e-2; beta=1e-1; gamma=1e-3

testmodel="--test_model $1" 

if [ "$1" = "physonly" ]; then
    # Physics-only
    commands="${options1} --dim-y ${dimy} --dim-z-phy ${dimz_ph} --dim-z-aux2 -1 --balance-dataug ${beta} --balance-unmix ${gamma}"
elif [ "$1" = "nnonly" ]; then
    # NN-only
    commands=" ${options1} --dim-y ${dimy} --dim-z-aux2 ${dimz} --no_phy"  
elif [ "$1" = "physnn" ]; then
    # Phys+VAE  
    commands=" ${options1} --dim-y ${dimy} --dim-z-phy ${dimz_ph} --dim-z-aux2 ${dimz} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha} --balance-dataug ${beta} --balance-unmix ${gamma}"
elif [ "$1" = "nf" ]; then
    #Phys+NFVAE
    commands = "${options2} --dim-y ${dimy} --dim-z-phy ${dimz_ph} --dim-z-aux2 ${dimz} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha} --balance-dataug ${beta} --balance-unmix ${gamma}"
elif [ "$1" = "attnf" ]; then
    #Phys+attentive+NFVAE
    commands = "${options3} --dim-y ${dimy} --dim-z-phy ${dimz_ph} --dim-z-aux2 ${dimz} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha} --balance-dataug ${beta} --balance-unmix ${gamma}"
else
    echo "unknown option"
    commands=""
fi


if [[ -d "${outdir}" ]]; then
    echo "Output Directory Already exists: ${outdir}"
    echo "Removing ${outdir}"
    rm -r ${outdir}
    echo "Creating Output Directory.."
    mkdir ${outdir}
else
    echo "Creating Output Directory.."
    mkdir ${outdir}
fi

python -m physvae.locomotion.train ${commands} 
python -m physvae.locomotion.test ${testmodel}
