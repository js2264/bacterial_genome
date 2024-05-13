#!/bin/bash

# Nucleosome model
model_name=model_myco_nuc_2
# python Train_profile.py \
#     -arch 'mnase_Etienne' \
#     -g genome/W303_Mmmyco.npz \
#     -l data/labels_myco_nuc.npz \
#     -out Trainedmodels/$model_name/ \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII \
#     -cv XIV XV \
#     -w 2001 -b 1024 \
#     -dist -v 2 -mt 262144 -bal batch
python predict_profile.py \
    -m Trainedmodels/$model_name/model \
    -g genome/W303_Mmmyco.npz \
    -o results/$model_name/ \
    -b 1024

# # Cohesin model
# model_name=model_myco_coh_14
# python Train_profile.py \
#     -arch 'bassenji_Etienne' \
#     -g genome/W303_Mmmyco.npz \
#     -l data/GSE217022/labels_myco_coh_ratio.npz \
#     -r data/GSE217022/invalid_myco_coh_ratio.npz \
#     -out Trainedmodels/$model_name/ \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -w 32768 -h_int 128 -b 512 \
#     -dist -v 2 -mt 262144 -bal batch
# python predict_profile.py \
#     -m Trainedmodels/$model_name/model \
#     -g genome/W303_Mmmyco.npz \
#     -o results/$model_name/ \
#     -b 512 -h_int 128

# # Polymerase model
model_name=model_myco_pol_17
# python Train_profile.py \
#     -arch 'bassenji_Etienne' \
#     -g genome/W303_Mmmyco.npz \
#     -l data/GSE217022/labels_myco_pol_ratio.npz \
#     -r data/GSE217022/invalid_myco_pol_ratio.npz \
#     -out Trainedmodels/$model_name/ \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII \
#     -cv XIV XV \
#     -w 2048 -h_int 128 -b 1024 \
#     -dist -v 2 -mt 262144 -bal batch \
#     -rN
python predict_profile.py \
    -m Trainedmodels/$model_name/model \
    -g genome/W303_Mmmyco.npz \
    -o results/$model_name/ \
    -b 1024 -h_int 128 -mid


