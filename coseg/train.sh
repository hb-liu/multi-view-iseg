#!/bin/bash

##################
#    pretrain    #
##################

# run pretraining
python3 pretrain.py

# save pretrained model and log to results folder
mv experiments/* results/pretrain
mv log/* results/pretrain
mv tmp/* results/pretrain

##################
#    finetune    #
##################

# run finetuning
python3 finetune.py

# save finetuned model and log to results folder
mv experiments/* "results/finetune"
mv log/* "results/finetune"
mv tmp/* "results/finetune"