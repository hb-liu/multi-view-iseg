#!/bin/bash

###############################
#    build coseg directory    #
###############################

mkdir coseg/data
mkdir coseg/data/syn
mkdir coseg/data/syn/6m
mkdir coseg/data/syn/12m
mkdir coseg/data/processed
mkdir coseg/data/processed/6m
mkdir coseg/data/processed/12m

mkdir coseg/experiments
mkdir coseg/log
mkdir coseg/tmp

mkdir coseg/results
mkdir coseg/results/pretrain
mkdir coseg/results/finetune

##############################
#    build dcan directory    #
##############################

mkdir dcan/data
mkdir dcan/data/raw
mkdir dcan/data/raw/6m
mkdir dcan/data/raw/12m
mkdir dcan/data/syn
mkdir dcan/data/syn/6m
mkdir dcan/data/syn/12m
mkdir dcan/data/processed
mkdir dcan/data/processed/6m
mkdir dcan/data/processed/12m

mkdir dcan/experiments
mkdir dcan/log
mkdir dcan/tmp