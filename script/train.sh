#!/usr/bin/env bash

python main.py \
--dataset "Citeseer" \
--eval_step 5 \
--device "cuda:0" \
--experiment_name "sbm_gnn_pytorch"
