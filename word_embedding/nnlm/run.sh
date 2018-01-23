#!/usr/bin/env bash


python nnlm.py --input_file data/train.en \
               --batch_size 100 \
               --win_size 2 \
               --hidden_num 256 \
               --num_epochs 10 \
               --word_dim 256










