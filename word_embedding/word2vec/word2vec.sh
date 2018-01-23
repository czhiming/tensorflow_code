#!/bin/sh

python word2vec.py --data_dir datasets \
				   --batch_size 64 \
				   --win_size 10 \
				   --word_dim 100 \
				   --neg_size 10 \
				   --num_epochs 50 




