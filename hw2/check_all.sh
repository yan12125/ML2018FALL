#!/bin/bash
rm -vf model*.npz submission*
python3.6 ./train.py
python3.6 ./train_best.py
./hw2.sh data/train_x.csv data/train_y.csv data/test_x.csv submission-generative.txt
./hw2_best.sh data/train_x.csv data/train_y.csv data/test_x.csv submission-logistic.txt
