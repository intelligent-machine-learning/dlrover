#!/bin/bash
python train.py --model_type toy \
    2>&1 | tee log_toy_fully_automatic.txt 