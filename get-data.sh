#!/usr/bin/env bash

mkdir -p data

python make-binary-classes.py ~/data/neubig-data/classes/train.txt data/train.txt
python make-binary-classes.py ~/data/neubig-data/classes/dev.txt data/dev.txt
python make-binary-classes.py ~/data/neubig-data/classes/test.txt data/test.txt
