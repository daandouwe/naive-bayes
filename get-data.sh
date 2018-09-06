#!/usr/bin/env bash

mkdir -p data/classes
mkdir -p data/binary

wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt
wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt
wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt

mv train.txt dev.txt test.txt data/classes

python make-binary-classes.py data/classes/train.txt data/binary/train.txt
python make-binary-classes.py data/classes/dev.txt data/binary/dev.txt
python make-binary-classes.py data/classes/test.txt data/binary/test.txt
