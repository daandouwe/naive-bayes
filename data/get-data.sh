#!/usr/bin/env bash

mkdir classes binary

wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt
wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt
wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt

mv train.txt dev.txt test.txt classes

python make-binary-classes.py classes/train.txt binary/train.txt
python make-binary-classes.py classes/dev.txt binary/dev.txt
python make-binary-classes.py classes/test.txt binary/test.txt
