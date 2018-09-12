#!/usr/bin/env python
import argparse
import os
import string
from collections import Counter

import numpy as np

from model import NaiveBayesClassifier
from plot import confusion_matrix
from utils import STOP_WORDS, PUNCT


def get_data(path, remove_stopwords=False, remove_punct=False):
    path = os.path.expanduser(path)
    data = []
    with open(path) as f:
        for line in f.readlines():
            label, sentence = line.strip().split('|||')
            sentence = [word.lower() for word in sentence.split()]
            if remove_stopwords:
                sentence = [word for word in sentence if word not in STOP_WORDS]
            if remove_punct:
                sentence = [word for word in sentence if word not in PUNCT]
            data.append((label, sentence))
    return data


def main(args):
    train_data = get_data(os.path.join(args.data, 'train.txt'),
                          remove_stopwords=args.no_stop,
                          remove_punct=args.no_punct)
    dev_data = get_data(os.path.join(args.data, 'dev.txt'),
                          remove_stopwords=args.no_stop,
                          remove_punct=args.no_punct)
    test_data = get_data(os.path.join(args.data, 'test.txt'),
                          remove_stopwords=args.no_stop,
                          remove_punct=args.no_punct)

    classes = sorted(set(int(label) for label, _ in train_data))

    classes_data = {i: [] for i in range(len(classes))}
    for label, data in train_data:
        classes_data[int(label)].extend(data)
    top_words = dict()
    for i in range(len(classes)):
        top_words[i] = Counter(classes_data[i]).most_common(args.remove)
    for i, most_common in top_words.items():
        top_words[i] = set(word for word, _ in most_common)
    remove = set(top_words[0])
    for tops in top_words.values():
        remove = remove & tops
    print(f'Removed from data: {remove}')

    # all_categories = sum([line for _, line in train_data], [])
    # remove = Counter(all_categories).most_common(args.remove)
    # remove, _ = zip(*remove)
    # train_data = [
    #     (label, [word for word in line if word not in remove])
    #         for label, line in train_data]

    model = NaiveBayesClassifier()
    model.inference(train_data)

    dev_labels, dev_sentences = zip(*dev_data)
    predicted = []
    for sentence in dev_sentences:
        pred, logprob = model.predict(sentence)
        predicted.append(pred)

    tops = model.top(args.num_lines)
    for label in tops:
        print('class', label, f'({model.class_probs[label]:.3f})')
        print('\n'.join((f'  {word:<12} {prob:8.5f}' for word, prob in tops[label])))
        print()

    accuracy = model.accuracy(dev_labels, predicted)
    print('label  accuracy')
    print('\n'.join(f'  {label}      {100*acc:.2f}' for label, acc in enumerate(accuracy)))

    confusion_matrix(dev_labels, predicted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/binary')
    parser.add_argument('--no-stop', action='store_true')
    parser.add_argument('--no-punct', action='store_true')
    parser.add_argument('--remove', type=int, default=5, help='remove top n most frequent words in both cospora')
    parser.add_argument('-n', '--num-lines', type=int, default=30)
    args = parser.parse_args()

    main(args)
