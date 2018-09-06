#!/usr/bin/env python
import argparse
import os
import string
from collections import Counter

import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import precision_score

from plot import confusion_matrix

STOP_WORDS = set(stopwords.words('english'))

PUNCT = {
    '.', ',', '-', '--', ';', ':',  # want to keep `!` and `?` because these are semantic
    "``", "'", '`', '""', "''",
    '...',  '-lrb-', '-rrb-'
    }

EPS = 1e-45


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



class NaiveBayesClassifier:
    def __init__(self):
        pass

    def __call__(self, words):
        assert isinstance(words, list)
        logprobs = dict((label, self.class_probs[label]) for label in self.classes)
        for word in words:
            for label in self.classes:
                prob = self.prob(word, label)
                logprobs[label] += np.log(prob)
        return logprobs

    def inference(self, data):
        self.classes = sorted(set(label for label, _ in data))
        self.num_classes = len(self.classes)

        # Estimate class p(c).
        class_counts = Counter(label for label, _ in data)
        total = sum(class_counts.values())
        self.class_probs = dict((label, count / total) for label, count in class_counts.items())

        # Estimate p(x|c)
        class_sentences = dict((label, Counter()) for label in self.classes)
        for label, sentence in data:
            class_sentences[label].update(sentence)
        self.conditional_probs = dict((label, dict()) for label in self.classes)
        for label in self.classes:
            word_counts = class_sentences[label]
            total = sum(word_counts.values())
            for word, count in word_counts.items():
                self.conditional_probs[label][word] = count / total

    def predict(self, words):
        logprobs = self(words)
        label, logprob = Counter(logprobs).most_common()[0]
        return label, logprob

    def prob(self, word, label):
        assert label in self.classes, f'unknown class label {label}'
        cond_prob = self.conditional_probs[label].get(word, EPS)
        return cond_prob

    def accuracy(self, pred, gold):
        return precision_score(gold, pred, average=None)

    def top(self, n=10):
        tops = dict()
        for label in self.classes:
            top = Counter(self.conditional_probs[label]).most_common(n)
            tops[label] = top
        return tops

def main(args):
    train_data = get_data(os.path.join(args.data, 'train.txt'),
                          remove_stopwords=args.no_stop, remove_punct=args.no_punct)
    dev_data = get_data(os.path.join(args.data, 'dev.txt'),
                          remove_stopwords=args.no_stop, remove_punct=args.no_punct)
    test_data = get_data(os.path.join(args.data, 'test.txt'),
                          remove_stopwords=args.no_stop, remove_punct=args.no_punct)

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
    print('Accuracy:')
    print('label  acc')
    print('\n'.join(f'  {label}    {100*acc:.2f}' for label, acc in enumerate(accuracy)))

    confusion_matrix(dev_labels, predicted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/binary')
    parser.add_argument('--no-stop', action='store_true')
    parser.add_argument('--no-punct', action='store_true')
    parser.add_argument('-n', '--num-lines', type=int, default=30)
    args = parser.parse_args()

    main(args)
