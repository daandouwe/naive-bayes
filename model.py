from collections import Counter

import numpy as np
from sklearn.metrics import precision_score

from utils import EPS


class NaiveBayesClassifier:
    """A Naive Bayes text classifier."""
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
