from __future__ import division
from collections import Counter
import math
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def ngrams(sequence, n):
    """
    borrowed from NLTK
    """

    sequence = iter(sequence)
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def calc_bleu(reference, hypothesis, bp=True):
    """
    BLEU score with NIST geometric sequence smoothing
    """
    ngram_precisions = []
    k = 1
    for n in xrange(1, 5):
        hyp_ngram_counts = Counter(ngrams(hypothesis, n))
        ref_ngram_counts = Counter(ngrams(reference, n))

        num_hits = sum(min(hyp_ngram_count, ref_ngram_counts[hyp_ngram])
                       for hyp_ngram, hyp_ngram_count in hyp_ngram_counts.iteritems())

        if len(hypothesis) >= n:
            ref_ngram_num = len(hypothesis) - n + 1
        else:
            ref_ngram_num = 1

        if num_hits == 0:
            precision = 1. / (2 ** k * ref_ngram_num)
            k += 1
        else:
            precision = num_hits / ref_ngram_num
        ngram_precisions.append(precision)

    mean = (ngram_precisions[0] * ngram_precisions[1] * ngram_precisions[2] * ngram_precisions[3]) ** 0.25

    if bp:
        bp_score = get_bp(reference, hypothesis)
    else:
        bp_score = 1.0

    bleu = bp_score * mean

    return bleu

def calc_f1(reference, hypothesis):
    """
    F1 score between reference and hypothesis
    """
    f1_scores = []
    k = 1
    for n in xrange(1, 5):
        hyp_ngram_counts = Counter(ngrams(hypothesis, n))
        ref_ngram_counts = Counter(ngrams(reference, n))

        ngram_f1 = 0.
        if len(reference) < n:
            continue

        if len(hypothesis) >= n and len(reference) >= n:
            ngram_prec = sum(ngram_count for ngram, ngram_count in hyp_ngram_counts.iteritems() if ngram in ref_ngram_counts) / (len(hypothesis) - n + 1)
            ngram_recall = sum(ngram_count for ngram, ngram_count in ref_ngram_counts.iteritems() if ngram in hyp_ngram_counts) / (len(reference) - n + 1)

            ngram_f1 = 2 * ngram_prec * ngram_recall / (ngram_prec + ngram_recall) if ngram_prec + ngram_recall > 0. else 0.

        if len(hypothesis) < n <= len(reference) or ngram_f1 == 0.:
            ngram_f1 = 1. / (2 ** k * (len(reference) - n + 1))

        f1_scores.append(ngram_f1)

    f1_mean = (np.prod(f1_scores)) ** (1 / len(f1_scores))

    return f1_mean


def get_bp(reference, hypothesis):
    if len(hypothesis) >= len(reference):
        return 1.
    else:
        return math.exp(1 - len(reference) / len(hypothesis))


if __name__ == '__main__':
    ref_sent = 'thank you .'.split(' ')
    hyp_sent = 'thank'.split(' ')

    sm = SmoothingFunction()
    print sentence_bleu([ref_sent], hyp_sent, smoothing_function=sm.method3)
    print calc_bleu(ref_sent, hyp_sent)
    print calc_f1(ref_sent, hyp_sent)