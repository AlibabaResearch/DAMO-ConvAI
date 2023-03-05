import math, logging, copy, json
from collections import Counter, OrderedDict

import numpy as np
from nltk.util import ngrams
from sklearn.metrics import f1_score


def DAEvaluate(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    results = {}
    # for avg_name in ['micro', 'macro', 'weighted', 'samples']:
    for avg_name in ['micro']:
        my_f1_score = f1_score(y_true=labels, y_pred=preds, average=avg_name)
        results["f1_{}".format(avg_name)] = my_f1_score

    return results


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


if __name__ == '__main__':
    pass
