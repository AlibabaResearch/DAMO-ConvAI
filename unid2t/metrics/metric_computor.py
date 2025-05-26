import os
import argparse
import sacrebleu
import sys
sys.path.append("/mnt/disk1/liliang/experiment/AuoregressiveTable2Text")

from metrics.meteor.meteor import Meteor
from uda import utils


def cal_bleu(refs, hyps):
    if type(refs[0]) == str:
        refs = [refs]
    else:
        n_set_ref = len(refs[0])
        multi_refs = []
        for i in range(n_set_ref):
            tmp_refs = []
            for ref in refs:
                tmp_refs.append(ref[i])

            multi_refs.append(tmp_refs)

        refs = multi_refs

    bleu_scores = sacrebleu.corpus_bleu(hyps, refs)
    raw_bleu_score = sacrebleu.raw_corpus_bleu(hyps, refs, 0.01).score

    bleu_scores = str(bleu_scores)
    # print(bleu_scores)
    bleu_scores = bleu_scores.split(' ')
    bleu = float(bleu_scores[2])
    bleu_metrics = dict()
    bleu_metrics['bleu'] = bleu
    bleu_list = bleu_scores[3].split('/')
    for i, score in enumerate(bleu_list):
        bleu_metrics['bleu-{}'.format(i+1)] = float(score)
    """
    bleu = bleu_scores.score
    bleu_metrics = dict()
    bleu_metrics['bleu'] = bleu
    bleu_list = bleu_scores.prec_str.split('/')
    for i, score in enumerate(bleu_list):
        bleu_metrics['bleu-{}'.format(i + 1)] = float(score)
    """
    bleu_metrics['bleu-raw'] = float(raw_bleu_score)
    return bleu_metrics


def cal_chrf(refs, hyps):
    if type(refs[0]) == str:
        refs = [refs]
    else:
        n_set_ref = len(refs[0])
        multi_refs = []
        for i in range(n_set_ref):
            tmp_refs = []
            for ref in refs:
                tmp_refs.append(ref[i])

            multi_refs.append(tmp_refs)

        refs = multi_refs
    chrf_score = sacrebleu.corpus_chrf(hyps, refs)
    chrf_score = chrf_score.score

    return chrf_score


def cal_meteor(refs, hyps):
    res, gts = {}, {}
    for i, (h, r) in enumerate(zip(hyps, refs)):
        res[i] = [h]
        gts[i] = [r]

    score, scores = Meteor().compute_score(gts, res)

    return score


def compute_metrics(refs, hyps):
    refs = [ref[0] for ref in refs]
    rval = cal_bleu(refs, hyps)
    # rval['ROUGE-L'] = cal_rouge_l(refs, hyps)
    # rval['meteor'] = cal_meteor(refs, hyps)
    rval['chrf++'] = cal_chrf(refs, hyps)

    return rval


def compute_one_file(ref_src, hyp_src):
    refs = utils.read_text_file(ref_src)
    hyps = utils.read_text_file(hyp_src)

    assert len(refs) == len(hyps)
    print("------ Total {} examples -------".format(len(refs)))
    print("Tmp ref:", refs[0])
    print("Tmp hyp:", hyps[0])

    results = compute_metrics(refs=refs, hyps=hyps)
    print("------- Cal Metrics ------")
    for key, val in results.items():
        print("{}: {}".format(key, val))


def compute(ref_src, hyp_dir):
    if os.path.isfile(hyp_dir):
        hyp_files = [hyp_dir]
    else:
        files = os.listdir(hyp_dir)
        files = [file for file in files if file.endswith('.txt')]
        hyp_files = [os.path.join(hyp_dir, file) for file in files]

    n_file = len(hyp_files)
    print("------ Total {} files waiting for evaluation ------".format(n_file))
    for i, hyp_src in enumerate(hyp_files):
        file_name = os.path.basename(hyp_src)
        print("------ Evaluating {},  {} / {}th".format(file_name, i+1, n_file))
        compute_one_file(ref_src, hyp_src)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file_src", type=str, default='')
    parser.add_argument("--hyp_file_src", type=str, default='')
    args = parser.parse_args()
    compute(args.ref_file_src, args.hyp_file_src)