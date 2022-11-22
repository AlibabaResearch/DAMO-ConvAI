# encoding=utf8

from metrics.spider.spider_exact_match import compute_exact_match_metric
from metrics.spider.spider_test_suite import compute_test_suite_metric
from metrics.sparc.interaction_scores import compute_interaction_metric


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        if self.args.seq2seq.target_with_db_id:
            # Remove database id from all predictions
            preds = [pred.split("|", 1)[-1].strip() for pred in preds]
        # exact_match = compute_exact_match_metric(preds, golds)
        test_suite = compute_test_suite_metric(preds, golds, db_dir=self.args.test_suite_db_dir)
        interaction_scores = compute_interaction_metric(preds, golds)

        return {**test_suite, **interaction_scores}
