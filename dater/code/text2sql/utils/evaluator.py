import re

from utils.normalizer import str_normalize
from utils.wtq.evaluator import to_value_list, check_denotation
from utils.mmqa.evaluator import acc


class Evaluator:
    def __init__(self):
        pass

    def evaluate(
            self,
            pred_answer,
            gold_answer,
            dataset,
            allow_semantic=True,
            question=None
    ):
        if dataset == 'wikitq':
            return self.eval_ex_match(pred_answer, gold_answer, allow_semantic, question)
        elif dataset == 'tab_fact':
            return self.eval_tabfact_match(pred_answer, gold_answer)
        elif dataset == 'mmqa':
            # For more metrics on MMQA,
            # please use the utils/mmqa/eval_mmqa.py to call official on all prediction data
            return self.eval_mmqa_match(pred_answer, gold_answer)
        else:
            raise ValueError(f'{dataset} evaluator is not supported.')

    def eval_ex_match(self, pred, gold, allow_semantic=True, question=None):
        pred = [str(p).lower().strip() for p in pred]
        gold = [str(g).lower().strip() for g in gold]

        if not allow_semantic:
            # WikiTQ eval w. string normalization using recognizer
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)
        else:
            assert isinstance(question, str)
            question = re.sub('\s+', ' ', question).strip().lower()
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = sorted(list(set(pred)))
            gold = sorted(list(set(gold)))
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
            if len(pred) == 1 and len(gold) == 1:
                if (pred[0] == '0' and gold[0] == 'no') \
                        or (pred[0] == '1' and gold[0] == 'yes'):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index('or')
                    token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                    if (pred[0] == '0' and gold[0] == token_after_or) \
                            or (pred[0] == '1' and gold[0] == token_before_or):
                        return True
                except Exception as e:
                    pass
            # (2) Number value (allow units) and Date substring match
            if len(pred) == 1 and len(gold) == 1:
                NUMBER_UNITS_PATTERN = re.compile('^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
                DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
                DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred[0], gold[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace('-', ' ').split()), set(g.replace('-', ' ').split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)

    def eval_tabfact_match(self, pred, gold):
        if isinstance(pred, list):
            pred = pred[0]
        pred, gold = str(pred), str(gold)
        return pred == gold

    def eval_mmqa_match(self, pred_answer, gold_answer):
        return acc(pred_answer, gold_answer)
