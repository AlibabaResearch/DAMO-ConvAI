"""
Retriever to retrieve relevant examples from annotations.
"""

import copy
from typing import Dict, List, Tuple, Any
import nltk
from nltk.stem import SnowballStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils.normalizer import normalize
from retrieval.retrieve_pool import OpenAIQARetrievePool, QAItem


class OpenAIQARetriever(object):
    def __init__(self, retrieve_pool: OpenAIQARetrievePool):
        self.retrieve_pool = retrieve_pool

    @staticmethod
    def _string_bleu(q1: str, q2: str, stop_words=None, stemmer=None):
        """
        BLEU score.
        """
        q1, q2 = normalize(q1), normalize(q2)
        reference = [[tk for tk in nltk.word_tokenize(q1)]]
        candidate = [tk for tk in nltk.word_tokenize(q2)]
        if stemmer is not None:
            reference = [[stemmer.stem(tk) for tk in reference[0]]]
            candidate = [stemmer.stem(tk) for tk in candidate]

        chencherry_smooth = SmoothingFunction()  # bleu smooth to avoid hard behaviour when no ngram overlaps
        bleu_score = sentence_bleu(
            reference,
            candidate,
            weights=(0.25, 0.3, 0.3, 0.15),
            smoothing_function=chencherry_smooth.method1
        )
        return bleu_score

    def _qh2qh_similarity(
            self,
            item: QAItem,
            num_retrieve_samples: int,
            score_func: str,
            qa_type: str,
            weight_h: float = 0.2,
            verbose: bool = False
    ):
        """
        Retrieve top K nsqls based on query&header to query&header similarities.
        """
        q = item.qa_question
        header_wo_row_id = copy.copy(item.table['header'])
        header_wo_row_id.remove('row_id')
        h = ' '.join(header_wo_row_id)
        stemmer = SnowballStemmer('english')
        if score_func == 'bleu':
            retrieve_q_list = [(d, self._string_bleu(q, d.qa_question.split('@')[1], stemmer=stemmer))
                               for d in self.retrieve_pool if d.qa_question.split('@')[0] == qa_type]
            retrieve_h_list = [(d, self._string_bleu(h, ' '.join(d.table['header']), stemmer=stemmer))
                               for d in self.retrieve_pool if d.qa_question.split('@')[0] == qa_type]
            retrieve_list = [(retrieve_q_list[idx][0], retrieve_q_list[idx][1] + weight_h * retrieve_h_list[idx][1])
                             for idx in range(len(retrieve_q_list))]
        else:
            raise ValueError
        retrieve_list = sorted(retrieve_list, key=lambda x: x[1], reverse=True)
        retrieve_list = list(map(lambda x: x[0], retrieve_list))[:num_retrieve_samples]

        if verbose:
            print(retrieve_list)

        return retrieve_list

    def retrieve(
            self,
            item: QAItem,
            num_shots: int,
            method: str = 'qh2qh_bleu',
            qa_type: str = 'map',
            verbose: bool = False
    ) -> List[QAItem]:
        """
        Retrieve a list of relevant QA samples.
        """
        if method == 'qh2qh_bleu':
            retrieved_items = self._qh2qh_similarity(
                item=item,
                num_retrieve_samples=num_shots,
                score_func='bleu',
                qa_type=qa_type,
                verbose=verbose
            )
            return retrieved_items
        else:
            raise ValueError(f'Retrieve method {method} is not supported.')
