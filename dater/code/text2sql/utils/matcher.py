from fuzzywuzzy import fuzz
import pandas as pd
import string

from utils.normalizer import str_normalize


class Matcher(object):
    def __init__(self):
        pass

    def match_sentence_with_table(self, sent: str, df: pd.DataFrame, fuzz_threshold=100):
        phrase2matched_cells = dict()
        sent = str_normalize(sent)
        sent = sent.strip(string.punctuation)
        for ngram in range(5, 0, -1):
            ngram_tokens_list = self._create_ngram_list(sent.split(), ngram)
            for row_id, row in df.iterrows():
                for col_id, cell in enumerate(row):
                    if df.columns[col_id] == 'row_id':
                        continue
                    cell = str(cell)
                    for ngram_phrase in ngram_tokens_list:
                        fuzz_score = fuzz.ratio(ngram_phrase, cell)
                        if fuzz_score >= fuzz_threshold:
                            if ngram_phrase not in phrase2matched_cells:
                                phrase2matched_cells[ngram_phrase] = []
                            phrase2matched_cells[ngram_phrase].append((cell, fuzz_score, (row_id, col_id)))
        # Remove non-longest phrase
        phrases = list(phrase2matched_cells.keys())
        for phrase in phrases:
            for other_phrase in phrases:
                if phrase != other_phrase and phrase in other_phrase:
                    del phrase2matched_cells[phrase]
                    break
        # Sort by fuzzy score
        for matched_cells in phrase2matched_cells.values():
            matched_cells.sort(key=lambda x: x[1], reverse=True)

        return phrase2matched_cells

    def match_phrase_with_table(self, phrase: str, df: pd.DataFrame, fuzz_threshold=70):
        matched_cells = []
        for row_id, row in df.iterrows():
            for col_id, cell in enumerate(row):
                cell = str(cell)
                fuzz_score = fuzz.ratio(phrase, cell)
                # if fuzz_score == 100:
                #     matched_cells = [(cell, fuzz_score, (row_id, col_id))]
                #     return matched_cells
                if fuzz_score >= fuzz_threshold:
                    matched_cells.append((cell, fuzz_score, (row_id, col_id)))
        # Sort by fuzzy score
        matched_cells.sort(key=lambda x: x[1], reverse=True)
        return matched_cells

    def _create_ngram_list(self, input_list, ngram_num):
        ngram_list = []
        if len(input_list) <= ngram_num:
            ngram_list.extend(input_list)
        else:
            for tmp in zip(*[input_list[i:] for i in range(ngram_num)]):
                tmp = " ".join(tmp)
                ngram_list.append(tmp)
        return ngram_list