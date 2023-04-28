from typing import List, Dict
import pandas as pd
import recognizers_suite
from recognizers_suite import Culture
import re
import unicodedata
from fuzzywuzzy import fuzz

from utils.sql.extraction_from_sql import *
from utils.sql.all_keywords import ALL_KEY_WORDS

culture = Culture.English


def str_normalize(user_input, recognition_types=None):
    """A string normalizer which recognize and normalize value based on recognizers_suite"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end:idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = ["datetime",
                             "number",
                             # "ordinal",
                             # "percentage",
                             # "age",
                             # "currency",
                             # "dimension",
                             # "temperature",
                             ]

    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(recognizers_suite, "recognize_{}".format(recognition_type))(user_input,
                                                                                              culture)  # may match multiple parts
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # if the datetime is a period:
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # in some cases, this variable could be none.
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex'])  # We use timex as normalization
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[:-len("00:00:00") - 1]
        # '2008-04-13 00:00:00' -> '2008-04-13'
    return user_input


def prepare_df_for_neuraldb_from_table(table: Dict, add_row_id=True, normalize=True, lower_case=True):
    header, rows = table['header'], table['rows']
    if add_row_id and 'row_id' not in header:
        header = ["row_id"] + header
        rows = [["{}".format(i)] + row for i, row in enumerate(rows)]
    if normalize:
        df = convert_df_type(pd.DataFrame(data=rows, columns=header), lower_case=lower_case)
    else:
        df = pd.DataFrame(data=rows, columns=header)

    return df


def convert_df_type(df: pd.DataFrame, lower_case=True):
    """
    A simple converter of dataframe data type from string to int/float/datetime.
    """

    def get_table_content_in_column(table):
        if isinstance(table, pd.DataFrame):
            header = table.columns.tolist()
            rows = table.values.tolist()
        else:
            # Standard table dict format
            header, rows = table['header'], table['rows']
        all_col_values = []
        for i in range(len(header)):
            one_col_values = []
            for _row in rows:
                one_col_values.append(_row[i])
            all_col_values.append(one_col_values)
        return all_col_values

    # Rename empty columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header == '':
            new_columns.append('FilledColumnName')  # Fixme: give it a better name when all finished!
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Rename duplicate columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header in new_columns:
            new_header, suffix = header, 2
            while new_header in new_columns:
                new_header = header + '_' + str(suffix)
                suffix += 1
            new_columns.append(new_header)
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Recognize null values like "-"
    null_tokens = ['', '-', '/']
    for header in df.columns:
        df[header] = df[header].map(lambda x: str(None) if x in null_tokens else x)

    # Convert the null values in digit column to "NaN"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_number_flag = True
        for row_i, cell_value in enumerate(one_col_values):
            try:
                float(cell_value)
            except Exception as e:
                if not cell_value in [str(None), str(None).lower()]:
                    # None or none
                    all_number_flag = False
        if all_number_flag:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: "NaN" if x in [str(None), str(None).lower()] else x)

    # Normalize cell values.
    for header in df.columns:
        df[header] = df[header].map(lambda x: str_normalize(x))

    # Strip the mis-added "01-01 00:00:00"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_with_00_00_00 = True
        all_with_01_00_00_00 = True
        all_with_01_01_00_00_00 = True
        for row_i, cell_value in enumerate(one_col_values):
            if not str(cell_value).endswith(" 00:00:00"):
                all_with_00_00_00 = False
            if not str(cell_value).endswith("-01 00:00:00"):
                all_with_01_00_00_00 = False
            if not str(cell_value).endswith("-01-01 00:00:00"):
                all_with_01_01_00_00_00 = False
        if all_with_01_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01-01 00:00:00")])
            continue

        if all_with_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01 00:00:00")])
            continue

        if all_with_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len(" 00:00:00")])
            continue

    # Do header and cell value lower case
    if lower_case:
        new_columns = []
        for header in df.columns:
            lower_header = str(header).lower()
            if lower_header in new_columns:
                new_header, suffix = lower_header, 2
                while new_header in new_columns:
                    new_header = lower_header + '-' + str(suffix)
                    suffix += 1
                new_columns.append(new_header)
            else:
                new_columns.append(lower_header)
        df.columns = new_columns
        for header in df.columns:
            # df[header] = df[header].map(lambda x: str(x).lower())
            df[header] = df[header].map(lambda x: str(x).lower().strip())

    # Recognize header type
    for header in df.columns:

        float_able = False
        int_able = False
        datetime_able = False

        # Recognize int & float type
        try:
            df[header].astype("float")
            float_able = True
        except:
            pass

        if float_able:
            try:
                if all(df[header].astype("float") == df[header].astype(int)):
                    int_able = True
            except:
                pass

        if float_able:
            if int_able:
                df[header] = df[header].astype(int)
            else:
                df[header] = df[header].astype(float)

        # Recognize datetime type
        try:
            df[header].astype("datetime64")
            datetime_able = True
        except:
            pass

        if datetime_able:
            df[header] = df[header].astype("datetime64")

    return df


def normalize(x):
    """ Normalize string. """
    # Copied from WikiTableQuestions dataset official evaluator.
    if x is None:
        return None
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", "\"", x)
    x = re.sub("[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub("((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub("(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub('^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub('\s+', ' ', x, flags=re.U).lower().strip()
    return x


def post_process_sql(sql_str, df, table_title=None, process_program_with_fuzzy_match_on_db=True, verbose=False):
    """Post process SQL: including basic fix and further fuzzy match on cell and SQL to process"""

    def basic_fix(sql_str, all_headers, table_title=None):
        def finditer(sub_str: str, mother_str: str):
            result = []
            start_index = 0
            while True:
                start_index = mother_str.find(sub_str, start_index, -1)
                if start_index == -1:
                    break
                end_idx = start_index + len(sub_str)
                result.append((start_index, end_idx))
                start_index = end_idx
            return result

        if table_title:
            sql_str = sql_str.replace("FROM " + table_title, "FROM w")
            sql_str = sql_str.replace("FROM " + table_title.lower(), "FROM w")

        """Case 1: Fix the `` missing. """
        # Remove the null header.
        while '' in all_headers:
            all_headers.remove('')

        # Remove the '\n' in header.
        # This is because the WikiTQ won't actually show the str in two lines,
        # they use '\n' to mean that, and display it in the same line when print.
        sql_str = sql_str.replace("\\n", "\n")
        sql_str = sql_str.replace("\n", "\\n")

        # Add `` in SQL.

        all_headers.sort(key=lambda x: len(x), reverse=True)
        have_matched = [0 for i in range(len(sql_str))]

        # match quotation
        idx_s_single_quotation = [_ for _ in range(1, len(sql_str)) if
                                  sql_str[_] in ["\'"] and sql_str[_ - 1] not in ["\'"]]
        idx_s_double_quotation = [_ for _ in range(1, len(sql_str)) if
                                  sql_str[_] in ["\""] and sql_str[_ - 1] not in ["\""]]
        for idx_s in [idx_s_single_quotation, idx_s_double_quotation]:
            if len(idx_s) % 2 == 0:
                for idx in range(int(len(idx_s) / 2)):
                    start_idx = idx_s[idx * 2]
                    end_idx = idx_s[idx * 2 + 1]
                    have_matched[start_idx: end_idx] = [2 for _ in range(end_idx - start_idx)]

        # match headers
        for header in all_headers:
            if (header in sql_str) and (header not in ALL_KEY_WORDS):
                all_matched_of_this_header = finditer(header, sql_str)
                for matched_of_this_header in all_matched_of_this_header:
                    start_idx, end_idx = matched_of_this_header
                    if all(have_matched[start_idx: end_idx]) == 0 and (not sql_str[start_idx - 1] == "`") and (
                            not sql_str[end_idx] == "`"):
                        have_matched[start_idx: end_idx] = [1 for _ in range(end_idx - start_idx)]
                        # a bit ugly, but anyway.

        # re-compose sql from the matched idx.
        start_have_matched = [0] + have_matched
        end_have_matched = have_matched + [0]
        start_idx_s = [idx - 1 for idx in range(1, len(start_have_matched)) if
                       start_have_matched[idx - 1] == 0 and start_have_matched[idx] == 1]
        end_idx_s = [idx for idx in range(len(end_have_matched) - 1) if
                     end_have_matched[idx] == 1 and end_have_matched[idx + 1] == 0]
        assert len(start_idx_s) == len(end_idx_s)
        spans = []
        current_idx = 0
        for start_idx, end_idx in zip(start_idx_s, end_idx_s):
            spans.append(sql_str[current_idx:start_idx])
            spans.append(sql_str[start_idx:end_idx + 1])
            current_idx = end_idx + 1
        spans.append(sql_str[current_idx:])
        sql_str = '`'.join(spans)

        return sql_str

    def fuzzy_match_process(sql_str, df, verbose=False):
        """
        Post-process SQL by fuzzy matching value with table contents.
        """

        def _get_matched_cells(value_str, df, fuzz_threshold=70):
            """
            Get matched table cells with value token.
            """
            matched_cells = []
            for row_id, row in df.iterrows():
                for cell in row:
                    cell = str(cell)
                    fuzz_score = fuzz.ratio(value_str, cell)
                    if fuzz_score == 100:
                        matched_cells = [(cell, fuzz_score)]
                        return matched_cells
                    if fuzz_score >= fuzz_threshold:
                        matched_cells.append((cell, fuzz_score))

            matched_cells = sorted(matched_cells, key=lambda x: x[1], reverse=True)
            return matched_cells

        def _check_valid_fuzzy_match(value_str, matched_cell):
            """
            Check if the fuzzy match is valid, now considering:
            1. The number/date should not be disturbed, but adding new number or deleting number is valid.
            """
            number_pattern = "[+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
            numbers_in_value = re.findall(number_pattern, value_str)
            numbers_in_matched_cell = re.findall(number_pattern, matched_cell)
            try:
                numbers_in_value = [float(num.replace(',', '')) for num in numbers_in_value]
            except:
                print(f"Can't convert number string {numbers_in_value} into float in _check_valid_fuzzy_match().")
            try:
                numbers_in_matched_cell = [float(num.replace(',', '')) for num in numbers_in_matched_cell]
            except:
                print(
                    f"Can't convert number string {numbers_in_matched_cell} into float in _check_valid_fuzzy_match().")
            numbers_in_value = set(numbers_in_value)
            numbers_in_matched_cell = set(numbers_in_matched_cell)

            if numbers_in_value.issubset(numbers_in_matched_cell) or numbers_in_matched_cell.issubset(numbers_in_value):
                return True
            else:
                return False

        # Drop trailing '\n```', a pattern that may appear in Codex SQL generation
        sql_str = sql_str.rstrip('```').rstrip('\n')

        # Replace QA module with placeholder
        qa_pattern = "QA\(.+?;.*?`.+?`.*?\)"
        qas = re.findall(qa_pattern, sql_str)
        for idx, qa in enumerate(qas):
            sql_str = sql_str.replace(qa, f"placeholder{idx}")

        # Parse and replace SQL value with table contents
        sql_tokens = tokenize(sql_str)
        sql_template_tokens = extract_partial_template_from_sql(sql_str)
        # Fix 'between' keyword bug in parsing templates
        fixed_sql_template_tokens = []
        sql_tok_bias = 0
        for idx, sql_templ_tok in enumerate(sql_template_tokens):
            sql_tok = sql_tokens[idx + sql_tok_bias]
            if sql_tok == 'between' and sql_templ_tok == '[WHERE_OP]':
                fixed_sql_template_tokens.extend(['[WHERE_OP]', '[VALUE]', 'and'])
                sql_tok_bias += 2  # pass '[VALUE]', 'and'
            else:
                fixed_sql_template_tokens.append(sql_templ_tok)
        sql_template_tokens = fixed_sql_template_tokens
        for idx, tok in enumerate(sql_tokens):
            if tok in ALL_KEY_WORDS:
                sql_tokens[idx] = tok.upper()

        if verbose:
            print(sql_tokens)
            print(sql_template_tokens)

        assert len(sql_tokens) == len(sql_template_tokens)
        value_indices = [idx for idx in range(len(sql_template_tokens)) if sql_template_tokens[idx] == '[VALUE]']
        for value_idx in value_indices:
            # Skip the value if the where condition column is QA module
            if value_idx >= 2 and sql_tokens[value_idx - 2].startswith('placeholder'):
                continue
            value_str = sql_tokens[value_idx]
            # Drop \"\" for fuzzy match
            is_string = False
            if value_str[0] == "\"" and value_str[-1] == "\"":
                value_str = value_str[1:-1]
                is_string = True
            # If already fuzzy match, skip
            if value_str[0] == '%' or value_str[-1] == '%':
                continue
            value_str = value_str.lower()
            # Fuzzy Match
            matched_cells = _get_matched_cells(value_str, df)

            if verbose:
                print(matched_cells)

            new_value_str = value_str
            if matched_cells:
                # new_value_str = matched_cells[0][0]
                for matched_cell, fuzz_score in matched_cells:
                    if _check_valid_fuzzy_match(value_str, matched_cell):
                        new_value_str = matched_cell
                        if verbose and new_value_str != value_str:
                            print("\tfuzzy match replacing!", value_str, '->', matched_cell, f'fuzz_score:{fuzz_score}')
                        break
            if is_string:
                new_value_str = f"\"{new_value_str}\""
            sql_tokens[value_idx] = new_value_str
        # Compose new sql string
        # Clean column name in SQL since columns may have been tokenized in the postprocessing, e.g., (ppp) -> ( ppp )
        new_sql_str = ' '.join(sql_tokens)
        sql_columns = re.findall('`\s(.*?)\s`', new_sql_str)
        for sql_col in sql_columns:
            matched_columns = []
            for col in df.columns:
                score = fuzz.ratio(sql_col.lower(), col)
                if score == 100:
                    matched_columns = [(col, score)]
                    break
                if score >= 80:
                    matched_columns.append((col, score))
            matched_columns = sorted(matched_columns, key=lambda x: x[1], reverse=True)
            if matched_columns:
                matched_col = matched_columns[0][0]
                new_sql_str = new_sql_str.replace(f"` {sql_col} `", f"`{matched_col}`")
            else:
                new_sql_str = new_sql_str.replace(f"` {sql_col} `", f"`{sql_col}`")

        # Restore QA modules
        for idx, qa in enumerate(qas):
            new_sql_str = new_sql_str.replace(f"placeholder{idx}", qa)

        # Fix '<>' when composing the new sql
        new_sql_str = new_sql_str.replace('< >', '<>')

        return new_sql_str

    sql_str = basic_fix(sql_str, list(df.columns), table_title)

    if process_program_with_fuzzy_match_on_db:
        try:
            sql_str = fuzzy_match_process(sql_str, df, verbose)
        except:
            pass

    return sql_str
