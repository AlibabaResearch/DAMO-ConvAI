"""
Build NSQL generation prompt.
Two main parts:
1) PromptBuilder makes prompt for calling codex to generate NSQL(Binder-SQL).
2) OpenAIQAPromptBuilder makes prompt for calling codex to generate QA answers.
"""

import random
from typing import Dict, Tuple
import pandas as pd
import copy

from utils.errors import DuplicateColumnsError

from retrieval.retrieve_pool import QAItem

from utils.normalizer import prepare_df_for_neuraldb_from_table


def _create_table_prompt(df: pd.DataFrame, title: str):
    """
    Return the CREATE TABLE clause as prompt.
    """
    string = "CREATE TABLE {}(\n".format(title)
    for header in df.columns:
        column_type = 'text'
        try:
            if df[header].dtype == 'int64':
                column_type = 'int'
            elif df[header].dtype == 'float64':
                column_type = 'real'
            elif df[header].dtype == 'datetime64':
                column_type = 'datetime'
        except AttributeError as e:
            raise DuplicateColumnsError(e)

        string += '\t{} {},\n'.format(header, column_type)
    string = string.rstrip(',\n') + ')\n'
    return string


class PromptBuilder(object):
    def __init__(self, args):
        self.args = args
        self.prompt_style = args.prompt_style
        random.seed(args.seed)

    def _select_x_prompt(self, df: pd.DataFrame, num_rows: int,
                         few_shot_demonstration=True):
        """
        Return the first X rows table contents as prompt.
        """
        if self.prompt_style == 'create_table_select_full_table':
            string = '/*\nAll rows of the table:\nSELECT * FROM w;\n'
        elif self.prompt_style == 'create_table_select_3':
            string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)
        elif self.prompt_style == 'create_table_select_3_hidden':
            string = '/*\n{} example rows:\n'.format(num_rows)
        elif few_shot_demonstration is True and self.prompt_style in \
                ["create_table_select_3_full_table",
                 "create_table_select_3_full_table_w_gold_passage_image",
                 "create_table_select_3_full_table_w_all_passage_image"]:
            string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)
        elif few_shot_demonstration is False and self.prompt_style in \
                ["create_table_select_3_full_table",
                 "create_table_select_3_full_table_w_gold_passage_image",
                 "create_table_select_3_full_table_w_all_passage_image"]:
            string = '/*\nAll rows of the table:\nSELECT * FROM w;\n'
        else:
            raise ValueError(f"Select x prompt style {self.prompt_style} is not supported.")

        for column_id, header in enumerate(df.columns):
            string += str(header)
            if column_id != len(df.columns) - 1:
                string += '\t'
        string += '\n'

        for row_id, row in df.iloc[:num_rows].iterrows():
            for column_id, header in enumerate(df.columns):
                string += str(row[header])
                if column_id != len(df.columns) - 1:
                    string += '\t'
            string += '\n'
        string += '*/\n'

        return string

    def _passage_prompt(self, passages, only_title, db_style_prompt=True):
        """
        Return the passage prompt.
        """
        if not db_style_prompt:
            string = "Passages: "
            for passage in passages:
                if only_title:
                    string += passage['title'] + ';; '
                else:
                    string += passage['title'] + f" ({passage['text']})" + ';; '
            string = string.rstrip(';; ')
            string += '\n'
            return string
        else:
            if len(passages) == 0:
                return ""
            passage_table_prompt = ""
            _header = []
            _rows = [[]]
            for passage in passages:
                _header.append(passage['title'])
                _rows[0].append(passage['text'])
            passage_table = prepare_df_for_neuraldb_from_table({"header": _header, "rows": _rows})
            passage_table_prompt += _create_table_prompt(passage_table, "Passages")
            if not only_title:
                passage_table_prompt += self._select_x_prompt(
                    df=passage_table,
                    num_rows=passage_table.shape[0]
                )
            return passage_table_prompt

    def _image_prompt(self, images, only_title, db_style_prompt=True):
        """
        Return the image prompt.
        """
        if not db_style_prompt:
            string = "Images: "
            for image in images:
                if only_title:
                    string += image['title'] + ';;'
                else:
                    string += image['title'] + f" ({image['caption']})" + ';; '
            string = string.rstrip(';; ')
            string += '\n'
            return string
        else:
            if len(images) == 0:
                return ""
            image_table_prompt = ""
            _header = []
            _rows = [[]]
            for image in images:
                _header.append(image['title'])
                _rows[0].append(image['caption'])
            image_table = prepare_df_for_neuraldb_from_table({"header": _header, "rows": _rows})
            image_table_prompt += _create_table_prompt(image_table, "Images")
            if not only_title:
                image_table_prompt += self._select_x_prompt(
                    df=image_table,
                    num_rows=image_table.shape[0]
                )
            return image_table_prompt

    def _pick_target_columns(self, df, strategy):
        """
        Pick the controllable target columns for generation.
        """
        if strategy == 'random':
            return random.choice(list(df.columns) + ['*'])
        elif strategy == 'traverse':
            raise NotImplementedError
        else:
            return ValueError

    def _pick_operators(self, df, strategy):
        """
        Pick the controllable operators for generation.
        """
        candidate_operators = ['none', 'count', 'max', 'min', 'sum']
        if strategy == 'random':
            return random.choice(candidate_operators)
        elif strategy == 'traverse':
            raise NotImplementedError
        else:
            return ValueError

    def _pick_nested_levels(self, df, strategy):
        """
        Pick the controllable(maybe) nested levels for generation.
        """
        if strategy == 'fixed':
            return 2
        elif strategy == 'random':
            raise NotImplementedError
        elif strategy == 'traverse':
            raise NotImplementedError
        else:
            raise ValueError

    def build_one_shot_prompt(
            self,
            prompt_type: Tuple,
            table: pd.DataFrame,
            question: str,
            answer_text: str,
            nsql: str,
            passages: Dict = None,
            images: Dict = None,
            title: str = None,
            only_title: bool = False,
            **kwargs
    ):
        """
        Build one-shot prompt with table-question-nsql.
        """
        one_shot_prompt = ""
        if self.prompt_style == 'create_table_select_full_table':
            one_shot_prompt += _create_table_prompt(table, title)
            one_shot_prompt += self._select_x_prompt(
                df=table,
                num_rows=table.shape[0]
            )
        elif self.prompt_style in ['create_table_select_3_full_table', 'create_table_select_3']:
            one_shot_prompt += _create_table_prompt(table, title)
            one_shot_prompt += self._select_x_prompt(
                df=table,
                num_rows=3,
            )
        elif self.prompt_style == 'create_table':
            one_shot_prompt += _create_table_prompt(table, title)
        elif self.prompt_style == 'no_table':
            # No table input, to test Codex QA with only internal knowledge
            pass
        else:
            raise ValueError('{} is not supported.'.format(self.prompt_style))

        # question and nsql pairs
        if prompt_type == ('question', 'nsql'):
            one_shot_prompt += 'Q: {}\n'.format(question)
            one_shot_prompt += 'NeuralSQL: {}\n'.format(nsql)
        elif prompt_type == ('question', 'sql'):
            one_shot_prompt += 'Q: {}\n'.format(question)
            one_shot_prompt += 'SQL: {}\n'.format(nsql)
        elif prompt_type == ('question', 'answer'):
            one_shot_prompt += 'Q: {}\n'.format(question)
            one_shot_prompt += 'A: {}\n'.format(', '.join(answer_text))
        else:
            raise ValueError(f'Prompt type {prompt_type} is not supported.')

        return one_shot_prompt

    def build_generate_prompt(
            self,
            generate_type: Tuple,
            table: pd.DataFrame,
            question: str = None,
            passages: Dict = None,
            images: Dict = None,
            title: str = None,
            only_title: bool = False,
            supporting_context: Dict = None,
            **kwargs
    ):
        """
        Build the prompt of the generation sample.
        """
        generate_prompt = ""

        # task instruction
        if generate_type == ('answer',):
            generate_prompt += """\n-- Answer the question based on the given table below.\n\n"""
        elif generate_type == ('nsql',):
            generate_prompt += """\n-- Parse the question into NeuralSQL based on the given table below.\n\n"""
        elif generate_type == ('sql',):
            generate_prompt += """\n-- Parse the question into SQL based on the given table below.\n\n"""
        elif generate_type == ('npython',):
            generate_prompt += """\n-- Parse the question into NeuralPython based on the given table below.\n\n"""
        elif generate_type == ('python',):
            generate_prompt += """\n-- Parse the question into Python based on the given table below.\n\n"""
        else:
            generate_prompt += """\n-- Generate NeuralSQL and question pairs based on the given table below.\n\n"""

        # table prompt
        if self.prompt_style in ['create_table_select_full_table', 'create_table_select_3_full_table']:
            generate_prompt += _create_table_prompt(table, title)
            generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=table.shape[0],
                few_shot_demonstration=False
            )
        elif self.prompt_style in ['create_table_select_3']:
            generate_prompt += _create_table_prompt(table, title)
            generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=3,
                few_shot_demonstration=False
            )
        elif self.prompt_style == 'create_table':
            generate_prompt += _create_table_prompt(table, title)
        elif self.prompt_style == 'no_table':
            # No table input, to test Codex QA with only internal knowledge
            pass
        else:
            raise ValueError('{} is not supported.'.format(self.prompt_style))

        # determine the target to generate
        if generate_type == ('answer',):
            generate_prompt += 'Q: {}\n'.format(question)
            generate_prompt += 'A: '
        elif generate_type == ('nsql',):
            generate_prompt += 'Q: {}\n'.format(question)
            generate_prompt += 'NeuralSQL: '
        elif generate_type == ('sql',):
            generate_prompt += 'Q: {}\n'.format(question)
            generate_prompt += 'SQL: '
        elif generate_type == ('npython',):
            generate_prompt += 'Q: {}\n'.format(question)
            generate_prompt += 'NeuralPython: '
        elif generate_type == ('python',):
            generate_prompt += 'Q: {}\n'.format(question)
            generate_prompt += 'Python: '
        else:
            raise ValueError(f'Generate type {generate_type} is not supported.')

        return generate_prompt


class OpenAIQAPromptBuilder(object):
    @staticmethod
    def table2codex_prompt(table, table_title=None, drop_row_id=True, ):
        _table = copy.deepcopy(table)
        header = _table['header']
        rows = _table['rows']
        if drop_row_id:
            if header[0] == "row_id":
                header = header[1:]
                rows = [_row[1:] for _row in rows]
        prompt_str = 'Table: {}\n'.format(table_title) if table_title else ''
        prompt_str += "/*\n"
        prompt_str += "\t".join(header) + "\n"
        prompt_str += '\n'.join(["\t".join([str(cell) for cell in row]) for row in rows]) + "\n"
        prompt_str += "*/"
        return prompt_str

    @staticmethod
    def build_one_shot_prompt(
            item: QAItem,
            answer_split_token: str = ';',
            verbose: bool = False,
            prompting_method='new_db',
            db_mapping_token="😅"
    ) -> str:
        """
        Build one-shot QA prompt.
        """
        assert prompting_method in ['basic', 'new_db']
        qa_type, qa_question = item.qa_question.split('@')
        prompt = ''
        db_prompt = OpenAIQAPromptBuilder.table2codex_prompt(item.table, item.title)
        prompt += "Give a database as shown below:\n{}\n\n".format(db_prompt)

        if prompting_method == 'basic':
            if qa_type == "map":
                prompt += "Q: Answer question \"{}\" row by row.".format(qa_question)
                assert answer_split_token is not None
                prompt += " The answer should be a list split by '{}' and have {} items in total.".format(
                    answer_split_token, len(item.table['rows']))
                prompt += "\nA: {}\n\n".format(f'{answer_split_token}'.join(item.qa_answer))
            elif qa_type == "ans":
                prompt += "Q: Answer question \"{}\" for the table.".format(qa_question)
                prompt += " "
                prompt += "\nA: {}\n\n".format(f'{answer_split_token}'.join(item.qa_answer))
            else:
                raise ValueError("The QA type is not supported!")

            return prompt

        elif prompting_method == "new_db":
            if qa_type == "map":
                prompt += "Q: Answer question \"{}\" row by row.".format(qa_question)
                assert answer_split_token is not None
                db_prompt_lines = db_prompt.split("\n")[2:-1]  # skip Title, /*, and */
                db_prompt_lines_with_answer = []
                db_prompt_lines_with_answer.append("/*")
                db_prompt_lines_with_answer.append(db_prompt_lines[0])
                assert len(db_prompt_lines[1:]) == len(
                    item.qa_answer), "answer items and table rows must be in the same number, check annotations"
                for db_prompt_line, qa_answer_item in zip(db_prompt_lines[1:], item.qa_answer):
                    db_prompt_lines_with_answer.append(
                        "{}{}{}".format(db_prompt_line, db_mapping_token, qa_answer_item))
                db_prompt_lines_with_answer.append("*/")
                prompt += "\n{}\n".format("\n".join(db_prompt_lines_with_answer))

            elif qa_type == "ans":
                prompt += "Q: Answer question \"{}\" for the table.".format(qa_question)
                prompt += " "
                prompt += "\nA: {}\n".format(f'{answer_split_token}'.join(item.qa_answer))
            else:
                raise ValueError("The QA type is not supported!")

            return prompt
