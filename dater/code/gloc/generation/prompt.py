import random
from typing import Dict, Tuple
import pandas as pd
import copy
from gloc.utils import table_linearization,twoD_list_transpose
from gloc.json_utils import NoIndent, MyEncoder
import json
class PromptBuilder(object):
    def __init__(self, args) -> None:
        self.args = args
        random.seed(args.seed)
    
    def _select_x_col_prompt(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                        format:str='codex')->str:
        prompt = "use f_col() api to filter out useless columns in the table according to informations in the statement and the table.\n"
        prompt += 'statement : '+ question +'.\n'
        prompt +='table caption : '

        if caption is not None:
            prompt += caption
        prompt += '\n'
        str_table = table_linearization(df.sample(num_rows) if len(df)>num_rows else df ,format=format)
        prompt += str_table + '\n'
        #------------
        prompt = '/*\n' + prompt +  '*/\n'
        # prompt += "If you are not sure which columns are useful, please select all columns.\n"
        #---------------
        # prompt += 'if you are not exactly sure link which columns, please link to possible columns.\n'
        
        return prompt
    def _select_x_col_prompt_json(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                        format:str='codex')->str:
        prompt = "use f_col() api to filter out useless columns in the table according to informations in the statement and the table.\n"
        prompt += 'statement : '+ question +'.\n'
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        tmp = df.sample(num_rows).values.tolist() if len(df) >= num_rows else df.values.tolist()
        list_table = [list(df.columns)] + tmp
       
        # list_table = list_table[:num_rows+1] if num_rows + 1 <= len(list_table) else list_table
        list_table = twoD_list_transpose(list_table, len(list_table))
        # print(list_table)
        # input()
        dic = {
            "table_caption": caption,
            "columns": NoIndent(list(df.columns)),
            "table_column_priority": [NoIndent(i) for i in list_table],
        }
        linear_dic = json.dumps(dic,cls=MyEncoder,ensure_ascii=False,sort_keys=False,indent=2)
        #------------
        prompt = '/*\n' + prompt + linear_dic +  '\n*/\n'
        # prompt += "If you are not sure which columns are useful, please link more value in statement.\n"
        #---------------
        # prompt += 'if you are not exactly sure link which columns, please pay more attention to column value link to columns part\n'
        # prompt += 'similar words link to columns :\n'
        return prompt
    def _select_x_col_prompt_json2(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                    format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        tmp = df.sample(num_rows).values.tolist() if len(df) >= num_rows else df.values.tolist()
        list_table = [list(df.columns)] + tmp
    
        # list_table = list_table[:num_rows+1] if num_rows + 1 <= len(list_table) else list_table
        list_table = twoD_list_transpose(list_table, len(list_table))
        # print(list_table)
        # input()
        if caption is not None:

            dic = {
                "table_caption": caption,
                "columns": NoIndent(list(df.columns)),
                "table_column_priority": [NoIndent(i) for i in list_table],
            }
        else:
            dic = {
                "columns": NoIndent(list(df.columns)),
                "table_column_priority": [NoIndent(i) for i in list_table],
            }
        linear_dic = json.dumps(dic,cls=MyEncoder,ensure_ascii=False,sort_keys=False,indent=2)
        #------------
        prompt = '/*\n' + prompt + linear_dic +  '\n*/\n'
        prompt += 'statement : '+ question +'\n'
        # prompt += "If you are not sure which columns are useful, please link more value in statement.\n"
        #---------------
        # prompt += 'if you are not exactly sure link which columns, please pay more attention to column value link to columns part\n'
        # prompt += 'similar words link to columns :\n'
        return prompt
    def _select_x_col_prompt_jsonv7(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                    format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        # tmp = df.sample(num_rows).values.tolist() if len(df) >= num_rows else df.values.tolist()
        tmp = df.values.tolist()[:num_rows] if len(df) >= num_rows else df.values.tolist()
        list_table = [list(df.columns)] + tmp
    
        # list_table = list_table[:num_rows+1] if num_rows + 1 <= len(list_table) else list_table
        list_table = twoD_list_transpose(list_table, len(list_table))
        # print(list_table)
        # input()
        if caption is not None:
            dic = {
                "table_caption": caption,
                "columns": NoIndent(list(df.columns)),
                "table_column_priority": [NoIndent(i) for i in list_table],
            }
        else:
            dic = {
                "columns": NoIndent(list(df.columns)),
                "table_column_priority": [NoIndent(i) for i in list_table],
            }
        linear_dic = json.dumps(dic,cls=MyEncoder,ensure_ascii=False,sort_keys=False,indent=2)
        #------------
        prompt = '/*\ntable = ' + prompt + linear_dic +  '\n*/\n'
        # prompt = '/*\n' + prompt + linear_dic +  '\n*/\n'
        prompt += 'statement : '+ question +'.\n'
        # prompt += "If you are not sure which columns are useful, please link more value in statement.\n"
        #---------------
        # prompt += 'if you are not exactly sure link which columns, please pay more attention to column value link to columns part\n'
        # prompt += 'similar words link to columns :\n'
        return prompt
    def _select_x_row_prompt(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        str_table = table_linearization(df.sample(num_rows) if len(df)>num_rows else df ,format=format)
        if caption is not None:
            prompt = 'table caption : ' + caption + '\n' + str_table + '\n'
        else:
            prompt = str_table + '\n'

        prompt = '/*\n'+prompt + '*/\n'
        prompt += 'statement : '+ question +'\n'
        prompt += 'explain :'
        return prompt

    def _select_x_cloze_prompt(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        str_table = table_linearization(df.sample(num_rows) if len(df)>num_rows else df ,format=format)
        if caption is not None:
            prompt = 'table caption : ' + caption + '\n' + str_table + '\n'
        else:
            prompt = str_table + '\n'
        prompt = '/*\n'+prompt + '*/\n'
        prompt += 'Q : '+ question +'\n'
        prompt += 'sub questions :'
        return prompt

    def _select_x_end2end_prompt(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        str_table = table_linearization(df ,format=format)
        if caption is not None:
            prompt = 'table caption : ' + caption + '\n' + str_table + '\n'
        else:
            prompt = str_table + '\n'
        prompt = '/*\n'+prompt + '*/\n'
        prompt += 'statement : '+ question +'\n'
        return prompt
    def _select_x_wtq_end2end_prompt(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,
                format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        str_table = table_linearization(df ,format=format)
        prompt = str_table + '\n'
        prompt = '/*\n'+prompt + '*/\n'
        prompt += 'question : '+ question +'\n'
        return prompt
    
    def _select_x_fetaqa_end2end_prompt(self, question: str, caption: str=None, df: pd.DataFrame=None, num_rows: int=3,format:str='codex')->str:
        # df = df.sample(num_rows) if len(df) >= num_rows else df
        prompt = ''
        str_table = table_linearization(df ,format=format)
        prompt = 'table caption : ' + caption + '\n' + str_table + '\n'
        prompt = '/*\n'+prompt + '*/\n'
        prompt += 'question : '+ question +'\n'
        # prompt += 'Answer questions according to the given table and be as detailed as possible.\n'

        return prompt

    def build_generate_prompt(
            self,
            table: pd.DataFrame,
            question: str = None,
            title: str = None,
            num_rows: int = 3,
            select_type: str = 'col',
            **kwargs
    ):
        """
        Build the prompt of the generation sample.
        """

        # assert select_type in ['all', 'row','col'], "select type Error "
        generate_prompt = ""
        if select_type == 'col':
            # means select col
            # generate_prompt += self._select_x_col_prompt_json(question, title, table, num_rows)
            # generate_prompt += self._select_x_col_prompt_json2(question, title, table, num_rows)
            # generate_prompt += self._select_x_col_prompt(question, title, table, num_rows)
            generate_prompt += self._select_x_col_prompt_jsonv7(question, title, table, num_rows)
        elif select_type == 'row':
            generate_prompt += self._select_x_row_prompt(question, title, table, num_rows)
        elif select_type == 'all':
            generate_prompt += self._select_x_col_prompt(question, title, table, num_rows)
        elif select_type == 'cloze':
            generate_prompt += self._select_x_cloze_prompt(question, title, table, num_rows)
        elif select_type == 'end2end':
            generate_prompt += self._select_x_end2end_prompt(question, title, table, num_rows)
        elif select_type == 'wtq_end2end':
            generate_prompt += self._select_x_wtq_end2end_prompt(question, title, table, num_rows)
        elif select_type=='fetaqa_end2end':
            generate_prompt += self._select_x_fetaqa_end2end_prompt(question, title, table, num_rows)
        return generate_prompt
        # task instruction

