import re
import json
import records
from typing import List, Dict
from sqlalchemy.exc import SQLAlchemyError
from utils.sql.all_keywords import ALL_KEY_WORDS


class WTQDBEngine:
    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute_wtq_query(self, sql_query: str):
        out = self.conn.query(sql_query)
        results = out.all()
        merged_results = []
        for i in range(len(results)):
            merged_results.extend(results[i].values())
        return merged_results

    def delete_rows(self, row_indices: List[int]):
        sql_queries = [
            "delete from w where id == {}".format(row) for row in row_indices
        ]
        for query in sql_queries:
            self.conn.query(query)


def process_table_structure(_wtq_table_content: Dict, _add_all_column: bool = False):
    # remove id and agg column
    headers = [_.replace("\n", " ").lower() for _ in _wtq_table_content["headers"][2:]]
    header_map = {}
    for i in range(len(headers)):
        header_map["c" + str(i + 1)] = headers[i]
    header_types = _wtq_table_content["types"][2:]

    all_headers = []
    all_header_types = []
    vertical_content = []
    for column_content in _wtq_table_content["contents"][2:]:
        # only take the first one
        if _add_all_column:
            for i in range(len(column_content)):
                column_alias = column_content[i]["col"]
                # do not add the numbered column
                if "_number" in column_alias:
                    continue
                vertical_content.append([str(_).replace("\n", " ").lower() for _ in column_content[i]["data"]])
                if "_" in column_alias:
                    first_slash_pos = column_alias.find("_")
                    column_name = header_map[column_alias[:first_slash_pos]] + " " + \
                                  column_alias[first_slash_pos + 1:].replace("_", " ")
                else:
                    column_name = header_map[column_alias]
                all_headers.append(column_name)
                if column_content[i]["type"] == "TEXT":
                    all_header_types.append("text")
                else:
                    all_header_types.append("number")
        else:
            vertical_content.append([str(_).replace("\n", " ").lower() for _ in column_content[0]["data"]])
    row_content = list(map(list, zip(*vertical_content)))

    if _add_all_column:
        ret_header = all_headers
        ret_types = all_header_types
    else:
        ret_header = headers
        ret_types = header_types
    return {
        "header": ret_header,
        "rows": row_content,
        "types": ret_types,
        "alias": list(_wtq_table_content["is_list"].keys())
    }


def retrieve_wtq_query_answer(_engine, _table_content, _sql_struct: List):
    # do not append id / agg
    headers = _table_content["header"]

    def flatten_sql(_ex_sql_struct: List):
        # [ "Keyword", "select", [] ], [ "Column", "c4", [] ]
        _encode_sql = []
        _execute_sql = []
        for _ex_tuple in _ex_sql_struct:
            keyword = str(_ex_tuple[1])
            # upper the keywords.
            if keyword in ALL_KEY_WORDS:
                keyword = str(keyword).upper()

            # extra column, which we do not need in result
            if keyword == "w" or keyword == "from":
                # add 'FROM w' make it executable
                _encode_sql.append(keyword)
            elif re.fullmatch(r"c\d+(_.+)?", keyword):
                # only take the first part
                index_key = int(keyword.split("_")[0][1:]) - 1
                # wrap it with `` to make it executable
                _encode_sql.append("`{}`".format(headers[index_key]))
            else:
                _encode_sql.append(keyword)
            # c4_list, replace it with the original one
            if "_address" in keyword or "_list" in keyword:
                keyword = re.findall(r"c\d+", keyword)[0]

            _execute_sql.append(keyword)

        return " ".join(_execute_sql), " ".join(_encode_sql)

    _exec_sql_str, _encode_sql_str = flatten_sql(_sql_struct)
    try:
        _sql_answers = _engine.execute_wtq_query(_exec_sql_str)
    except SQLAlchemyError as e:
        _sql_answers = []
    _norm_sql_answers = [str(_).replace("\n", " ") for _ in _sql_answers if _ is not None]
    if "none" in _norm_sql_answers:
        _norm_sql_answers = []
    return _encode_sql_str, _norm_sql_answers, _exec_sql_str


def _load_table_w_page(table_path, page_title_path=None) -> dict:
    """
    attention: the table_path must be the .tsv path.
    Load the WikiTableQuestion from csv file. Result in a dict format like:
    {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
    """

    from utils.utils import _load_table

    table_item = _load_table(table_path)

    # Load page title
    if not page_title_path:
        page_title_path = table_path.replace("csv", "page").replace(".tsv", ".json")
    with open(page_title_path, "r") as f:
        page_title = json.load(f)['title']
    table_item['page_title'] = page_title

    return table_item
