import json
from tqdm import tqdm
from collections import defaultdict

from .TreeNode import *


def load_node_from_json(json_in):
    with open(json_in) as f:
        data_in = json.load(f)

    for data in tqdm(data_in):
        nl = data['logic_str']
        logic = data["logic"]
        table_header = data["table_header"]
        table_cont = data["table_cont"]

        try:
            pd_in = defaultdict(list)
            for ind, header in enumerate(table_header):
                for inr, row in enumerate(table_cont):

                    # remove last summarization row
                    if inr == len(table_cont) - 1 \
                            and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or \
                                 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                        continue
                    pd_in[header].append(row[ind])

            pd_table = pd.DataFrame(pd_in)
        except Exception:
            continue

        root = Node(pd_table, logic)
        yield root, nl
