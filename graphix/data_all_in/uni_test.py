import pdb
import json

# train = json.load(open("./data/seq2seq_train_dataset.json","rb"))
schema = 'schema: | perpetrator | perpetrator : perpetrator_id , people_id , date , year , location , country , killed , injured | people : people_id , name , height , weight , home town | *'
table_items = ['perpetrator', 'people']
column_items = ['*', 'Perpetrator_ID', 'People_ID', 'Date', 'Year', 'Location', 'Country', 'Killed', 'Injured', 'People_ID', 'Name', 'Height', 'Weight', 'Home Town']

def find_schema_idx(db_seq, table_items, column_items):
    seq_lst = db_seq.split(" ")
    special_token = ["|", ':', ',', 'schema:', '(', ')']
    table_idx_lst = []
    column_idx_lst = []
    schema_items = [item.lower() for item in table_items + column_items]
    db_id = ""

    for i, item in enumerate(seq_lst):
        if item in special_token:
            continue
        if seq_lst[i - 1] == '(':
            continue

        if i < len(seq_lst) - 1:
            if seq_lst[i - 1] == seq_lst[i + 1] == '|':
                db_id = item

            elif seq_lst[i - 1] == '|' and seq_lst[i + 1] == ':':
                table_idx_lst.append(i)

            elif seq_lst[i - 1] == ":":
                # head columns
                if seq_lst[i + 1] == ",":
                    # head columns without value
                    column_idx_lst.append(i)
                elif seq_lst[i + 1] == "(":
                    # head columns without value
                    column_idx_lst.append(i)

            elif seq_lst[i - 1] == ",":
                if seq_lst[i + 1] == "," or seq_lst[i + 1] == "(":
                    # middle columns (with values):
                    if item in schema_items:
                        column_idx_lst.append(i)
                elif seq_lst[i + 1] == "|":
                    # tail columns:
                    if item in schema_items:
                        column_idx_lst.append(i)

                elif seq_lst[i + 1] == ")":
                    continue

                else:
                    # columns with multiple words: "home town"
                    temp_idx_lst = []
                    match_multi_words(cur_idx=i, column_cur_idx=temp_idx_lst, seq_lst=seq_lst)
                    column_idx_lst.append(temp_idx_lst)
                # append the last element:
        else:
            if item in schema_items:
                if seq_lst[i - 1] == "," or seq_lst[i - 1] == ":" or item == '*':
                    column_idx_lst.append(i)

    # put * into the head position:
    star_idx = column_idx_lst.pop()
    column_idx_lst.insert(0, star_idx)

    # pdb.set_trace()
    assert len(table_idx_lst + column_idx_lst) == len(table_items + column_items)
    print(len(seq_lst))
    return table_idx_lst, column_idx_lst, db_id


def match_multi_words(cur_idx: int, column_cur_idx: list, seq_lst):
    column_cur_idx.append(cur_idx)
    if seq_lst[cur_idx + 1] in [",", "|"]:
        return column_cur_idx
    else:
        match_multi_words(cur_idx + 1, column_cur_idx, seq_lst)

print(find_schema_idx(schema, table_items, column_items))
# temp = []
# result = match_multi_words(cur_idx=32, column_cur_idx=temp, seq_lst=seq)
# print(temp)

s = ['*', 'Perpetrator_ID', 'People_ID', 'Date', 'Year', 'Location', 'Country', 'Killed', 'Injured', 'People_ID', 'Name', 'Height', 'Weight', 'Home Town']
print(" ".join([a.lower() for a in s]).split(" "))

print(" ")