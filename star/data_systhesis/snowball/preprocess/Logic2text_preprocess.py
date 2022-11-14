# run this file as a module by  `python -m preprocess.split_data`
import json
import random
import csv


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    return json.loads(data)


def random_choose(data, max_size):
    new_dict = {}
    for origin, mutated in data.items():
        mutated = dict(random.sample(list(mutated.items()), min(len(mutated), max_size)))
        new_dict[origin] = mutated
    return new_dict


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file)


def load_csv(path):
    data = []
    for i, line in enumerate(csv.reader(open(labeled_path, encoding='utf-8'))):
        if i == 0:
            continue
        data.append(line)
    return data


if __name__ == '__main__':
    # # sample a part of mutated queries to the data folder
    # path = "../mutated_all_data.json"
    # target_path = "../data/logic2text/preprocessed/"
    # target_name = "mutation.json"
    # MAX_SIZE = 25
    # large_data = load_json(path)
    # small_data = random_choose(large_data, MAX_SIZE)
    # write_json(target_path + target_name, small_data)

    # # split train dev test from translated_all.json
    # path = "../data/logic2text/preprocessed/translated_all.json"
    # target_path = "../data/logic2text/preprocessed/{}.json"
    # raw_mapping_path = "../data/logic2text/raw/all_data.json"
    # raw_path = "../data/logic2text/raw/{}.json"
    # data_split = ['train', 'test', 'valid']
    #
    # trans_all = load_json(path)
    # raw_all = load_json(raw_mapping_path)
    # raw_trans_dict = {}
    # for trans, raw in zip(trans_all, raw_all):
    #     raw_trans_dict[raw['logic_str']] = trans
    #
    # for split in data_split:
    #     print("Writing {}".format(split))
    #     count = 0
    #     split_target_path = target_path.format(split)
    #     split_raw_path = raw_path.format(split)
    #     raw_data = load_json(split_raw_path)
    #     new_split = {}
    #     for sample in raw_data:
    #         sample_raw = sample['logic_str']
    #         if sample_raw not in raw_trans_dict.keys():
    #             print(sample_raw)
    #             count += 1
    #             continue
    #         sample_trans = raw_trans_dict[sample_raw]
    #         new_split[sample_raw] = sample_trans
    #     with open(split_target_path, 'w', encoding='utf-8') as file:
    #         json.dump(new_split, file)
    #     print("total available:", len(new_split), "error:", count)

    # # from labeled csv to eval folder
    # labeled_path = '../labeled_eval.csv'
    # target_path = '../data/logic2text/eval/test.json'
    # # target format {"sql": str, "translated_sql": str, "question": str, "label": "0", "remark": "negative"}
    # # source format [Logic_str,	Translated,	Sentence, Negative,	Label]
    # data = load_csv(labeled_path)
    # with open(target_path, 'w', encoding='utf-8') as file:
    #     for sample in data:
    #         new_sample = {
    #             "label": "0",
    #             "remark": "negative",
    #             "sql": sample[0],
    #             "translated_sql": sample[1],
    #             "question": sample[3]}
    #         file.write(json.dumps(new_sample)+'\n')

    # # test what happens
    # path1 = "../data/logic2text/preprocessed/train.json"
    # path2 = "../data/logic2text/preprocessed/mutation.json"
    # path3 = "../data/logic2text/raw/train.json"
    # trans = load_json(path1) # {logic:nl}
    # mutate = load_json(path2) # {origin_logic: {logic:nl}}
    # train_raw = load_json(path3) # [{query: logic, question: nl}]
    #
    # for data in train_raw:
    #     logic = data['query']
    #     if logic not in trans.keys():
    #         print(data)
    #     if logic not in mutate.keys():
    #         print(data)

    # add some positive cases to eval
    path = '/home/yzh2749/snowball_new/data/logic2text/raw/test.json'
    trans_path = '/home/yzh2749/snowball_new/data/logic2text/preprocessed/test.json'
    target_path = '/home/yzh2749/snowball_new/data/logic2text/eval/test.json'
    raw_data = load_json(path)
    trans_data = load_json(trans_path)

    with open(target_path, 'a', encoding='utf-8') as file:
        for sample in raw_data:
            question = sample['question']
            code = sample['query']
            try:
                trans = trans_data[code]
            except:
                print(code)
                continue
            new_sample = {"label": "1",
                          "remark": "positive",
                          "sql": code,
                          "translated_sql": trans,
                          "question": question}

            file.write(json.dumps(new_sample)+'\n')
