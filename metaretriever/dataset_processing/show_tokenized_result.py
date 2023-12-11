from transformers import AutoTokenizer
import json
import argparse
import tabulate
from universal_ie.record_schema import RecordSchema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='t5-base')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--schema', default='event')
    options = parser.parse_args()

    if "chinese_t5_pegasus" in options.model:
        tokenizer = T5PegasusTokenizer.from_pretrained(options.model)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            options.model,
            use_fast=False,
            mirror='tuna',
        )

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<extra_id_0>", "<extra_id_1>"]}
    )

    folder_path = options.data

    schema_file = f"{folder_path}/{options.schema}.schema"

    event_schema = RecordSchema.read_from_file(schema_file)

    table = list()
    for typename in event_schema.type_list:
        typename = typename.replace('_', ' ')
        after_tokenzied = tokenizer.encode(typename, add_special_tokens=False)
        table += [[typename,
                   after_tokenzied,
                   tokenizer.convert_ids_to_tokens(after_tokenzied)]]

    print(tokenizer)
    print(type(tokenizer))

    print("===============Event Schema=================")
    print(tabulate.tabulate(
        table,
        headers=['type', 'token id', 'tokenized'],
        tablefmt='grid',
    ))

    print("===============Instance=================")

    table = list()
    for index, instance in enumerate(open(folder_path + "/val.json").readlines()[:10]):
        instance = json.loads(instance)
        table += [["Text  %s" % index] + [instance['text']]]
        table += [["Token %s" % index] +
                  ['|'.join(tokenizer.tokenize(instance['text']))]]
        if 'entity' in instance:
            table += [["Entity %s" % index] +
                      ['|'.join(tokenizer.tokenize(instance['event']))]]
        if 'relation' in instance:
            table += [["Relation %s" % index] +
                      ['|'.join(tokenizer.tokenize(instance['relation']))]]
        if 'event' in instance:
            table += [["Event %s" % index] +
                      ['|'.join(tokenizer.tokenize(instance['event']))]]
    print(tabulate.tabulate(table, headers=['text', 'event'], tablefmt='grid'))

    print("===============Specical Symbol=================")
    table = list()

    for name in ['<extra_id_0>', '<extra_id_1>']:
        table += [[name, tokenizer.encode(name), tokenizer.tokenize(name)]]
    print(tabulate.tabulate(
        table,
        headers=['specical symbol', 'token id', 'tokenized'],
        tablefmt='grid'
    ))


if __name__ == "__main__":
    main()
