#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re
from tqdm import tqdm
import transformers as huggingface_transformers
from uie.extraction.record_schema import RecordSchema
from uie.sel2record.record import MapConfig
from uie.extraction.scorer import *
from uie.sel2record.sel2record import SEL2Record
import math
import os


split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


def schema_to_ssi(schema: RecordSchema):
    ssi = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    ssi += "<extra_id_2> "
    return ssi


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()


class HuggingfacePredictor:
    def __init__(self, model_path, schema_file, max_source_length=256, max_target_length=192) -> None:
        self._tokenizer = huggingface_transformers.T5TokenizerFast.from_pretrained(
            model_path)
        self._model = huggingface_transformers.T5ForConditionalGeneration.from_pretrained(
            model_path)
        self._model.cuda()
        self._schema = RecordSchema.read_from_file(schema_file)
        self._ssi = schema_to_ssi(self._schema)
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def predict(self, text):
        text = [self._ssi + x for x in text]
        inputs = self._tokenizer(
            text, padding=True, return_tensors='pt').to(self._model.device)

        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:,
                                                            :self._max_source_length]

        result = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
        )
        return self._tokenizer.batch_decode(result, skip_special_tokens=False, clean_up_tokenization_spaces=False)


task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
    'event': EventScorer,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d', default='data/text2spotasoc/absa/14lap')
    parser.add_argument(
        '--model', '-m', default='./models/uie_n10_21_50w_absa_14lap')
    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('-c', '--config', dest='map_config',
                        help='Offset Re-mapping Config',
                        default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='normal',
                        choices=['set', 'normal', 'multimatch'])
    options = parser.parse_args()

    data_folder = options.data
    model_path = options.model

    predictor = HuggingfacePredictor(
        model_path=model_path,
        schema_file=f"{data_folder}/record.schema",
        max_source_length=options.max_source_length,
        max_target_length=options.max_target_length,
    )

    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict(data_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
    )

    for split, split_name in [('val', 'eval'), ('test', 'test')]:
        gold_filename = f"{data_folder}/{split}.json"

        text_list = [x['text'] for x in read_json_file(gold_filename)]
        token_list = [x['tokens'] for x in read_json_file(gold_filename)]

        batch_num = math.ceil(len(text_list) / options.batch_size)

        predict = list()
        for index in tqdm(range(batch_num)):
            start = index * options.batch_size
            end = index * options.batch_size + options.batch_size

            pred_seq2seq = predictor.predict(text_list[start: end])
            pred_seq2seq = [post_processing(x) for x in pred_seq2seq]

            predict += pred_seq2seq

        records = list()
        for p, text, tokens in zip(predict, text_list, token_list):
            r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
            records += [r]

        results = dict()
        for task, scorer in task_dict.items():

            gold_list = [x[task] for x in read_json_file(gold_filename)]
            pred_list = [x[task] for x in records]

            gold_instance_list = scorer.load_gold_list(gold_list)
            pred_instance_list = scorer.load_pred_list(pred_list)

            sub_results = scorer.eval_instance_list(
                gold_instance_list=gold_instance_list,
                pred_instance_list=pred_instance_list,
                verbose=options.verbose,
                match_mode=options.match_mode,
            )
            results.update(sub_results)

        with open(os.path.join(options.model, f'{split_name}_preds_record.txt'), 'w') as output:
            for record in records:
                output.write(f'{json.dumps(record)}\n')

        with open(os.path.join(options.model, f'{split_name}_preds_seq2seq.txt'), 'w') as output:
            for pred in predict:
                output.write(f'{pred}\n')

        with open(os.path.join(options.model, f'{split_name}_results.txt'), 'w') as output:
            for key, value in results.items():
                output.write(f'{split_name}_{key}={value}\n')


if __name__ == "__main__":
    main()
