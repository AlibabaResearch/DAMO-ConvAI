#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
from collections import OrderedDict
import numpy as np

from tabulate import tabulate

event_record_result_valid_keys = [
    'eval_offset-evt-trigger-P', 'eval_offset-evt-trigger-R', 'eval_offset-evt-trigger-F1', 'eval_offset-evt-role-P', 'eval_offset-evt-role-R', 'eval_offset-evt-role-F1',
    'test_offset-evt-trigger-P', 'test_offset-evt-trigger-R', 'test_offset-evt-trigger-F1', 'test_offset-evt-role-P', 'test_offset-evt-role-R', 'test_offset-evt-role-F1',
]

span_record_result_valid_keys = [
    'eval_offset-ent-P', 'eval_offset-ent-R', 'eval_offset-ent-F1',
    'test_offset-ent-P', 'test_offset-ent-R', 'test_offset-ent-F1',
]

relation_strict_record_result_valid_keys = [
    'eval_offset-rel-strict-P', 'eval_offset-rel-strict-R', 'eval_offset-rel-strict-F1',
    'test_offset-rel-strict-P', 'test_offset-rel-strict-R', 'test_offset-rel-strict-F1',
]

relation_boundary_record_result_valid_keys = [
    'eval_offset-rel-boundary-P', 'eval_offset-rel-boundary-R', 'eval_offset-rel-boundary-F1',
    'test_offset-rel-boundary-P', 'test_offset-rel-boundary-R', 'test_offset-rel-boundary-F1',
]

record_result_valid_keys = [
    'eval_offset-ent-F1', 'eval_offset-rel-boundary-F1', 'eval_offset-rel-strict-F1', 'eval_offset-evt-trigger-F1', 'eval_offset-evt-role-F1',
    'test_offset-ent-F1', 'test_offset-rel-boundary-F1', 'test_offset-rel-strict-F1', 'test_offset-evt-trigger-F1', 'test_offset-evt-role-F1',
]


def align_float(x):
    return '%.2f' % x if isinstance(x, float) else x


def parse_trainer_state(filename):
    trainer_state = json.load(open(filename))
    if trainer_state['best_model_checkpoint'] is not None:
        return trainer_state['best_model_checkpoint'].split('/')[-1].replace('checkpoint-', '')
    else:
        return 'last'


def parse_global_step(filename):
    return str(json.load(open(filename))['global_step'])


def check_out_of_memory(filename):
    if os.path.exists(filename):
        try:
            with open(filename) as fin:
                for line in fin:
                    if 'CUDA out of memory' in line:
                        return True
        except UnicodeDecodeError:
            return False
    return False


def get_run_name(folder_name, prefix):
    split_list = folder_name.replace('/', '_').split('_') \
        if prefix == 'run' \
        else folder_name.split('_')[1:]
    new_att_list = list()
    for att in split_list:
        if att.startswith(prefix):
            continue
        new_att_list += [att]
    return '_'.join(new_att_list)


class ResultSummary:
    def __init__(self, result_valid_keys):
        self.result_valid_keys = result_valid_keys
        self.header_result_valid_keys = [
            value.replace('trigger', 't').replace('role', 'r').replace('eval', 'e').replace('test', 't').replace(
                'F1', 'F').replace('offset', 'o').replace('string', 's').replace('strict', 's').replace('boundary', 'b')
            for value in result_valid_keys]
        for x, y in zip(self.result_valid_keys, self.header_result_valid_keys):
            print("%s -> %s" % (x, y))

    def parse_best_log(self, folder_name, file_map, default_key='running'):
        result = dict()

        eval_result_filename = os.path.join(folder_name, file_map['eval'])
        test_result_filename = os.path.join(folder_name, file_map['test'])

        lines = list()
        if os.path.exists(eval_result_filename):
            lines += open(eval_result_filename).readlines()
        if os.path.exists(test_result_filename):
            lines += open(test_result_filename).readlines()

        for line in lines:
            key, value = line.strip().split('=')
            if key.strip() not in self.result_valid_keys:
                continue
            result[key.strip()] = float(value.strip())

        for key in self.result_valid_keys:
            if key not in result:
                result[key] = default_key

        return result

    def get_valid_folder(self, model_folders, file_map, span_pretrain=False):
        all_result = list()

        for model_folder in model_folders:
            print(model_folder)
            sub_folder_list = sorted(os.listdir(model_folder))
            for sub_folder_name in sub_folder_list:
                if sub_folder_name.endswith('log') or sub_folder_name.endswith('err'):
                    continue

                sub_folder = os.path.join(model_folder, sub_folder_name)
                log_filename = sub_folder + '.log'

                if span_pretrain:
                    if os.path.exists(os.path.join(sub_folder, 'span_pretrain')):
                        default_key = 'running'
                        trained_folder = os.path.join(sub_folder, 'span_pretrain')
                        log_filename = os.path.join(
                            sub_folder, 'span_pretrain.log')
                        state_filename = os.path.join(
                            sub_folder, 'span_pretrain', 'trainer_state.json')
                    else:
                        print('Unused folder: %s' % sub_folder)
                        continue
                else:

                    if os.path.exists(os.path.join(sub_folder, 'event_finetune')):
                        default_key = 'finetune'
                        trained_folder = os.path.join(sub_folder, 'event_finetune')
                        log_filename = os.path.join(
                            sub_folder, 'event_finetune.log')
                        state_filename = os.path.join(
                            sub_folder, 'event_finetune', 'trainer_state.json')

                    elif os.path.exists(os.path.join(sub_folder, 'span_pretrain')):
                        default_key = 'pretrain'
                        trained_folder = os.path.join(sub_folder, 'span_pretrain')
                        log_filename = os.path.join(
                            sub_folder, 'span_pretrain.log')
                        state_filename = os.path.join(
                            sub_folder, 'span_pretrain', 'trainer_state.json')

                    else:
                        default_key = 'running'
                        state_filename = os.path.join(
                            sub_folder, 'trainer_state.json')
                        trained_folder = sub_folder

                if os.path.exists(log_filename):
                    out_of_memory = check_out_of_memory(log_filename)
                else:
                    out_of_memory = False

                if out_of_memory:
                    result = {key: 'OOM' for key in self.result_valid_keys}
                    checkpoint = 'OOM'
                else:
                    result = self.parse_best_log(
                        trained_folder, file_map, default_key)
                    checkpoint = parse_trainer_state(state_filename) if os.path.exists(
                        state_filename) else default_key
                    global_step = parse_global_step(state_filename) if os.path.exists(
                        state_filename) else default_key
                    checkpoint = checkpoint + '/' + global_step

                all_result += [[sub_folder, checkpoint, result]]
        return all_result

    def result_to_table(self, all_result, sort_key=0):
        table = list()
        for sub_folder_name, checkpoint, result in all_result:
            table += [[sub_folder_name, checkpoint] +
                      [result.get(key, 'running') for key in self.result_valid_keys]]

        table = [[align_float(x) for x in y] for y in table]

        table.sort()
        table.sort(key=lambda x: x[sort_key])

        print(tabulate(table, headers=[
              'exp', 'checkpoint'] + self.header_result_valid_keys))

    def result_to_table_reduce(self, all_result, sort_key=0, reduce_function=np.mean, reduce_key='run'):
        table = list()
        sub_run = OrderedDict()
        for sub_folder_name, checkpoint, result in all_result:

            sub_run_name = get_run_name(sub_folder_name, reduce_key)
            if sub_run_name not in sub_run:
                sub_run[sub_run_name] = list()

            sub_run_result = [result.get(key, 'running')
                              for key in self.result_valid_keys]
            if 'running' in sub_run_result or 'OOM' in sub_run_result:
                continue

            sub_run[sub_run_name] += [sub_run_result]

        for sub_run_name, sub_run_results in sub_run.items():
            if len(sub_run_results) == 0:
                table += [[sub_run_name, 0] + ['-']]
            else:
                table += [[sub_run_name, len(sub_run_results)] +
                          list(reduce_function(sub_run_results, 0))]

        table = [[align_float(x) for x in y] for y in table]

        table.sort()
        table.sort(key=lambda x: x[sort_key])

        print(tabulate(table, headers=['exp', 'num'] +
                       self.header_result_valid_keys))


def main():
    record_valid_keys_map = {
        'entity': span_record_result_valid_keys,
        'relation': relation_strict_record_result_valid_keys,
        'relation-boundary': relation_boundary_record_result_valid_keys,
        'event': event_record_result_valid_keys,
        'record': record_result_valid_keys,
    }

    import argparse
    parser = argparse.ArgumentParser(
        description='Summary Multi-run Result'
    )
    parser.add_argument('-model', dest='model', default=['output'], nargs='+',
                        help='Output Model Folder Path')
    parser.add_argument('-sort', dest='sort', default=0,
                        type=int, help='Sort Column Index')
    parser.add_argument('-mean', dest='mean', action='store_true',
                        help='Reduce by mean Function')
    parser.add_argument('-std', dest='std', action='store_true',
                        help='Reduce by std Function')
    parser.add_argument('-span-pretrain', dest='span_pretrain',
                        action='store_true',
                        help='Load Span Pretrain Result for Text2Event')
    parser.add_argument('-record', dest='record', default='record',
                        choices=record_valid_keys_map.keys(),
                        help='Record Type')
    parser.add_argument('-string', dest='offset', action='store_false',
                        help='Report String Match Result')
    parser.add_argument('-offset', dest='offset', action='store_true',
                        help='Report Offset Match Result (default)')
    parser.set_defaults(offset=True)
    parser.add_argument('-reduce', dest='reduce', default='run',
                        help='Reduce Key, default is `run`')
    options = parser.parse_args()

    if options.record in record_valid_keys_map:
        file_map = {
            'eval': 'eval_results.txt',
            'test': 'test_results.txt',
        }
    else:
        raise NotImplementedError('Invalid Record Type: %s' % options.record)

    result_valid_keys = record_valid_keys_map[options.record]

    if not options.offset:
        result_valid_keys = [key.replace('offset', 'string')
                             for key in result_valid_keys]

    result_summary = ResultSummary(
        result_valid_keys=result_valid_keys
    )
    print(options.model)

    def check_valid_model(x):
        return not (os.path.isfile(x) or x.endswith('_log'))

    valid_model_paths = filter(check_valid_model, options.model)

    all_result = result_summary.get_valid_folder(
        model_folders=valid_model_paths,
        file_map=file_map,
        span_pretrain=options.span_pretrain
    )

    if options.mean:
        result_summary.result_to_table_reduce(
            all_result,
            sort_key=options.sort,
            reduce_function=np.mean,
            reduce_key=options.reduce,
        )
    elif options.std:
        result_summary.result_to_table_reduce(
            all_result,
            sort_key=options.sort,
            reduce_function=np.std,
            reduce_key=options.reduce
        )
    else:
        result_summary.result_to_table(all_result, sort_key=options.sort)


if __name__ == "__main__":
    main()
