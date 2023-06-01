import re
import json
import argparse

from typing import Dict, List


def parse_flattened_result(to_parse):
    raw_str = to_parse
    reg = re.search(r'available[S|s]izes =.*( xxl | XXL | XL | xl | XS | xs | S | s | M | m | L | l )' , to_parse)
    if reg != None:
        start_pos , end_pos = reg.span()
        t = to_parse[start_pos:end_pos]
        size_list = t.strip().split('=')[1].split(',')
        size_list = [size.strip() for size in size_list if size.strip()!='']
        size_str = ' '.join(size_list)
        to_parse = to_parse.replace(t , f'availableSizes = {size_str}')
    dialog_act_regex = re.compile(r'(.*)?  *\[(.*)\] *\(([^\]]*)\)')    
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= (\[(.*)\]|[^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")

    belief = []
    # Parse
    to_parse = to_parse.lower()
    if "=> belief state : " not in to_parse:
        splits = ['', to_parse.strip()]
    else:
        splits = to_parse.strip().split("=> belief state : ")
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split("<EOB>")
        # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
        to_parse = splits[0].strip()

        for dialog_act in dialog_act_regex.finditer(to_parse):
            d = {
                "act": dialog_act.group(1).lower(),
                "slots": [],
                "request_slots": [],
                "objects": [],
                "raw":raw_str
            }

            for slot in slot_regex.finditer(dialog_act.group(2)):
                d["slots"].append([slot.group(1).strip().lower(), slot.group(2).strip().lower()])

            for request_slot in request_regex.finditer(dialog_act.group(3)):
                d["request_slots"].append(request_slot.group(1).strip().lower())

            if d != {}:
                belief.append(d)
    return belief


def format_for_dst(predictions: List[str]) -> List[Dict]:
    '''
        Formats model predictions for subtask 2, 3.

        NOTE: This follows the format given by the baseline.

        Args:
            predictions <List[str]>: predictions outputted by model
        Returns:
            submission <List[Dict]>: submission format
    '''
    submission = list()
    for pred in predictions:
        submission.append(
            parse_flattened_result(pred)
        )
    return submission

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction', type=str,
        default=None, help="file to convert (line-by-line *.txt format)"
    )
    parser.add_argument(
        '--output', type=str,
        default='dstc10-simmc-teststd-pred-subtask-3.json',
        help="json output path"
    )
    args = parser.parse_args()

    with open(args.prediction, 'r') as f:
        prediction = list(f.readlines())

    submission = format_for_dst(prediction)

    with open(args.output, 'w') as f:
        json.dump(submission, f)