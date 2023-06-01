#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import json
import re
import os
import copy
import numpy as np

START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"

def initialize_count_dict():
    c = {
        "n_frames": 0.0,
        "n_true_acts": 0.0,
        "n_pred_acts": 0.0,
        "n_correct_acts": 0.0,
        "n_true_slots": 0.0,
        "n_pred_slots": 0.0,
        "n_correct_slots": 0.0,
        "n_true_request_slots": 0.0,
        "n_pred_request_slots": 0.0,
        "n_correct_request_slots": 0.0,
        "n_true_objects": 0.0,
        "n_pred_objects": 0.0,
        "n_correct_objects": 0.0,
        "n_true_disamb_candidates": 0.0,
        "n_pred_disamb_candidates": 0.0,
        "n_correct_disamb_candidates": 0.0,        
        "n_correct_beliefs": 0.0,
    }
    return copy.deepcopy(c)

def evaluate_frame(true_frame, pred_frame, strict=True):
    """
    If strict=True,
        For each dialog_act (frame), set(slot values) must match.
        If dialog_act is incorrect, its set(slot values) is considered wrong.
    """
    count_dict = initialize_count_dict()
    count_dict["n_frames"] += 1

    # Compare Dialog Actss
    true_act = true_frame["act"] if "act" in true_frame else None
    pred_act = pred_frame["act"] if "act" in pred_frame else None
    b_correct_act = true_act == pred_act
    count_dict["n_correct_acts"] += b_correct_act
    count_dict["n_true_acts"] += "act" in true_frame
    count_dict["n_pred_acts"] += "act" in pred_frame

    # (1) Compare Slots
    #true_frame_slot_values = {f"{k}={v}" for k, v in true_frame.get("slots", [])}
    #pred_frame_slot_values = {f"{k}={v}" for k, v in pred_frame.get("slots", [])}

    true_frame_slot_values = set()
    pred_frame_slot_values = set()

    for k, v in true_frame.get("slots", []):
        if k in set(['availableSizes']):
            # For availableSizes, we expect that the type is <list>.
            # Otherwise, try converting it to a <list>.
            if type(v) == str:
                try:
                    v = list(eval(v))
                except:
                    v = [v]

            elif type(v) == tuple or type(v) == set:
                v = list(v)

            # Sort the elements to get consistent ordering.
            # For slots with a list of elements, all elements must be captured.
            if type(v) == list:
                v.sort()

        true_frame_slot_values.add(f"{k}={v}")

    for k, v in pred_frame.get("slots", []):
        if k in set(['availableSizes']):
            if type(v) == str:
                try:
                    v = list(eval(v))
                except:
                    v = [v]

            elif type(v) == tuple or type(v) == set:
                v = list(v)
            if type(v) == list:
                v.sort()

        pred_frame_slot_values.add(f"{k}={v}")

    count_dict["n_true_slots"] += len(true_frame_slot_values)
    count_dict["n_pred_slots"] += len(pred_frame_slot_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_slots"] += len(
            true_frame_slot_values.intersection(pred_frame_slot_values)
        )

    # Debug only
    # if len(true_frame_slot_values.intersection(pred_frame_slot_values)) != len(pred_frame_slot_values):
    # print(true_frame_slot_values)
    # print(pred_frame_slot_values)
    # print(len(true_frame_slot_values.intersection(pred_frame_slot_values)) == len(pred_frame_slot_values))
    # print('--')

    # (2) Compare Request slots
    true_frame_request_slot_values = {rs for rs in true_frame.get("request_slots", [])}
    pred_frame_request_slot_values = {rs for rs in pred_frame.get("request_slots", [])}
    # print(true_frame_request_slot_values)

    count_dict["n_true_request_slots"] += len(true_frame_request_slot_values)
    count_dict["n_pred_request_slots"] += len(pred_frame_request_slot_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_request_slots"] += len(
            true_frame_request_slot_values.intersection(pred_frame_request_slot_values)
        )

    # (3) Compare Objects
    true_frame_object_values = {
        object_id for object_id in true_frame.get("objects", [])
    }
    pred_frame_object_values = {
        object_id for object_id in pred_frame.get("objects", [])
    }
    # print(true_frame_object_values)

    count_dict["n_true_objects"] += len(true_frame_object_values)
    count_dict["n_pred_objects"] += len(pred_frame_object_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_objects"] += len(
            true_frame_object_values.intersection(pred_frame_object_values)
        )

    # (4) Compare Disambiguation Objects
    true_frame_disamb_candidate_values = {
        disamb_candidate_id for disamb_candidate_id in true_frame.get("disambiguation_candidates", [])
    }
    pred_frame_disamb_candidate_values = {
        disamb_candidate_id for disamb_candidate_id in pred_frame.get("disambiguation_candidates", [])
    }
    # print(true_frame_disamb_candidate_values)

    count_dict["n_true_disamb_candidates"] += len(true_frame_disamb_candidate_values)
    count_dict["n_pred_disamb_candidates"] += len(pred_frame_disamb_candidate_values)

    if strict and not b_correct_act:
        pass
    else:
        count_dict["n_correct_disamb_candidates"] += len(
            true_frame_disamb_candidate_values.intersection(pred_frame_disamb_candidate_values)
        )

    # (5) Joint
    count_dict["n_correct_beliefs"] += (
        b_correct_act
        and true_frame_slot_values == pred_frame_slot_values
        and true_frame_request_slot_values == pred_frame_request_slot_values
        and true_frame_object_values == pred_frame_object_values
    )

    return count_dict

def add_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1}

def evaluate_turn(true_turn, pred_turn):

    count_dict = initialize_count_dict()

    # Must preserve order in which frames appear.
    for frame_idx in range(len(true_turn)):
        # For each frame
        true_frame = true_turn[frame_idx]
        if frame_idx >= len(pred_turn):
            pred_frame = {}
        else:
            pred_frame = pred_turn[frame_idx]

        count_dict = add_dicts(
            count_dict, evaluate_frame(true_frame, pred_frame, strict=False)
        )

    return count_dict

def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1

def b_arr(n_total, n_pos):
    out = np.zeros(int(n_total))
    out[: int(n_pos)] = 1.0
    return out
def b_stderr(n_total, n_pos):
    return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)
def d_f1(n_true, n_pred, n_correct):
    # 1/r + 1/p = 2/F1
    # dr / r^2 + dp / p^2 = 2dF1 /F1^2
    # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2)
    dr = b_stderr(n_true, n_correct)
    dp = b_stderr(n_pred, n_correct)

    r = n_correct / n_true
    p = n_correct / n_pred
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0

    d_f1 = 0.5 * f1 ** 2 * (dr / r ** 2 + dp / p ** 2)
    return d_f1

def evaluate_from_flat_list(d_true, d_pred):
    """
    <list>d_true and <list>d_pred are in the following format:
    (Each element represents a single turn, with (multiple) frames)
    [
        [
            {
                'act': <str>,
                'slots': [
                    [
                        SLOT_NAME, SLOT_VALUE
                    ],
                    ...
                ],
                'request_slots': [ SLOT_NAME, ... ],
                'objects': [ <int> ],
                'disambiguation_candidates': [ <int> ]
            },
            [End of a frame]
            ...
        ],
        [End of a turn]
        ...
    ]
    """
    c = initialize_count_dict()

    # Count # corrects & # wrongs
    for i in range(len(d_true)):
        true_turn = d_true[i]
        pred_turn = d_pred[i]
        turn_evaluation = evaluate_turn(true_turn, pred_turn)

        c = add_dicts(c, turn_evaluation)

    # Calculate metrics
    joint_accuracy = c["n_correct_beliefs"] / c["n_frames"]

    act_rec, act_prec, act_f1 = rec_prec_f1(
        n_correct=c["n_correct_acts"], n_true=c["n_true_acts"], n_pred=c["n_pred_acts"]
    )

    slot_rec, slot_prec, slot_f1 = rec_prec_f1(
        n_correct=c["n_correct_slots"],
        n_true=c["n_true_slots"],
        n_pred=c["n_pred_slots"],
    )

    request_slot_rec, request_slot_prec, request_slot_f1 = rec_prec_f1(
        n_correct=c["n_correct_request_slots"],
        n_true=c["n_true_request_slots"],
        n_pred=c["n_pred_request_slots"],
    )

    object_rec, object_prec, object_f1 = rec_prec_f1(
        n_correct=c["n_correct_objects"],
        n_true=c["n_true_objects"],
        n_pred=c["n_pred_objects"],
    )

    disamb_candidate_rec, disamb_candidate_prec, disamb_candidate_f1 = rec_prec_f1(
        n_correct=c["n_correct_disamb_candidates"],
        n_true=c["n_true_disamb_candidates"],
        n_pred=c["n_pred_disamb_candidates"],
    )    

    # Calculate std err
    act_f1_stderr = d_f1(c["n_true_acts"], c["n_pred_acts"], c["n_correct_acts"])
    slot_f1_stderr = d_f1(c["n_true_slots"], c["n_pred_slots"], c["n_correct_slots"])
    request_slot_f1_stderr = d_f1(
        c["n_true_request_slots"],
        c["n_pred_request_slots"],
        c["n_correct_request_slots"],
    )
    object_f1_stderr = d_f1(
        c["n_true_objects"],
        c["n_pred_objects"],
        c["n_correct_objects"]
    )
    disamb_candidate_f1_stderr = d_f1(
        c["n_true_disamb_candidates"],
        c["n_pred_disamb_candidates"],
        c["n_correct_disamb_candidates"]
    )    

    return {
        "joint_accuracy": joint_accuracy,
        "act_rec": act_rec,
        "act_prec": act_prec,
        "act_f1": act_f1,
        "act_f1_stderr": act_f1_stderr,
        "slot_rec": slot_rec,
        "slot_prec": slot_prec,
        "slot_f1": slot_f1,
        "slot_f1_stderr": slot_f1_stderr,
        "request_slot_rec": request_slot_rec,
        "request_slot_prec": request_slot_prec,
        "request_slot_f1": request_slot_f1,
        "request_slot_f1_stderr": request_slot_f1_stderr,
        "object_rec": object_rec,
        "object_prec": object_prec,
        "object_f1": object_f1,
        "object_f1_stderr": object_f1_stderr,
        "disamb_candidate_rec": disamb_candidate_rec,
        "disamb_candidate_prec": disamb_candidate_prec,
        "disamb_candidate_f1": disamb_candidate_f1,
        "disamb_candidate_f1_stderr": disamb_candidate_f1_stderr,        
    }


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    dialog_act_regex = re.compile(
        r'([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\> *\|([^\]]*)\|'
    )    
    
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= (\[(.*)\]|[^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")
    disamb_candidate_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []

    # Parse
    splits = to_parse.strip().split(START_BELIEF_STATE)
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split(END_OF_BELIEF)

        if len(splits) == 2:
            # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
            to_parse = splits[0].strip()

            for dialog_act in dialog_act_regex.finditer(to_parse):
                d = {
                    "act": dialog_act.group(1),
                    "slots": [],
                    "request_slots": [],
                    "objects": [],
                    "disambiguation_candidates": [],
                }

                for slot in slot_regex.finditer(dialog_act.group(2)):
                    d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])

                for request_slot in request_regex.finditer(dialog_act.group(3)):
                    d["request_slots"].append(request_slot.group(1).strip())

                for object_id in object_regex.finditer(dialog_act.group(4)):
                    str_object_id = object_id.group(1).strip()

                    try:
                        # Object ID should always be <int>.
                        int_object_id = int(str_object_id)
                        d["objects"].append(int_object_id)
                    except:
                        pass

                for disamb_candidate_id in disamb_candidate_regex.finditer(dialog_act.group(5)):
                    str_disamb_candidate_id = disamb_candidate_id.group(1).strip()

                    try:
                        # disamb_candidate ID should always be <int>.
                        int_disamb_candidate_id = int(str_disamb_candidate_id)
                        d["disambiguation_candidates"].append(int_disamb_candidate_id)
                    except:
                        pass

                if d != {}:
                    belief.append(d)

    return belief

def eval_simmic21rg(ret, output_path_report=None):

    preds_results = []
    labels_results = []
    for id, values in ret.items():
        for value in values:
            preds_results.append(parse_flattened_result(value['pred']))
            labels_results.append(parse_flattened_result(value['label']))

    report = evaluate_from_flat_list(labels_results, preds_results)
    print(report)
    if output_path_report:
        with open(output_path_report, "w") as f_out:
            json.dump(report, f_out)