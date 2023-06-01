#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.
  Util functions for evaluating the DST model predictions.
    The script includes a main function which takes
    the original JSON data file and the predicted model output file
    (in the same format), and outputs the report.
"""
import argparse
import json
import copy
import numpy as np
from pace.utils.eval_mmconv_rg import normalize_sentence

def evaluate_from_json(d_true, d_pred):
    """
    <list>d_true and <list>d_pred are in the following format:
    (Equivalent to "dialogue_data" field in the input data JSON file)
    [
        {
            "dialogue": [
                {
                    "transcript_annotated":  {
                        'act': <str>,
                        'act_attributes': {
                            'slot_values': {
                                SLOT_NAME: SLOT_VALUE,
                                ...
                            },
                            'request_slots': [
                                SLOT_NAME, ...
                            ],
                            'objects': [ <int> ]
                        }
                    },
                    ...
                }
                [End of a turn]
                ...
            ],
        }
        [End of a dialogue]
        ...
    ]
    """
    d_true_flattened = []
    d_pred_flattened = []

    for i in range(len(d_true)):
        # Iterate through each dialog
        dialog_true = d_true[i]["dialogue"]
        dialog_pred = d_pred[i]["dialogue"]

        # ** Assumes dialogue_idx and turn_idx are ordered
        # exactly the same for `dialog_true` and `dialog_pred`
        _ = d_true[i]["dialogue_idx"]

        for j in range(len(dialog_true)):
            # Iterate through each turn
            turn_true = reformat_turn(dialog_true[j]["transcript_annotated"])
            turn_pred = reformat_turn(dialog_pred[j]["transcript_annotated"])

            d_true_flattened.append(turn_true)
            d_pred_flattened.append(turn_pred)

    return evaluate_from_flat_list(d_true_flattened, d_pred_flattened)


def reformat_turn(t):
    frame = {
        'act': t['act'],
        'slots': [[s,v] for s, v in t['act_attributes']['slot_values'].items()],
        'request_slots': t['act_attributes']['request_slots'],
        'objects': t['act_attributes']['objects'],
    }
    return [frame]


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
                'objects': [ <int> ]
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

    # Calculate std err
    act_f1_stderr = d_f1(c["n_true_acts"], c["n_pred_acts"], c["n_correct_acts"])
    slot_f1_stderr = d_f1(c["n_true_slots"], c["n_pred_slots"], c["n_correct_slots"])
    request_slot_f1_stderr = d_f1(
        c["n_true_request_slots"],
        c["n_pred_request_slots"],
        c["n_correct_request_slots"],
    )
    # object_f1_stderr = d_f1(
    #     c["n_true_objects"], c["n_pred_objects"], c["n_correct_objects"]
    # )

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
        # "object_rec": object_rec,
        # "object_prec": object_prec,
        # "object_f1": object_f1,
        # "object_f1_stderr": object_f1_stderr,
    }


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
    true_act = ' '.join(normalize_sentence(true_act)) if true_act != None else None
    pred_act = ' '.join(normalize_sentence(pred_act)) if pred_act != None else None
    b_correct_act = true_act == pred_act
    count_dict["n_correct_acts"] += b_correct_act
    count_dict["n_true_acts"] += "act" in true_frame
    count_dict["n_pred_acts"] += "act" in pred_frame

    # (1) Compare Slots
    #true_frame_slot_values = {f"{k}={v}" for k, v in true_frame.get("slots", [])}
    #pred_frame_slot_values = {f"{k}={v}" for k, v in pred_frame.get("slots", [])}

    true_frame_slot_values = set()
    pred_frame_slot_values = set()

    do_write = False
    for k, v in true_frame.get("slots", []):
        if k in set(['availableSizes']):
            # For availableSizes, we expect that the type is <list>.
            # Otherwise, try converting it to a <list>.
            if type(v) == str:
                try:
                    v = v.split()
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
                    v = v.split()
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
        if len(true_frame_slot_values.intersection(pred_frame_slot_values)) < len(true_frame_slot_values):
            do_write = True
    
    if do_write:
        with open("tmp_badcase_simmc2_dst.txt","a+") as out:
            out.write(f"{str(true_frame_slot_values)}\t{str(pred_frame_slot_values)}\n")
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

    # (4) Joint
    count_dict["n_correct_beliefs"] += (
        b_correct_act
        and true_frame_slot_values == pred_frame_slot_values
        and true_frame_request_slot_values == pred_frame_request_slot_values
        and true_frame_object_values == pred_frame_object_values
    )

    return count_dict


def add_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1}


def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


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


def b_stderr(n_total, n_pos):
    return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)


def b_arr(n_total, n_pos):
    out = np.zeros(int(n_total))
    out[: int(n_pos)] = 1.0
    return out


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
        "n_correct_beliefs": 0.0,
    }
    return copy.deepcopy(c)


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_target", help="path for target (.json)")
    parser.add_argument(
        "--input_path_predicted", help="path for model prediction output (.json)"
    )
    parser.add_argument(
        "--output_path_report", help="path for saving evaluation summary (.json)"
    )

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Read the JSON file input
    # json_predicted must have the same structure as the original input JSON
    # e.g. {'dialogue_data': [ ... ]}
    json_target = json.load(open(input_path_target, "r"))
    json_predicted = json.load(open(input_path_predicted, "r"))

    # Evaluate
    report = evaluate_from_json(
        json_target["dialogue_data"], json_predicted["dialogue_data"]
    )

    # Save report
    with open(output_path_report, "w") as f_out:
        json.dump(report, f_out)