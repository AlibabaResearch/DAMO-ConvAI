from utils.args import parse_arguments
from utils.metric import cal_metric


if __name__ == '__main__':
    args = parse_arguments()

    print("------------------ All metrics: ------------------")
    cal_metric(args, tag="eval_response")
    print("")

    print(f"------------------ Level metrics: ------------------")
    for level in [1, 2, 3, 4]:
        print(f"------------------ Level {level} metrics: ------------------")
        cal_metric(args, tag="eval_response", level=level)
    print("")

    print(f"------------------ Set metrics: ------------------")
    for set in [1, 2, 3, 4]:
        print(f"------------------ Set {set} metrics ------------------")
        for level in [1, 2, 3, 4]:
            cal_metric(args, tag="eval_response", set=set, level=level)
        cal_metric(args, tag="eval_response", set=set, level=None)
        print("")
