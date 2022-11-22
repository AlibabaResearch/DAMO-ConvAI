import argparse
from typing import List, Dict, Any, Tuple
import pickle as pkl
import tqdm
from .exec_eval import exec_on_db, result_eq
import os
from collections import defaultdict
import time
from multiprocessing import cpu_count, Pool, Manager
from itertools import repeat

NUM_PROCESSES = cpu_count() // 3
if NUM_PROCESSES == 0:
    NUM_PROCESSES = 1
MULTIPLICATIVE_OVERHEAD = 3
ADDITIVE_OVERHEAD = 30
GOLD_TIMEOUT = 100

cache_path = "cache.pkl"
m = Manager()
cache = m.dict()


def load_predictions(f_path: str) -> List[str]:
    preds = []
    with open(f_path, "r") as in_file:
        for l in in_file:
            preds.append(l.strip())
    return preds


def acc(l, idxes=None):
    if idxes is None:
        idxes = [_ for _ in range(len(l))]
    c = 0
    for idx in idxes:
        if l[idx]:
            c += 1
    return float(c) / len(idxes)


# the input is a tuple of gold_dict, model prediction and whether to use cache
# and teh output is whether the model prediction passes the entire test suite
def judge(args: Tuple[Dict[str, Any], str, bool]) -> bool:
    gold_dict, pred, use_cache = args

    testsuite_paths = gold_dict["testsuite"]
    gold_query = gold_dict["query"]
    order_matters = "order by" in gold_query.lower()
    db_path = gold_dict["db_path"]

    # if already computed sometime before
    # and cache allowed, directly return the result
    k = (db_path, gold_query, pred)
    if use_cache and k in cache:
        return cache[k]

    pass_all_testcase = True
    for testcase_path in testsuite_paths:

        start = time.time()
        flg, gold_result = exec_on_db(testcase_path, gold_query, timeout=GOLD_TIMEOUT)
        duration = time.time() - start
        timeout = ADDITIVE_OVERHEAD + MULTIPLICATIVE_OVERHEAD * duration

        if flg != "result":
            print("Warning: executing gold query results in an exception")
            continue
        flg, pred_result = exec_on_db(testcase_path, pred, timeout=int(timeout))
        if flg != "result":
            pass_all_testcase = False
            break
        if not result_eq(gold_result, pred_result, order_matters):
            pass_all_testcase = False
            break

    # save the results in the cache
    if use_cache:
        cache[k] = pass_all_testcase
    return pass_all_testcase


# cache is a dictionary
# the key is a ternary tuple (empty_database_path, SQL1, SQL2)
# the value is whether SQL1 and SQL2 are equivalent, judged by the test suites
def load_cache() -> Dict[Tuple[str, str, str], bool]:
    if os.path.exists(cache_path):
        d = m.dict(pkl.load(open(cache_path, "rb")))
        for k, v in d.items():
            cache[k] = v
    return cache


# dump the cache
def save_cache():
    pkl.dump(dict(cache), open(cache_path, "wb"))


def main(
    preds: List[str],
    gold_file: str = "classical_test.pkl",
    verbose: bool = True,
    num_processes: int = NUM_PROCESSES,
    subset: str = "full",
    use_cache: bool = True,
) -> List[bool]:
    gold_dicts = pkl.load(open(gold_file, "rb"))
    if subset != "full":
        gold_dicts = [
            d
            for d in gold_dicts
            if d["db_path"] == "database/{db_id}/{db_id}.sqlite".format(db_id=subset)
        ]
    assert len(gold_dicts) == len(
        preds
    ), "number of gold and prediction should be equal"
    group_name2idxes = defaultdict(list)

    for idx, gold_dict in enumerate(gold_dicts):
        group_name2idxes[gold_dict["db_id"]].append(idx)

    with Pool(num_processes) as pool:
        result = list(
            tqdm.tqdm(
                pool.imap(judge, zip(gold_dicts, preds, repeat(use_cache, len(preds)))),
                total=len(gold_dicts),
            )
        )

    if verbose:
        print("overall accuracy: ", acc(result))
        for group, idxes in group_name2idxes.items():
            print("accuracy for ", group, acc(result, idxes))
    return result


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        dest="gold",
        type=str,
        default="classical_test.pkl",
        help="the path to the predicted queries",
    )
    parser.add_argument(
        "--pred", dest="pred", type=str, help="the path to the predicted queries"
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="the output file path"
    )
    parser.add_argument(
        "--num_processes", default=NUM_PROCESSES, help="number of processes to use"
    )
    parser.add_argument(
        "--subset",
        default="full",
        choices=(
            "atis",
            "advising",
            "academic",
            "imdb",
            "restaurants",
            "geography",
            "scholar",
            "yelp",
            "full",
        ),
        help="which subset to evaluate on.",
    )
    parser.add_argument(
        "--disable_cache",
        default=False,
        action="store_true",
        help="whether to directly apply previously computed result and cache the current results. "
        "use this flag to disable caching.",
    )
    args = parser.parse_args()

    preds = load_predictions(args.pred)
    assert not os.path.exists(args.out_file), (
        "output file path %s already exists" % args.out_file
    )

    use_cache = not args.disable_cache
    if use_cache:
        load_cache()

    result = main(
        preds=preds,
        gold_file=args.gold,
        verbose=True,
        num_processes=args.num_processes,
        subset=args.subset,
        use_cache=use_cache,
    )
    pkl.dump(result, open(args.out_file, "wb"))
    print("total time used: ", time.time() - start)

    if use_cache:
        save_cache()
