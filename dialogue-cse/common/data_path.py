#!/usr/bin/python
#_*_coding:utf-8_*_

DATA_SUFFIX = "jddc"
PREPROCESS_SESSION_FILE_PATH = "./dataset/preprocess_session_%s.txt" % DATA_SUFFIX


PREPROCESS_SESSION_FILE_COLUMN_NUM = 4

SESSION_QUERY_ROLE = "0"
SESSION_ANSWER_ROLE = "1"
REMAIN_KW_LIST = None
ENABLE_SMOOTH = False

PREPROCESS_SESSION_FILE_SESSION_ID_FIELD = 0
PREPROCESS_SESSION_FILE_ROLE_FIELD = 1
PREPROCESS_SESSION_FILE_TEXT_FIELD = 2
PREPROCESS_SESSION_FILE_WEIGHT_FIELD = 3
PREPROCESS_SESSION_FILE_FEATURE_FIELD = 4
PREPROCESS_SESSION_FILE_CLUSTER_FIELD = 5
PREPROCESS_SESSION_FILE_DISTANCE_FIELD = 6
PREPROCESS_SESSION_FILE_SESSION_CLUSTER_FIELD = 7
PREPROCESS_SESSION_FILE_SESSION_DISTANCE_FIELD = 8

SESSION_FILE_ID_FIELD = 0
SESSION_FILE_SESSION_ID_FIELD = 1
SESSION_FILE_QUERY_TURN_FIELD = 2
SESSION_FILE_ANSWER_TURN_FIELD = 3
SESSION_FILE_TOPIC_FIELD = 4
SESSION_FILE_QUERY_FIELD = 5
SESSION_FILE_ANSWER_FIELD = 6
SESSION_FILE_QUERY_FEATURE_FIELD = 7
SESSION_FILE_ANSWER_FEATURE_FIELD = 8
SESSION_FILE_QUERY_ID_FIELD = 9
SESSION_FILE_ANSWER_ID_FIELD = 10
SESSION_FILE_ANSWER_QUALITY_FIELD = 11
SESSION_FILE_VIEW_FIELD = 12
SESSION_FILE_ADOPTION_FIELD = 13

QUERY_FEATURE_LEN = 768
ANSWER_FEATURE_LEN = 768

QUERY_CLUSTERING_FILE = "/data/human_session/query_clustering.out"
ANSWER_CLUSTERING_FILE = "/data/human_session/answer_clustering.out"
SESSION_CLUSTERING_FILE = "/data/human_session/session_clustering.out"

BERT_TRAIN_FILE = "/data/pretrain_data/train.txt"

ST_QA_TRAIN_FILE = "/data/similarity_data/train_data.txt"
ST_QA_TEST_FILE = "/data/similarity_data/test_data.txt"

MT_QA_TRAIN_FILE = "/data/context_matching_data/train.txt"
MT_QA_TEST_FILE = "/data/context_matching_data/test.txt"

ALIGNMENT_DATA_FILE_PATH = "/data/alignment_data/alignment_data.txt"
ALIGNMENT_DATA_FIELD_NUM = 4
ALIGNMENT_TRAIN_DATA_FILE = "/data/alignment_data/train_data.txt"
ALIGNMENT_TEST_DATA_FILE = "/data/alignment_data/test_data.txt"

QA_REPRESENTATION_TRAIN_DATA_FILE = "/data/qa_representation/train_data.txt"
QA_REPRESENTATION_TEST_DATA_FILE = "/data/qa_representation/test_data.txt"

SESSION_REPRESENTATION_TRAIN_DATA_FILE = "/data/session_representation/train_data.txt"
SESSION_REPRESENTATION_TEST_DATA_FILE = "/data/session_representation/test_data.txt"

PROBING_TRAIN_DATA_FILE = "/data/probing/train_data.txt"
PROBING_TEST_DATA_FILE = "/data/probing/test_data.txt"


def add_path_suffix(path_variable, suffix):
    """
    增加分区后缀
    """
    return str(path_variable) + "." + str(suffix)
