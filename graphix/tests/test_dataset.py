from seq2seq.utils.picard_model_wrapper import get_picard_schema
import pytest
from dataclasses import replace
from transformers.models.auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import DataArguments, DataTrainingArguments
from seq2seq.utils.dataset_loader import load_dataset


@pytest.fixture
def training_args(tmpdir, split) -> TrainingArguments:
    if split == "train":
        do_train = True
        do_eval = False
    elif split == "eval":
        do_train = False
        do_eval = True
    else:
        raise NotImplementedError()
    return TrainingArguments(
        output_dir=str(tmpdir),
        do_train=do_train,
        do_eval=do_eval,
        do_predict=False,
    )


@pytest.fixture(params=[True, False])
def schema_serialization_with_db_content(request) -> bool:
    return request.param


@pytest.fixture
def data_training_args(schema_serialization_with_db_content: bool) -> DataTrainingArguments:
    return DataTrainingArguments(
        max_source_length=4096,
        max_target_length=4096,
        schema_serialization_type="peteshaw",
        schema_serialization_with_db_id=True,
        schema_serialization_with_db_content=schema_serialization_with_db_content,
        normalize_query=True,
        target_with_db_id=True,
    )


@pytest.fixture(params=["cosql", "spider", "cosql+spider"])
def data_args(request) -> DataArguments:
    return DataArguments(dataset=request.param)


@pytest.fixture(params=["train", "eval"])
def split(request) -> str:
    return request.param


@pytest.fixture
def expected_max_input_ids_len(data_args: DataArguments, split: str, schema_serialization_with_db_content: bool) -> int:
    def _expected_max_input_ids_len(_data_args: DataArguments) -> int:
        if _data_args.dataset == "spider" and split == "train" and schema_serialization_with_db_content:
            return 1927
        elif _data_args.dataset == "spider" and split == "train" and not schema_serialization_with_db_content:
            return 1892
        elif _data_args.dataset == "spider" and split == "eval" and schema_serialization_with_db_content:
            return 468
        elif _data_args.dataset == "spider" and split == "eval" and not schema_serialization_with_db_content:
            return 468
        elif _data_args.dataset == "cosql" and split == "train" and schema_serialization_with_db_content:
            return 2227
        elif _data_args.dataset == "cosql" and split == "train" and not schema_serialization_with_db_content:
            return 1984
        elif _data_args.dataset == "cosql" and split == "eval" and schema_serialization_with_db_content:
            return 545
        elif _data_args.dataset == "cosql" and split == "eval" and not schema_serialization_with_db_content:
            return 545
        elif _data_args.dataset == "cosql+spider":
            return max(
                _expected_max_input_ids_len(_data_args=replace(_data_args, dataset="spider")),
                _expected_max_input_ids_len(_data_args=replace(_data_args, dataset="cosql")),
            )
        else:
            raise NotImplementedError()

    return _expected_max_input_ids_len(_data_args=data_args)


@pytest.fixture
def expected_max_labels_len(data_args: DataArguments, split: str, schema_serialization_with_db_content: bool) -> int:
    def _expected_max_labels_len(_data_args: DataArguments) -> int:
        if _data_args.dataset == "spider" and split == "train" and schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "spider" and split == "train" and not schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "spider" and split == "eval" and schema_serialization_with_db_content:
            return 166
        elif _data_args.dataset == "spider" and split == "eval" and not schema_serialization_with_db_content:
            return 166
        elif _data_args.dataset == "cosql" and split == "train" and schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "cosql" and split == "train" and not schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "cosql" and split == "eval" and schema_serialization_with_db_content:
            return 210
        elif _data_args.dataset == "cosql" and split == "eval" and not schema_serialization_with_db_content:
            return 210
        elif _data_args.dataset == "cosql+spider":
            return max(
                _expected_max_labels_len(_data_args=replace(_data_args, dataset="spider")),
                _expected_max_labels_len(_data_args=replace(_data_args, dataset="cosql")),
            )
        else:
            raise NotImplementedError()

    return _expected_max_labels_len(_data_args=data_args)


@pytest.fixture
def model_name_or_path() -> str:
    return "t5-small"


@pytest.fixture
def model_args(model_name_or_path: str) -> ModelArguments:
    return ModelArguments(model_name_or_path=model_name_or_path)


@pytest.fixture
def tokenizer(model_args: ModelArguments) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(model_args.model_name_or_path)


def test_dataset_loader(
    data_args: DataArguments,
    split: str,
    expected_max_input_ids_len: int,
    expected_max_labels_len: int,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizerFast,
) -> None:

    _metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    if split == "train":
        dataset = dataset_splits.train_split.dataset
        schemas = dataset_splits.train_split.schemas
    elif split == "eval":
        dataset = dataset_splits.eval_split.dataset
        schemas = dataset_splits.eval_split.schemas
    elif split == "test":
        dataset = dataset_splits.test_split.dataset
        schemas = dataset_splits.test_split.schemas
    else:
        raise NotImplementedError()
    
    print(list(schemas.keys()))

    for db_id, schema in schemas.items():
        if db_id in ["academic", "scholar", "geo", "imdb", "yelp", "restaurants", "orchestra", "wta_1", "singer", "cre_Doc_Template_Mgt", "dog_kennels", "poker_player", "museum_visit", "student_transcripts_tracking"]:
            sql_schema = get_picard_schema(**schema)
            c_names_str = ", ".join((f"(\"{c_id}\", \"{c_name}\")" for c_id, c_name in sql_schema.columnNames.items()))
            c_type_str = lambda c_type: str(c_type).replace('.', "_")
            c_types_str = ", ".join((f"(\"{c_id}\", {c_type_str(c_type)})" for c_id, c_type in sql_schema.columnTypes.items()))
            t_names_str = ", ".join((f"(\"{t_id}\", \"{t_name}\")" for t_id, t_name in sql_schema.tableNames.items()))
            c_to_t_str = ", ".join((f"(\"{c_id}\", \"{t_id}\")" for c_id, t_id in sql_schema.columnToTable.items()))
            c_ids_str = lambda c_ids: ", ".join((f"\"{c_id}\"" for c_id in c_ids))
            t_to_cs_str = ", ".join((f"(\"{t_id}\", [{c_ids_str(c_ids)}])" for t_id, c_ids in sql_schema.tableToColumns.items()))
            fks_str = ", ".join((f"(\"{c_id}\", \"{other_cid}\")" for c_id, other_cid in sql_schema.foreignKeys.items()))
            pks_str = ", ".join((f"\"{pk}\"" for pk in sql_schema.primaryKeys))
            schema_str = "SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}" 
            print(f"""
{db_id}:
  let columnNames = HashMap.fromList [{c_names_str}]
      columnTypes = HashMap.fromList [{c_types_str}]
      tableNames = HashMap.fromList [{t_names_str}]
      columnToTable = HashMap.fromList [{c_to_t_str}]
      tableToColumns = HashMap.fromList [{t_to_cs_str}]
      foreignKeys = HashMap.fromList [{fks_str}]
      primaryKeys = [{pks_str}]
   in {schema_str}
            """)

    max_input_ids_len = max(len(item["input_ids"]) for item in dataset)
    assert max_input_ids_len == expected_max_input_ids_len

    max_labels_len = max(len(item["labels"]) for item in dataset)
    assert max_labels_len == expected_max_labels_len
