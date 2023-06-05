from dataclasses import dataclass
from typing import Union, List, Dict, Optional
from transformers.pipelines.text2text_generation import ReturnType, Text2TextGenerationPipeline
from transformers.tokenization_utils import TruncationStrategy
from transformers.tokenization_utils_base import BatchEncoding
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from seq2seq.utils.dataset import serialize_schema
from seq2seq.utils.spider import spider_get_input
from seq2seq.utils.cosql import cosql_get_input


@dataclass
class Text2SQLInput(object):
    utterance: str
    db_id: str


class Text2SQLGenerationPipeline(Text2TextGenerationPipeline):
    """
    Pipeline for text-to-SQL generation using seq2seq models.

    The models that this pipeline can use are models that have been fine-tuned on the Spider text-to-SQL task.

    Usage::

        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        db_path = ... path to "concert_singer" parent folder
        text2sql_generator = Text2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=db_path,
        )
        text2sql_generator(inputs=Text2SQLInput(utterance="How many singers do we have?", db_id="concert_singer"))
    """

    def __init__(self, *args, **kwargs):
        self.db_path: str = kwargs.pop("db_path")
        self.prefix: Optional[str] = kwargs.pop("prefix", None)
        self.normalize_query: bool = kwargs.pop("normalize_query", True)
        self.schema_serialization_type: str = kwargs.pop("schema_serialization_type", "peteshaw")
        self.schema_serialization_randomized: bool = kwargs.pop("schema_serialization_randomized", False)
        self.schema_serialization_with_db_id: bool = kwargs.pop("schema_serialization_with_db_id", True)
        self.schema_serialization_with_db_content: bool = kwargs.pop("schema_serialization_with_db_content", True)
        self.schema_cache: Dict[str, dict] = dict()
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: Union[Text2SQLInput, List[Text2SQLInput]], *args, **kwargs):
        r"""
        Generate the output SQL expression(s) using text(s) given as inputs.

        Args:
            inputs (:obj:`Text2SQLInput` or :obj:`List[Text2SQLInput]`):
                Input text(s) for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_sql** (:obj:`str`, present when ``return_text=True``) -- The generated SQL.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated SQL.
        """
        result = super().__call__(inputs, *args, **kwargs)
        if (
            isinstance(inputs, list)
            and all(isinstance(el, Text2SQLInput) for el in inputs)
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        return result

    def preprocess(
        self,
        inputs: Union[Text2SQLInput, List[Text2SQLInput]],
        *args,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        **kwargs
    ) -> BatchEncoding:
        encodings = self._parse_and_tokenize(inputs, *args, truncation=truncation, **kwargs)
        return encodings

    def _parse_and_tokenize(
        self,
        inputs: Union[Text2SQLInput, List[Text2SQLInput]],
        *args,
        truncation: TruncationStrategy
    ) -> BatchEncoding:
        if isinstance(inputs, list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            inputs = [self._pre_process(input=input) for input in inputs]
            padding = True
        elif isinstance(inputs, Text2SQLInput):
            inputs = self._pre_process(input=inputs)
            padding = False
        else:
            raise ValueError(
                f" `inputs`: {inputs} have the wrong format. The should be either of type `Text2SQLInput` or type `List[Text2SQLInput]`"
            )
        encodings = self.tokenizer(inputs, padding=padding, truncation=truncation, return_tensors=self.framework)
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in encodings:
            del encodings["token_type_ids"]
        return encodings

    def _pre_process(self, input: Text2SQLInput) -> str:
        prefix = self.prefix if self.prefix is not None else ""
        if input.db_id not in self.schema_cache:
            self.schema_cache[input.db_id] = get_schema(db_path=self.db_path, db_id=input.db_id)
        schema = self.schema_cache[input.db_id]
        if hasattr(self.model, "add_schema"):
            self.model.add_schema(db_id=input.db_id, db_info=schema)
        serialized_schema = serialize_schema(
            question=input.utterance,
            db_path=self.db_path,
            db_id=input.db_id,
            db_column_names=schema["db_column_names"],
            db_table_names=schema["db_table_names"],
            schema_serialization_type=self.schema_serialization_type,
            schema_serialization_randomized=self.schema_serialization_randomized,
            schema_serialization_with_db_id=self.schema_serialization_with_db_id,
            schema_serialization_with_db_content=self.schema_serialization_with_db_content,
            normalize_query=self.normalize_query,
        )
        return spider_get_input(question=input.utterance, serialized_schema=serialized_schema, prefix=prefix)

    def postprocess(self, model_outputs: dict, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": model_outputs}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                    .split("|", 1)[-1]
                    .strip()
                }
            records.append(record)
        return records


@dataclass
class ConversationalText2SQLInput(object):
    utterances: List[str]
    db_id: str


class ConversationalText2SQLGenerationPipeline(Text2TextGenerationPipeline):
    """
    Pipeline for conversational text-to-SQL generation using seq2seq models.

    The models that this pipeline can use are models that have been fine-tuned on the CoSQL SQL-grounded dialogue state tracking task.

    Usage::

        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        db_path = ... path to "concert_singer" parent folder
        conv_text2sql_generator = ConversationalText2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=db_path,
        )
        conv_text2sql_generator(inputs=ConversationalText2SQLInput(utterances=["How many singers do we have?"], db_id="concert_singer"))
    """

    def __init__(self, *args, **kwargs):
        self.db_path: str = kwargs.pop("db_path")
        self.prefix: Optional[str] = kwargs.pop("prefix", None)
        self.normalize_query: bool = kwargs.pop("normalize_query", True)
        self.schema_serialization_type: str = kwargs.pop("schema_serialization_type", "peteshaw")
        self.schema_serialization_randomized: bool = kwargs.pop("schema_serialization_randomized", False)
        self.schema_serialization_with_db_id: bool = kwargs.pop("schema_serialization_with_db_id", True)
        self.schema_serialization_with_db_content: bool = kwargs.pop("schema_serialization_with_db_content", True)
        self.schema_cache: Dict[str, dict] = dict()
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: Union[ConversationalText2SQLInput, List[ConversationalText2SQLInput]], *args, **kwargs):
        r"""
        Generate the output SQL expression(s) using conversation(s) given as inputs.

        Args:
            inputs (:obj:`ConversationalText2SQLInput` or :obj:`List[ConversationalText2SQLInput]`):
                Input conversation(s) for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_sql** (:obj:`str`, present when ``return_text=True``) -- The generated SQL.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated SQL.
        """
        result = super().__call__(inputs, *args, **kwargs)
        if (
            isinstance(inputs, list)
            and all(isinstance(el, ConversationalText2SQLInput) for el in inputs)
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        return result

    def preprocess(
        self,
        inputs: Union[ConversationalText2SQLInput, List[ConversationalText2SQLInput]],
        *args,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        **kwargs,
    ) -> BatchEncoding:
        encodings = self._parse_and_tokenize(inputs, *args, truncation=truncation, **kwargs)
        return encodings

    def _parse_and_tokenize(
        self,
        inputs: Union[ConversationalText2SQLInput, List[ConversationalText2SQLInput]],
        *args,
        truncation: TruncationStrategy,
    ) -> BatchEncoding:
        if isinstance(inputs, list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            inputs = [self._pre_process(input=input) for input in inputs]
            padding = True
        elif isinstance(inputs, ConversationalText2SQLInput):
            inputs = self._pre_process(input=inputs)
            padding = False
        else:
            raise ValueError(
                f" `inputs`: {inputs} have the wrong format. The should be either of type `ConversationalText2SQLInput` or type `List[ConversationalText2SQLInput]`"
            )
        encodings = self.tokenizer(inputs, padding=padding, truncation=truncation, return_tensors=self.framework)
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in encodings:
            del encodings["token_type_ids"]
        return encodings

    def _pre_process(self, input: ConversationalText2SQLInput) -> str:
        prefix = self.prefix if self.prefix is not None else ""
        if input.db_id not in self.schema_cache:
            self.schema_cache[input.db_id] = get_schema(db_path=self.db_path, db_id=input.db_id)
        schema = self.schema_cache[input.db_id]
        if hasattr(self.model, "add_schema"):
            self.model.add_schema(db_id=input.db_id, db_info=schema)
        serialized_schema = serialize_schema(
            question=" | ".join(input.utterances),
            db_path=self.db_path,
            db_id=input.db_id,
            db_column_names=schema["db_column_names"],
            db_table_names=schema["db_table_names"],
            schema_serialization_type=self.schema_serialization_type,
            schema_serialization_randomized=self.schema_serialization_randomized,
            schema_serialization_with_db_id=self.schema_serialization_with_db_id,
            schema_serialization_with_db_content=self.schema_serialization_with_db_content,
            normalize_query=self.normalize_query,
        )
        return cosql_get_input(utterances=input.utterances, serialized_schema=serialized_schema, prefix=prefix)

    def postprocess(self, model_outputs: dict, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": model_outputs}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                    .split("|", 1)[-1]
                    .strip()
                }
            records.append(record)
        return records


def get_schema(db_path: str, db_id: str) -> dict:
    schema = dump_db_json_schema(db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id)
    return {
        "db_table_names": schema["table_names_original"],
        "db_column_names": {
            "table_id": [table_id for table_id, _ in schema["column_names_original"]],
            "column_name": [column_name for _, column_name in schema["column_names_original"]],
        },
        "db_column_types": schema["column_types"],
        "db_primary_keys": {"column_id": [column_id for column_id in schema["primary_keys"]]},
        "db_foreign_keys": {
            "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
            "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]],
        },
    }
