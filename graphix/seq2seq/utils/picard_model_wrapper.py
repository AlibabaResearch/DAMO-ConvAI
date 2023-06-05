from copy import deepcopy
from typing import Optional, Union, Any, Callable, AsyncContextManager, List, Dict
from dataclasses import dataclass, field
import collections
import asyncio
import sys
import subprocess
import warnings
import time
from tenacity import retry, wait_random_exponential, stop_after_delay, before_sleep_log
import torch
from transformers import LogitsProcessorList
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput
from transformers.generation_logits_process import LogitsProcessor
from transformers.file_utils import copy_func
from transformers.models.auto.auto_factory import _get_model_class
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto import AutoModelForSeq2SeqLM
import logging
import pdb
logger = logging.getLogger(__name__)

try:
    from picard.clients import Picard
    from picard.types import (
        FeedException,
        FeedTimeoutFailure,
        FeedParseFailure,
        FeedPartialSuccess,
        FeedCompleteSuccess,
        SQLSchema,
        RegisterSQLSchemaException,
        Mode,
        ColumnType,
    )
    from thrift.py3.client import get_client
    from thrift.py3.common import Protocol
    from thrift.py3.exceptions import TransportError

    picard_available = True
except:
    logger.warning("Picard is not available.")
    Picard = Any
    SQLSchema = Any
    RegisterSQLSchemaFail = Any
    ColumnType = Any
    picard_available = False


@dataclass
class PicardArguments:
    """
    Arguments pertaining to Picard.
    """

    use_picard: bool = field(default=True, metadata={"help": "Whether or not to use Picard."})
    launch_picard: bool = field(
        default=True,
        metadata={"help": "Whether or not to launch Picard. If ``False``, an already running Picard is used."},
    )
    picard_host: str = field(default="localhost", metadata={"help": "The host name for Picard."})
    picard_port: int = field(default=9090, metadata={"help": "The port number for Picard."})
    picard_mode: str = field(
        default="parse_with_guards",
        metadata={
            "help": "Picard mode. Choose between ``lex``, ``parse_without_guards``, ``parse_with_guards``, and ``parse_with_guards_and_type_checking."
        },
    )
    picard_schedule: str = field(
        default="incremental",
        metadata={"help": "Picard schedule. Choose between ``incremental`` and ``finalizing``."},
    )
    picard_max_tokens_to_check: int = field(
        default=2,
        metadata={"help": "The maximum number of tokens to check with Picard."},
    )

    def __post_init__(self):
        self.use_picard = picard_available and self.use_picard
        self.launch_picard = self.use_picard and self.launch_picard


class PicardLauncher(subprocess.Popen):
    def __init__(self) -> None:
        try:
            super().__init__(["picard"])
        except FileNotFoundError:
            with subprocess.Popen(
                ["cabal", "install", "--overwrite-policy=always", "--install-method=copy", "exe:picard"]
            ) as picard_build_pid:
                picard_build_pid.wait(timeout=1000)
            super().__init__(["picard"])
        time.sleep(1)

    def __exit__(self, exc_type, value, traceback):
        self.kill()
        super().__exit__(exc_type, value, traceback)

    def __del__(self, _maxsize=sys.maxsize, _warn=warnings.warn):
        self.kill()
        super().__del__(_maxsize, _warn)


def with_picard(
    model_cls, #AutoModelForSeq2SeqLM,
    picard_args: PicardArguments,
    tokenizer: PreTrainedTokenizerFast,
    schemas: Optional[Dict[str, dict]] = None,
):
    schema_cache: Dict[str, dict] = deepcopy(schemas) if schemas is not None else dict()

    def get_picard_client() -> AsyncContextManager[Picard]:
        return get_client(
            Picard,
            host=picard_args.picard_host,
            port=picard_args.picard_port,
            timeout=1,
            protocol=Protocol.BINARY,
        )

    async def _init_picard() -> None:
        async with get_picard_client() as client:
            for db_id, db_info in schema_cache.items():
                await _register_schema(db_id=db_id, db_info=db_info, picard_client=client)
            await _register_tokenizer(picard_client=client)

    async def _register_schema(db_id: str, db_info: dict, picard_client: Picard) -> None:
        sql_schema = get_picard_schema(**db_info)
        try:
            await picard_client.registerSQLSchema(db_id, sql_schema)
        except RegisterSQLSchemaException:
            # db already registered
            logger.debug(f"schema already registered: {db_id}")
            pass

    async def _register_schema_without_client(db_id: str, db_info: dict) -> None:
        async with get_picard_client() as client:
            await _register_schema(db_id=db_id, db_info=db_info, picard_client=client)

    async def _register_tokenizer(picard_client: Picard) -> None:
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        json_str = tokenizer.backend_tokenizer.to_str(pretty=False)
        await picard_client.registerTokenizer(json_str)

    def _add_schema(db_id: str, db_info: dict) -> None:
        if not db_id in schema_cache:
            schema_cache[db_id] = deepcopy(db_info)
            asyncio.run(_register_schema_without_client(db_id=db_id, db_info=db_info), debug=False)
        else:
            assert db_info == schema_cache[db_id], "unexpected schema change"

    @torch.no_grad()
    def _generate(
        self,
        *args,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        # pdb.set_trace()
        # print("Picard successfully load! \n")
        # self.config.decoder_start_token_id = self.config.decoder_start_token_id + kwargs['db_id']
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        logits_processor.append(
            PicardLogitsProcessor(
                eos_token_id=eos_token_id,
                get_client=get_picard_client,
                max_tokens_to_check=picard_args.picard_max_tokens_to_check,
                mode=picard_args.picard_mode,
                schedule=picard_args.picard_schedule,
            )
        )

        return self.old_generate(*args, logits_processor=logits_processor, eos_token_id=eos_token_id, **kwargs)

    class _PicardAutoModelClass(model_cls):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,  **kwargs):
            config = kwargs.pop("config", None)
            kwargs["_from_auto"] = True
            
            if not isinstance(config, PretrainedConfig):
                config, kwargs = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
                )

            if type(config) in cls._model_mapping.keys():
                model_class = _get_model_class(config, cls._model_mapping)
                if "t5" in model_class.config_class.model_type:
                    from seq2seq.models.modeling_t5 import T5ForConditionalGeneration
                    model_class = T5ForConditionalGeneration
                generate = copy_func(_generate)
                generate.__doc__ = model_class.generate.__doc__
                model_class.old_generate = copy_func(model_class.generate)
                model_class.generate = generate
                model_class.add_schema = staticmethod(copy_func(_add_schema))
                
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
            raise ValueError(
                f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
                f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
            )

    asyncio.run(_init_picard(), debug=False)

    return _PicardAutoModelClass


class PicardLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        eos_token_id: int,
        get_client: Callable[[], AsyncContextManager[Picard]],
        filter_value: float = -float("Inf"),
        max_tokens_to_check: int = 1,
        mode: str = "parse_with_guards",
        schedule: str = "incremental",
    ):
        self.eos_token_id = eos_token_id
        self.get_client = get_client
        self.filter_value = filter_value
        self.max_tokens_to_check = max_tokens_to_check
        self.mode = mode
        self.schedule = schedule

    async def _feed(self, client: Picard, input_ids: List[int], token: int) -> bool:
        if self.mode == "lex":
            mode = Mode.LEXING
        elif self.mode == "parse_without_guards":
            mode = Mode.PARSING_WITHOUT_GUARDS
        elif self.mode == "parse" or self.mode == "parse_with_guards":
            mode = Mode.PARSING_WITH_GUARDS
        elif self.mode == "parse_with_guards_and_type_checking":
            mode = Mode.PARSING_WITH_GUARDS_AND_TYPE_CHECKING
        else:
            raise ValueError("unexpected picard mode")

        try:
            res = await client.feed(input_ids, token, mode)
        except FeedException as e:
            logger.error(f"unexpected feed error: {e}, input ids were: {input_ids}, token was: {token}")
            raise e
        except TransportError as e:
            logger.error(f"unexpected transport error: {e}, input ids were: {input_ids}, token was: {token}")
            raise e

        if isinstance(res.feedResult.value, FeedTimeoutFailure):
            logger.warning(f"timeout failure: {input_ids + [token]}")
            return False
        elif isinstance(res.feedResult.value, FeedParseFailure):
            logger.debug(f"parsing failure: {input_ids + [token]}")
            return False
        elif isinstance(res.feedResult.value, FeedPartialSuccess):
            logger.debug(f"parsing partial: {input_ids + [token]}")
            return True
        elif isinstance(res.feedResult.value, FeedCompleteSuccess):
            logger.info(f"parsing success: {input_ids + [token]}")
            return True
        else:
            # unexpected parsing result
            raise ValueError("unexpected picard parsing result")

    async def _check_token(self, client: Picard, input_ids: List[int], token: int) -> bool:
        if self.schedule == "incremental":
            # check at every step
            return await self._feed(client=client, input_ids=input_ids, token=token)
        elif self.schedule == "finalizing":
            # only check when decoded string is finalized
            if token == self.eos_token_id:
                return await self._feed(client=client, input_ids=input_ids, token=token)
            else:
                return True
        else:
            raise ValueError("unexpected picard schedule")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_delay(600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _mask(
        self,
        client: Picard,
        indices_to_remove: torch.Tensor,
        batch_idx: int,
        input_ids_batch: torch.Tensor,
        top_token: torch.Tensor,
    ) -> None:
        res = await self._check_token(client=client, input_ids=input_ids_batch.tolist(), token=top_token.item())
        if not res:
            indices_to_remove[batch_idx, top_token] = True

    async def _mask_top_k(
        self,
        indices_to_remove: torch.Tensor,
        input_ids: torch.Tensor,
        top_tokens: torch.Tensor,
    ) -> None:
        async with self.get_client() as client:
            futures = [
                self._mask(
                    client=client,
                    indices_to_remove=indices_to_remove,
                    batch_idx=batch_idx,
                    input_ids_batch=input_ids_batch,
                    top_token=top_token,
                )
                for batch_idx, (input_ids_batch, top_token_batch) in enumerate(zip(input_ids, top_tokens))
                for top_token in top_token_batch
            ]
            for f in asyncio.as_completed(futures):
                await f

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_delay(600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _batch_mask_top_k(
        self,
        indices_to_remove: torch.Tensor,
        input_ids: torch.Tensor,
        top_tokens: torch.Tensor,
    ) -> None:
        if self.mode == "lex":
            mode = Mode.LEXING
        elif self.mode == "parse_without_guards":
            mode = Mode.PARSING_WITHOUT_GUARDS
        elif self.mode == "parse" or self.mode == "parse_with_guards":
            mode = Mode.PARSING_WITH_GUARDS
        elif self.mode == "parse_with_guards_and_type_checking":
            mode = Mode.PARSING_WITH_GUARDS_AND_TYPE_CHECKING
        else:
            raise ValueError("unexpected picard mode")

        async with self.get_client() as client:
            try:
                res = await client.batchFeed(input_ids.tolist(), top_tokens.tolist(), mode)
            except FeedException as e:
                logger.error(
                    f"unexpected feed error: {e}, input ids were: {input_ids.tolist()}, top tokens were: {top_tokens.tolist()}"
                )
                raise e
            except TransportError as e:
                logger.error(
                    f"unexpected transport error: {e}, input ids were: {input_ids.tolist()}, top tokens were: {top_tokens.tolist()}"
                )
                raise e

        for r in res:
            if isinstance(r.feedResult.value, FeedTimeoutFailure):
                logger.warning(f"timeout failure: {input_ids[r.batchId].tolist() + [r.topToken]}")
                indices_to_remove[r.batchId, r.topToken] = True
            elif isinstance(r.feedResult.value, FeedParseFailure):
                logger.debug(f"parsing failure: {input_ids[r.batchId].tolist() + [r.topToken]}")
                indices_to_remove[r.batchId, r.topToken] = True
            elif isinstance(r.feedResult.value, FeedPartialSuccess):
                logger.debug(f"parsing partial: {input_ids[r.batchId].tolist() + [r.topToken]}")
            elif isinstance(r.feedResult.value, FeedCompleteSuccess):
                logger.info(f"parsing success: {input_ids[r.batchId].tolist() + [r.topToken]}")
            else:
                # unexpected parsing result
                raise ValueError("unexpected picard parsing result")

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(1, self.max_tokens_to_check), scores.size(-1))  # Safety check
        top_scores, top_tokens = torch.topk(scores, top_k)
        # Remove all tokens with a probability less than the last token of the top-k
        lowest_top_k_scores = top_scores[..., -1, None]
        del top_scores
        indices_to_remove = scores < lowest_top_k_scores
        del lowest_top_k_scores
        # Do not mask the EOS token because otherwise production can continue indefinitely if all other tokens are masked
        indices_to_remove[:, self.eos_token_id] = False
        # Mask top-k tokens rejected by picard
        asyncio.run(
            self._batch_mask_top_k(
                indices_to_remove=indices_to_remove,
                input_ids=input_ids,
                top_tokens=top_tokens,
            )
            if self.schedule == "incremental"
            else self._mask_top_k(
                indices_to_remove=indices_to_remove,
                input_ids=input_ids,
                top_tokens=top_tokens,
            ),
            debug=False,
        )
        del top_tokens
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        del indices_to_remove
        return scores


def _get_picard_column_type(column_type: str) -> ColumnType:
    if column_type == "text":
        return ColumnType.TEXT
    elif column_type == "number":
        return ColumnType.NUMBER
    elif column_type == "time":
        return ColumnType.TIME
    elif column_type == "boolean":
        return ColumnType.BOOLEAN
    elif column_type == "others":
        return ColumnType.OTHERS
    else:
        raise ValueError(f"unexpected column type {column_type}")


def get_picard_schema(
    db_table_names: List[str],
    db_column_names: Dict[str, Union[List[str], List[int]]],
    db_column_types: List[str],
    db_primary_keys: Dict[str, List[int]],
    db_foreign_keys: Dict[str, List[int]],
) -> SQLSchema:
    star_id = next((c_id for c_id, c_name in enumerate(db_column_names["column_name"]) if c_name == "*"))
    column_names = dict(
        (str(c_id), c_name) for c_id, c_name in enumerate(db_column_names["column_name"]) if c_id != star_id
    )
    column_types = dict(
        (str(c_id), _get_picard_column_type(c_type)) for c_id, c_type in enumerate(db_column_types) if c_id != star_id
    )
    table_names = dict((str(t_id), t_name) for t_id, t_name in enumerate(db_table_names))
    column_to_table = dict(
        (str(c_id), str(t_id))
        for c_id, (t_id, _c_name) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"]))
        if c_id != star_id
    )
    table_to_columns = collections.defaultdict(list)
    for c_id, (t_id, _c_name) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"])):
        if c_id == star_id:
            continue
        table_to_columns[str(t_id)].append(str(c_id))
    foreign_keys = dict(
        (str(c_id), str(other_c_id))
        for c_id, other_c_id in zip(db_foreign_keys["column_id"], db_foreign_keys["other_column_id"])
        if c_id != star_id and other_c_id != star_id
    )
    primary_keys = [str(c_id) for c_id in db_primary_keys["column_id"] if c_id != star_id]
    return SQLSchema(
        columnNames=column_names,
        columnTypes=column_types,
        tableNames=table_names,
        columnToTable=column_to_table,
        tableToColumns=table_to_columns,
        foreignKeys=foreign_keys,
        primaryKeys=primary_keys,
    )
