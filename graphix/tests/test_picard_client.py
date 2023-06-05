import asyncio
import sys
from typing import AsyncContextManager, Dict, List, Callable, Any
import logging
from random import sample

from thrift.py3.client import get_client
from thrift.py3.common import Protocol
from picard.clients import Picard
from picard.types import (
    FeedTimeoutFailure,
    SQLSchema,
    RegisterSQLSchemaException,
    FeedException,
    Mode,
    ColumnType,
)
from seq2seq.utils.picard_model_wrapper import PicardLauncher

try:
    from transformers.models.auto import AutoTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    from alive_progress import alive_bar

    has_tokenizer = True
except:
    PreTrainedTokenizerFast = Any
    has_tokenizer = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)


async def test_register_sql_schema(
    sql_schemas: Dict[str, SQLSchema],
    get_picard_client: Callable[[], AsyncContextManager[Picard]],
) -> None:
    async with get_picard_client() as client:
        for db_id, sql_schema in sql_schemas.items():
            logger.info(f"registering {db_id}...")
            try:
                await client.registerSQLSchema(db_id, sql_schema)
            except RegisterSQLSchemaException:
                # db already registered
                logger.warning(f"schema already registered: {db_id}")
                pass
            logger.info(f"{db_id} registered")


async def test_register_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    get_picard_client: Callable[[], AsyncContextManager[Picard]],
) -> None:
    async with get_picard_client() as client:
        logger.info(f"registering {tokenizer}...")
        json_str = tokenizer.backend_tokenizer.to_str(pretty=False)
        await client.registerTokenizer(json_str)
        logger.info(f"{tokenizer} registered")


async def _feed(client: Picard, input_ids: List[int], token: int, with_guards: bool = True) -> None:
    try:
        await client.feed(
            input_ids,
            token,
            Mode.PARSING_WITH_GUARDS if with_guards else Mode.PARSING_WITHOUT_GUARDS,
        )
    except FeedException:
        pass


async def test_feedParser(
    input_ids: List[List[int]],
    tokens: List[int],
    get_picard_client: Callable[[], AsyncContextManager[Picard]],
    with_guards: bool = True,
) -> None:

    async with get_picard_client() as client:
        logger.info(
            f"synchronously feeding {len(tokens)} token continuations for {len(input_ids)} input sequence(s)..."
        )
        with alive_bar(len(tokens) * len(input_ids)) as bar:
            for input_ids_ in input_ids:
                for token in tokens:
                    await _feed(client, input_ids_, token, with_guards)
                    bar()
        logger.info(f"synchronously feeding complete")

    async with get_picard_client() as client:
        logger.info(
            f"asynchronously feeding {len(tokens)} token continuations for {len(input_ids)} input sequence(s)..."
        )
        futures = [_feed(client, input_ids_, token, with_guards) for token in tokens for input_ids_ in input_ids]
        with alive_bar(len(tokens) * len(input_ids)) as bar:
            for f in asyncio.as_completed(futures):
                await f
                bar()
        logger.info(f"asynchronously feeding complete")


def in_vitro_picard(
    input_ids: List[List[int]],
    tokens: List[int],
    max_tokens_to_check: int,
    get_picard_client: Callable[[], AsyncContextManager[Picard]],
    with_guards: bool = True,
    batched: bool = True,
):
    async def _simulate_step(input_ids: List[List[int]], top_tokens: List[List[int]]):
        async with get_picard_client() as client:
            futures = [
                # asyncio.sleep(0.01)
                _feed(
                    client=client,
                    input_ids=input_ids_batch,
                    token=top_token,
                    with_guards=with_guards,
                )
                for _batch_idx, (input_ids_batch, top_token_batch) in enumerate(zip(input_ids, top_tokens))
                for top_token in top_token_batch
            ]
            for f in asyncio.as_completed(futures):
                await f

    async def _simulate_batched_step(input_ids: List[List[int]], top_tokens: List[List[int]]):
        async with get_picard_client() as client:
            try:
                results = await client.batchFeed(
                    input_ids,
                    top_tokens,
                    Mode.PARSING_WITH_GUARDS if with_guards else Mode.PARSING_WITHOUT_GUARDS,
                )
                # for r in results:
                #     if isinstance(r.feedResult.value, FeedTimeoutFailure):
                #         raise RuntimeError
            except FeedException:
                pass

    beam_size = len(input_ids)
    assert beam_size > 0
    seq_len = len(input_ids[0])
    assert all(len(seq) == seq_len for seq in input_ids)
    assert len(tokens) >= max_tokens_to_check - 1
    for n in range(seq_len):
        truncated_input_ids = [seq[:n] for seq in input_ids]
        simulated_top_tokens = [
            [seq[n]] + sample(tokens, k=len(tokens))[: max_tokens_to_check - 1] for seq in input_ids
        ]
        asyncio.run(
            _simulate_batched_step(input_ids=truncated_input_ids, top_tokens=simulated_top_tokens)
            if batched
            else _simulate_step(input_ids=truncated_input_ids, top_tokens=simulated_top_tokens)
        )


sql_schemas = {
    "test": SQLSchema(
        columnNames={"0": "column"},
        columnTypes={"0": ColumnType.NUMBER},
        tableNames={"0": "table"},
        columnToTable={"0": "0"},
        tableToColumns={"0": ["0"]},
        foreignKeys=dict(),
        primaryKeys=list(),
    ),
    "car_1": SQLSchema(
        columnNames={
            "1": "ContId",
            "10": "ModelId",
            "11": "Maker",
            "12": "Model",
            "13": "MakeId",
            "14": "Model",
            "15": "Make",
            "16": "Id",
            "17": "MPG",
            "18": "Cylinders",
            "19": "Edispl",
            "2": "Continent",
            "20": "Horsepower",
            "21": "Weight",
            "22": "Accelerate",
            "23": "Year",
            "3": "CountryId",
            "4": "CountryName",
            "5": "Continent",
            "6": "Id",
            "7": "Maker",
            "8": "FullName",
            "9": "Country",
        },
        columnTypes={
            "1": ColumnType.NUMBER,
            "10": ColumnType.NUMBER,
            "11": ColumnType.NUMBER,
            "12": ColumnType.TEXT,
            "13": ColumnType.NUMBER,
            "14": ColumnType.TEXT,
            "15": ColumnType.TEXT,
            "16": ColumnType.NUMBER,
            "17": ColumnType.TEXT,
            "18": ColumnType.NUMBER,
            "19": ColumnType.NUMBER,
            "2": ColumnType.TEXT,
            "20": ColumnType.TEXT,
            "21": ColumnType.NUMBER,
            "22": ColumnType.NUMBER,
            "23": ColumnType.NUMBER,
            "3": ColumnType.NUMBER,
            "4": ColumnType.TEXT,
            "5": ColumnType.NUMBER,
            "6": ColumnType.NUMBER,
            "7": ColumnType.TEXT,
            "8": ColumnType.TEXT,
            "9": ColumnType.TEXT,
        },
        tableNames={
            "0": "continents",
            "1": "countries",
            "2": "car_makers",
            "3": "model_list",
            "4": "car_names",
            "5": "cars_data",
        },
        columnToTable={
            "1": "0",
            "10": "3",
            "11": "3",
            "12": "3",
            "13": "4",
            "14": "4",
            "15": "4",
            "16": "5",
            "17": "5",
            "18": "5",
            "19": "5",
            "2": "0",
            "20": "5",
            "21": "5",
            "22": "5",
            "23": "5",
            "3": "1",
            "4": "1",
            "5": "1",
            "6": "2",
            "7": "2",
            "8": "2",
            "9": "2",
        },
        tableToColumns={
            "0": ["1", "2"],
            "1": ["3", "4", "5"],
            "2": ["6", "7", "8", "9"],
            "3": ["10", "11", "12"],
            "4": ["13", "14", "15"],
            "5": ["16", "17", "18", "19", "20", "21", "22", "23"],
        },
        foreignKeys={"11": "6", "14": "12", "16": "13", "5": "1", "9": "3"},
        primaryKeys=["1", "3", "6", "10", "13", "16"],
    ),
}


def main() -> None:
    def get_picard_client() -> AsyncContextManager[Picard]:
        return get_client(Picard, host="127.0.0.1", port=9090, timeout=1, protocol=Protocol.BINARY)

    asyncio.run(test_register_sql_schema(sql_schemas=sql_schemas, get_picard_client=get_picard_client))

    if has_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
        asyncio.run(test_register_tokenizer(tokenizer=tokenizer, get_picard_client=get_picard_client))

        tokens = list(tokenizer.get_vocab().values())  # they'll come in a random order

        input_ids = tokenizer(
            [
                "<pad>test | select table.column from table</s>",
                "<pad>car_1 | select avg(edispl) from cars_data where id in (select id from cars_data where id in (select id from cars_data where id in (select id from cars_data where id in (select id from cars_data where id in (select id from cars_data where id in (select id from cars_data where",
                '<pad>car_1 | select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t2.id = t3.maker where t3.model = "Fiat"</s>',
            ],
            add_special_tokens=False,
        )["input_ids"]

        # fmt: off
        assert input_ids == [
            [0, 794, 1820, 1738, 953, 5, 3297, 440, 29, 45, 953, 1],
            [0, 443, 834, 536, 1820, 1738, 3, 9, 208, 122, 599, 15, 10475, 40, 61, 45, 2948, 834, 6757, 213, 3, 23, 26, 16, 41, 7, 15, 3437, 3, 23, 26, 45, 2948, 834, 6757, 213, 3, 23, 26, 16, 41, 7, 15, 3437, 3, 23, 26, 45, 2948, 834, 6757, 213, 3, 23, 26, 16, 41, 7, 15, 3437, 3, 23, 26, 45, 2948, 834, 6757, 213, 3, 23, 26, 16, 41, 7, 15, 3437, 3, 23, 26, 45, 2948, 834, 6757, 213, 3, 23, 26, 16, 41, 7, 15, 3437, 3, 23, 26, 45, 2948, 834, 6757, 213, 3, 23, 26, 16, 41, 7, 15, 3437, 3, 23, 26, 45, 2948, 834, 6757, 213],
            [0, 443, 834, 536, 1820, 1738, 3, 17, 5411, 17529, 23, 26, 6, 3, 17, 5411, 17529, 4350, 45, 1440, 38, 3, 17, 536, 1715, 443, 834, 8910, 38, 3, 17, 357, 30, 3, 17, 5411, 17529, 23, 26, 3274, 3, 17, 4416, 17529, 563, 57, 3, 17, 5411, 17529, 23, 26, 578, 3476, 599, 1935, 61, 2490, 220, 7021, 1738, 3, 17, 5411, 17529, 23, 26, 6, 3, 17, 5411, 17529, 4350, 45, 1440, 38, 3, 17, 536, 1715, 443, 834, 8910, 38, 3, 17, 357, 30, 3, 17, 5411, 17529, 23, 26, 3274, 3, 17, 4416, 17529, 1715, 825, 834, 3350, 38, 3, 17, 519, 30, 3, 17, 4416, 23, 26, 3274, 3, 17, 5787, 8337, 213, 3, 17, 5787, 21770, 3274, 96, 3183, 144, 121, 1],
        ]
        # fmt: on

        # In-vitro Picard
        trials = 100
        max_tokens_to_check = 2
        simulated_beam_size = 4
        with alive_bar(trials * max_tokens_to_check * simulated_beam_size) as bar:
            for _ in range(trials):
                in_vitro_picard(
                    input_ids=[input_ids[2]] * simulated_beam_size,
                    tokens=tokens[: max_tokens_to_check * 10],
                    max_tokens_to_check=max_tokens_to_check,
                    get_picard_client=get_picard_client,
                    with_guards=True,
                )
                bar(max_tokens_to_check * simulated_beam_size)

        # Populate parser cache
        # asyncio.run(
        #     test_feedParser(
        #         input_ids=[
        #             input_ids[2][:103],
        #         ],
        #         tokens=[input_ids[2][103]],
        #         get_picard_client=get_picard_client,
        #         with_guards=True,
        #     )
        # )

        # Test performance with input id sequence of intermediate length
        # asyncio.run(
        #     test_feedParser(
        #         input_ids=[
        #             input_ids[2][:104],
        #         ],
        #         tokens=tokens[:1000],
        #         get_picard_client=get_picard_client,
        #         with_guards=True,
        #     )
        # )

        # Test scaling with short input id sequence and many tokens
        # asyncio.run(
        #     test_feedParser(
        #         input_ids=[
        #             input_ids[0][:-1],
        #         ],  # do not add </s> so that the sequence is not terminated
        #         tokens=tokens,
        #         get_picard_client=get_picard_client,
        #         with_guards=True,
        #     )
        # )

        # Test timeout with adversarial input id sequence
        # asyncio.run(
        #     test_feedParser(
        #         input_ids=[input_ids[1] + [2029]],
        #         tokens=tokens[:20],
        #         get_picard_client=get_picard_client,
        #         with_guards=True,
        #     )
        # )


if __name__ == "__main__":
    with PicardLauncher():
        main()
