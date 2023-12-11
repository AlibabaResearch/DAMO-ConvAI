#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import defaultdict, Counter
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span


class CASIE(TaskFormat):
    def __init__(self, sentence_dict, language="en"):
        super().__init__(language=language)
        self.sent_id = sentence_dict["sent_id"]
        self.tokens = sentence_dict["tokens"]
        self.entities = sentence_dict["entity_mentions"]
        self.events = sentence_dict["event_mentions"]

    def generate_instance(self):
        entities = {}
        events = {}

        for entity in self.entities:
            indexes = entity["indexes"]
            tokens = [self.tokens[id] for id in indexes]
            entities[entity["id"]] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id,
                ),
                label=Label(entity["type"]),
                text_id=self.sent_id,
                record_id=entity["id"],
            )

        for event in self.events:
            indexes = event["trigger"]["indexes"]
            tokens = [self.tokens[id] for id in indexes]
            events[event["id"]] = Event(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id,
                ),
                label=Label(event["type"]),
                args=[
                    (Label(x["role"]), entities[x["id"]])
                    for x in event["arguments"]
                ],
                text_id=self.sent_id,
                record_id=event["id"],
            )

        return Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            events=events.values(),
            text_id=self.sent_id,
        )

    @staticmethod
    def load_from_file(filename, language="en") -> List[Sentence]:
        sentence_list = []
        cross_sentence_cnt = 0
        counter = Counter()

        with open(filename) as fin:
            for line in fin:
                doc = json.loads(line.strip())

                entity_mentions = defaultdict(list)
                event_mentions = defaultdict(list)

                for event in doc["event"]:
                    for mention in event["mentions"]:
                        nugget = mention["nugget"]
                        sent_id = nugget["tokens"][0][0]

                        event_mention = {
                            "id": mention["id"],
                            "type": mention["subtype"],
                            "trigger": {"indexes": [x[1] for x in nugget["tokens"]],},
                            "arguments": [],
                        }
                        counter.update(['event mention'])

                        for argument in mention["arguments"]:
                            arg_sent_id = argument["tokens"][0][0]
                            entity_mention = {
                                "id": argument["id"],
                                "indexes": [x[1] for x in argument["tokens"]],
                                "type": argument["filler_type"],
                            }
                            entity_mentions[arg_sent_id].append(entity_mention)
                            counter.update(['entity'])
                            if arg_sent_id == sent_id:
                                event_mention["arguments"].append(
                                    {
                                        "id": argument["id"],
                                        "trigger": {
                                            "indexes": [x[1] for x in nugget["tokens"]],
                                        },
                                        "role": argument["role"],
                                    }
                                )
                                counter.update(['argument'])
                            else:
                                counter.update(['cross_sentence_cnt'])

                        event_mentions[sent_id].append(event_mention)

                for sent_id, sentence in enumerate(doc["sentences"]):
                    tokens = [token["word"] for token in sentence["tokens"]]

                    sentence_dict = {
                        "sent_id": sent_id,
                        "tokens": tokens,
                        "entity_mentions": entity_mentions[sent_id],
                        "event_mentions": event_mentions[sent_id],
                    }
                    instance = CASIE(
                        sentence_dict, language=language
                    ).generate_instance()

                    sentence_list.append(instance)
                    counter.update(['sentence'])

        print(filename, counter)
        return sentence_list


"""
{
    "id": "10231.txt",
    "sentences": [
        {
            "tokens": [
                {
                    "characterOffsetBegin": 0,
                    "characterOffsetEnd": 2,
                    "word": "An",
                    "originalText": "An",
                },
                {
                    "characterOffsetBegin": 3,
                    "characterOffsetEnd": 8,
                    "word": "email",
                    "originalText": "email",
                },
                {
                    "characterOffsetBegin": 9,
                    "characterOffsetEnd": 13,
                    "word": "scam",
                    "originalText": "scam",
                },
                {
                    "characterOffsetBegin": 14,
                    "characterOffsetEnd": 21,
                    "word": "passing",
                    "originalText": "passing",
                },
                {
                    "characterOffsetBegin": 22,
                    "characterOffsetEnd": 24,
                    "word": "as",
                    "originalText": "as",
                },
                {
                    "characterOffsetBegin": 25,
                    "characterOffsetEnd": 26,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 27,
                    "characterOffsetEnd": 34,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 35,
                    "characterOffsetEnd": 47,
                    "word": "notification",
                    "originalText": "notification",
                },
                {
                    "characterOffsetBegin": 48,
                    "characterOffsetEnd": 51,
                    "word": "has",
                    "originalText": "has",
                },
                {
                    "characterOffsetBegin": 52,
                    "characterOffsetEnd": 56,
                    "word": "been",
                    "originalText": "been",
                },
                {
                    "characterOffsetBegin": 57,
                    "characterOffsetEnd": 66,
                    "word": "targeting",
                    "originalText": "targeting",
                },
                {
                    "characterOffsetBegin": 67,
                    "characterOffsetEnd": 78,
                    "word": "subscribers",
                    "originalText": "subscribers",
                },
                {
                    "characterOffsetBegin": 79,
                    "characterOffsetEnd": 81,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 82,
                    "characterOffsetEnd": 85,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 86,
                    "characterOffsetEnd": 95,
                    "word": "streaming",
                    "originalText": "streaming",
                },
                {
                    "characterOffsetBegin": 96,
                    "characterOffsetEnd": 103,
                    "word": "service",
                    "originalText": "service",
                },
                {
                    "characterOffsetBegin": 103,
                    "characterOffsetEnd": 104,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [0, 104],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 105,
                    "characterOffsetEnd": 108,
                    "word": "The",
                    "originalText": "The",
                },
                {
                    "characterOffsetBegin": 109,
                    "characterOffsetEnd": 110,
                    "word": "“",
                    "originalText": "“",
                },
                {
                    "characterOffsetBegin": 110,
                    "characterOffsetEnd": 120,
                    "word": "suspension",
                    "originalText": "suspension",
                },
                {
                    "characterOffsetBegin": 121,
                    "characterOffsetEnd": 133,
                    "word": "notification",
                    "originalText": "notification",
                },
                {
                    "characterOffsetBegin": 133,
                    "characterOffsetEnd": 134,
                    "word": "”",
                    "originalText": "”",
                },
                {
                    "characterOffsetBegin": 135,
                    "characterOffsetEnd": 140,
                    "word": "looks",
                    "originalText": "looks",
                },
                {
                    "characterOffsetBegin": 141,
                    "characterOffsetEnd": 148,
                    "word": "similar",
                    "originalText": "similar",
                },
                {
                    "characterOffsetBegin": 149,
                    "characterOffsetEnd": 151,
                    "word": "in",
                    "originalText": "in",
                },
                {
                    "characterOffsetBegin": 152,
                    "characterOffsetEnd": 158,
                    "word": "design",
                    "originalText": "design",
                },
                {
                    "characterOffsetBegin": 159,
                    "characterOffsetEnd": 162,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 163,
                    "characterOffsetEnd": 169,
                    "word": "format",
                    "originalText": "format",
                },
                {
                    "characterOffsetBegin": 170,
                    "characterOffsetEnd": 172,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 173,
                    "characterOffsetEnd": 178,
                    "word": "other",
                    "originalText": "other",
                },
                {
                    "characterOffsetBegin": 179,
                    "characterOffsetEnd": 186,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 187,
                    "characterOffsetEnd": 193,
                    "word": "emails",
                    "originalText": "emails",
                },
                {
                    "characterOffsetBegin": 193,
                    "characterOffsetEnd": 194,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [105, 194],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 195,
                    "characterOffsetEnd": 197,
                    "word": "It",
                    "originalText": "It",
                },
                {
                    "characterOffsetBegin": 198,
                    "characterOffsetEnd": 206,
                    "word": "notifies",
                    "originalText": "notifies",
                },
                {
                    "characterOffsetBegin": 207,
                    "characterOffsetEnd": 210,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 211,
                    "characterOffsetEnd": 216,
                    "word": "urges",
                    "originalText": "urges",
                },
                {
                    "characterOffsetBegin": 217,
                    "characterOffsetEnd": 222,
                    "word": "users",
                    "originalText": "users",
                },
                {
                    "characterOffsetBegin": 223,
                    "characterOffsetEnd": 225,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 226,
                    "characterOffsetEnd": 232,
                    "word": "update",
                    "originalText": "update",
                },
                {
                    "characterOffsetBegin": 233,
                    "characterOffsetEnd": 238,
                    "word": "their",
                    "originalText": "their",
                },
                {
                    "characterOffsetBegin": 239,
                    "characterOffsetEnd": 250,
                    "word": "information",
                    "originalText": "information",
                },
                {
                    "characterOffsetBegin": 251,
                    "characterOffsetEnd": 253,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 254,
                    "characterOffsetEnd": 259,
                    "word": "avoid",
                    "originalText": "avoid",
                },
                {
                    "characterOffsetBegin": 260,
                    "characterOffsetEnd": 263,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 264,
                    "characterOffsetEnd": 274,
                    "word": "suspension",
                    "originalText": "suspension",
                },
                {
                    "characterOffsetBegin": 275,
                    "characterOffsetEnd": 277,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 278,
                    "characterOffsetEnd": 283,
                    "word": "their",
                    "originalText": "their",
                },
                {
                    "characterOffsetBegin": 284,
                    "characterOffsetEnd": 291,
                    "word": "account",
                    "originalText": "account",
                },
                {
                    "characterOffsetBegin": 291,
                    "characterOffsetEnd": 292,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [195, 292],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 293,
                    "characterOffsetEnd": 296,
                    "word": "The",
                    "originalText": "The",
                },
                {
                    "characterOffsetBegin": 297,
                    "characterOffsetEnd": 301,
                    "word": "goal",
                    "originalText": "goal",
                },
                {
                    "characterOffsetBegin": 302,
                    "characterOffsetEnd": 304,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 305,
                    "characterOffsetEnd": 308,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 309,
                    "characterOffsetEnd": 313,
                    "word": "scam",
                    "originalText": "scam",
                },
                {
                    "characterOffsetBegin": 314,
                    "characterOffsetEnd": 316,
                    "word": "is",
                    "originalText": "is",
                },
                {
                    "characterOffsetBegin": 317,
                    "characterOffsetEnd": 319,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 320,
                    "characterOffsetEnd": 325,
                    "word": "steal",
                    "originalText": "steal",
                },
                {
                    "characterOffsetBegin": 326,
                    "characterOffsetEnd": 334,
                    "word": "personal",
                    "originalText": "personal",
                },
                {
                    "characterOffsetBegin": 335,
                    "characterOffsetEnd": 338,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 339,
                    "characterOffsetEnd": 345,
                    "word": "credit",
                    "originalText": "credit",
                },
                {
                    "characterOffsetBegin": 346,
                    "characterOffsetEnd": 350,
                    "word": "card",
                    "originalText": "card",
                },
                {
                    "characterOffsetBegin": 351,
                    "characterOffsetEnd": 362,
                    "word": "information",
                    "originalText": "information",
                },
                {
                    "characterOffsetBegin": 362,
                    "characterOffsetEnd": 363,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 364,
                    "characterOffsetEnd": 373,
                    "word": "according",
                    "originalText": "according",
                },
                {
                    "characterOffsetBegin": 374,
                    "characterOffsetEnd": 376,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 377,
                    "characterOffsetEnd": 378,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 379,
                    "characterOffsetEnd": 385,
                    "word": "report",
                    "originalText": "report",
                },
                {
                    "characterOffsetBegin": 386,
                    "characterOffsetEnd": 390,
                    "word": "from",
                    "originalText": "from",
                },
                {
                    "characterOffsetBegin": 391,
                    "characterOffsetEnd": 400,
                    "word": "Mailguard",
                    "originalText": "Mailguard",
                },
                {
                    "characterOffsetBegin": 400,
                    "characterOffsetEnd": 401,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [293, 401],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 402,
                    "characterOffsetEnd": 405,
                    "word": "The",
                    "originalText": "The",
                },
                {
                    "characterOffsetBegin": 406,
                    "characterOffsetEnd": 411,
                    "word": "email",
                    "originalText": "email",
                },
                {
                    "characterOffsetBegin": 412,
                    "characterOffsetEnd": 420,
                    "word": "contains",
                    "originalText": "contains",
                },
                {
                    "characterOffsetBegin": 421,
                    "characterOffsetEnd": 422,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 423,
                    "characterOffsetEnd": 427,
                    "word": "link",
                    "originalText": "link",
                },
                {
                    "characterOffsetBegin": 428,
                    "characterOffsetEnd": 430,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 431,
                    "characterOffsetEnd": 432,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 433,
                    "characterOffsetEnd": 437,
                    "word": "fake",
                    "originalText": "fake",
                },
                {
                    "characterOffsetBegin": 438,
                    "characterOffsetEnd": 445,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 446,
                    "characterOffsetEnd": 453,
                    "word": "website",
                    "originalText": "website",
                },
                {
                    "characterOffsetBegin": 454,
                    "characterOffsetEnd": 459,
                    "word": "where",
                    "originalText": "where",
                },
                {
                    "characterOffsetBegin": 460,
                    "characterOffsetEnd": 465,
                    "word": "users",
                    "originalText": "users",
                },
                {
                    "characterOffsetBegin": 466,
                    "characterOffsetEnd": 469,
                    "word": "are",
                    "originalText": "are",
                },
                {
                    "characterOffsetBegin": 470,
                    "characterOffsetEnd": 478,
                    "word": "required",
                    "originalText": "required",
                },
                {
                    "characterOffsetBegin": 479,
                    "characterOffsetEnd": 481,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 482,
                    "characterOffsetEnd": 487,
                    "word": "enter",
                    "originalText": "enter",
                },
                {
                    "characterOffsetBegin": 488,
                    "characterOffsetEnd": 491,
                    "word": "log",
                    "originalText": "log",
                },
                {
                    "characterOffsetBegin": 491,
                    "characterOffsetEnd": 492,
                    "word": "-",
                    "originalText": "-",
                },
                {
                    "characterOffsetBegin": 492,
                    "characterOffsetEnd": 494,
                    "word": "in",
                    "originalText": "in",
                },
                {
                    "characterOffsetBegin": 495,
                    "characterOffsetEnd": 506,
                    "word": "information",
                    "originalText": "information",
                },
                {
                    "characterOffsetBegin": 507,
                    "characterOffsetEnd": 510,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 511,
                    "characterOffsetEnd": 512,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 513,
                    "characterOffsetEnd": 519,
                    "word": "credit",
                    "originalText": "credit",
                },
                {
                    "characterOffsetBegin": 520,
                    "characterOffsetEnd": 524,
                    "word": "card",
                    "originalText": "card",
                },
                {
                    "characterOffsetBegin": 525,
                    "characterOffsetEnd": 531,
                    "word": "number",
                    "originalText": "number",
                },
                {
                    "characterOffsetBegin": 531,
                    "characterOffsetEnd": 532,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [402, 532],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 533,
                    "characterOffsetEnd": 536,
                    "word": "The",
                    "originalText": "The",
                },
                {
                    "characterOffsetBegin": 537,
                    "characterOffsetEnd": 541,
                    "word": "faux",
                    "originalText": "faux",
                },
                {
                    "characterOffsetBegin": 542,
                    "characterOffsetEnd": 549,
                    "word": "website",
                    "originalText": "website",
                },
                {
                    "characterOffsetBegin": 550,
                    "characterOffsetEnd": 553,
                    "word": "has",
                    "originalText": "has",
                },
                {
                    "characterOffsetBegin": 554,
                    "characterOffsetEnd": 557,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 558,
                    "characterOffsetEnd": 565,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 566,
                    "characterOffsetEnd": 570,
                    "word": "logo",
                    "originalText": "logo",
                },
                {
                    "characterOffsetBegin": 571,
                    "characterOffsetEnd": 573,
                    "word": "on",
                    "originalText": "on",
                },
                {
                    "characterOffsetBegin": 574,
                    "characterOffsetEnd": 581,
                    "word": "display",
                    "originalText": "display",
                },
                {
                    "characterOffsetBegin": 582,
                    "characterOffsetEnd": 586,
                    "word": "plus",
                    "originalText": "plus",
                },
                {
                    "characterOffsetBegin": 587,
                    "characterOffsetEnd": 590,
                    "word": "The",
                    "originalText": "The",
                },
                {
                    "characterOffsetBegin": 591,
                    "characterOffsetEnd": 596,
                    "word": "Crown",
                    "originalText": "Crown",
                },
                {
                    "characterOffsetBegin": 597,
                    "characterOffsetEnd": 600,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 601,
                    "characterOffsetEnd": 606,
                    "word": "House",
                    "originalText": "House",
                },
                {
                    "characterOffsetBegin": 607,
                    "characterOffsetEnd": 609,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 610,
                    "characterOffsetEnd": 615,
                    "word": "Cards",
                    "originalText": "Cards",
                },
                {
                    "characterOffsetBegin": 616,
                    "characterOffsetEnd": 623,
                    "word": "banners",
                    "originalText": "banners",
                },
                {
                    "characterOffsetBegin": 624,
                    "characterOffsetEnd": 626,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 627,
                    "characterOffsetEnd": 634,
                    "word": "further",
                    "originalText": "further",
                },
                {
                    "characterOffsetBegin": 635,
                    "characterOffsetEnd": 640,
                    "word": "trick",
                    "originalText": "trick",
                },
                {
                    "characterOffsetBegin": 641,
                    "characterOffsetEnd": 649,
                    "word": "visitors",
                    "originalText": "visitors",
                },
                {
                    "characterOffsetBegin": 649,
                    "characterOffsetEnd": 650,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [533, 650],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 651,
                    "characterOffsetEnd": 653,
                    "word": "In",
                    "originalText": "In",
                },
                {
                    "characterOffsetBegin": 654,
                    "characterOffsetEnd": 655,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 656,
                    "characterOffsetEnd": 663,
                    "word": "stament",
                    "originalText": "stament",
                },
                {
                    "characterOffsetBegin": 664,
                    "characterOffsetEnd": 668,
                    "word": "sent",
                    "originalText": "sent",
                },
                {
                    "characterOffsetBegin": 669,
                    "characterOffsetEnd": 671,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 672,
                    "characterOffsetEnd": 674,
                    "word": "EW",
                    "originalText": "EW",
                },
                {
                    "characterOffsetBegin": 674,
                    "characterOffsetEnd": 675,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 676,
                    "characterOffsetEnd": 677,
                    "word": "a",
                    "originalText": "a",
                },
                {
                    "characterOffsetBegin": 678,
                    "characterOffsetEnd": 685,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 686,
                    "characterOffsetEnd": 698,
                    "word": "spokesperson",
                    "originalText": "spokesperson",
                },
                {
                    "characterOffsetBegin": 699,
                    "characterOffsetEnd": 706,
                    "word": "assured",
                    "originalText": "assured",
                },
                {
                    "characterOffsetBegin": 707,
                    "characterOffsetEnd": 718,
                    "word": "subscribers",
                    "originalText": "subscribers",
                },
                {
                    "characterOffsetBegin": 719,
                    "characterOffsetEnd": 723,
                    "word": "that",
                    "originalText": "that",
                },
                {
                    "characterOffsetBegin": 724,
                    "characterOffsetEnd": 727,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 728,
                    "characterOffsetEnd": 735,
                    "word": "company",
                    "originalText": "company",
                },
                {
                    "characterOffsetBegin": 736,
                    "characterOffsetEnd": 741,
                    "word": "takes",
                    "originalText": "takes",
                },
                {
                    "characterOffsetBegin": 742,
                    "characterOffsetEnd": 745,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 746,
                    "characterOffsetEnd": 747,
                    "word": "“",
                    "originalText": "“",
                },
                {
                    "characterOffsetBegin": 747,
                    "characterOffsetEnd": 755,
                    "word": "security",
                    "originalText": "security",
                },
                {
                    "characterOffsetBegin": 756,
                    "characterOffsetEnd": 758,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 759,
                    "characterOffsetEnd": 762,
                    "word": "our",
                    "originalText": "our",
                },
                {
                    "characterOffsetBegin": 763,
                    "characterOffsetEnd": 770,
                    "word": "members",
                    "originalText": "members",
                },
                {
                    "characterOffsetBegin": 770,
                    "characterOffsetEnd": 771,
                    "word": "’",
                    "originalText": "’",
                },
                {
                    "characterOffsetBegin": 772,
                    "characterOffsetEnd": 780,
                    "word": "accounts",
                    "originalText": "accounts",
                },
                {
                    "characterOffsetBegin": 781,
                    "characterOffsetEnd": 790,
                    "word": "seriously",
                    "originalText": "seriously",
                },
                {
                    "characterOffsetBegin": 790,
                    "characterOffsetEnd": 791,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 791,
                    "characterOffsetEnd": 792,
                    "word": "”",
                    "originalText": "”",
                },
                {
                    "characterOffsetBegin": 793,
                    "characterOffsetEnd": 797,
                    "word": "also",
                    "originalText": "also",
                },
                {
                    "characterOffsetBegin": 798,
                    "characterOffsetEnd": 805,
                    "word": "stating",
                    "originalText": "stating",
                },
                {
                    "characterOffsetBegin": 806,
                    "characterOffsetEnd": 810,
                    "word": "that",
                    "originalText": "that",
                },
                {
                    "characterOffsetBegin": 811,
                    "characterOffsetEnd": 816,
                    "word": "these",
                    "originalText": "these",
                },
                {
                    "characterOffsetBegin": 817,
                    "characterOffsetEnd": 821,
                    "word": "type",
                    "originalText": "type",
                },
                {
                    "characterOffsetBegin": 822,
                    "characterOffsetEnd": 824,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 825,
                    "characterOffsetEnd": 830,
                    "word": "scams",
                    "originalText": "scams",
                },
                {
                    "characterOffsetBegin": 831,
                    "characterOffsetEnd": 834,
                    "word": "are",
                    "originalText": "are",
                },
                {
                    "characterOffsetBegin": 834,
                    "characterOffsetEnd": 837,
                    "word": "n’t",
                    "originalText": "n’t",
                },
                {
                    "characterOffsetBegin": 838,
                    "characterOffsetEnd": 846,
                    "word": "uncommon",
                    "originalText": "uncommon",
                },
                {
                    "characterOffsetBegin": 846,
                    "characterOffsetEnd": 847,
                    "word": ":",
                    "originalText": ":",
                },
                {
                    "characterOffsetBegin": 848,
                    "characterOffsetEnd": 849,
                    "word": "“",
                    "originalText": "“",
                },
                {
                    "characterOffsetBegin": 849,
                    "characterOffsetEnd": 856,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 857,
                    "characterOffsetEnd": 864,
                    "word": "employs",
                    "originalText": "employs",
                },
                {
                    "characterOffsetBegin": 865,
                    "characterOffsetEnd": 873,
                    "word": "numerous",
                    "originalText": "numerous",
                },
                {
                    "characterOffsetBegin": 874,
                    "characterOffsetEnd": 883,
                    "word": "proactive",
                    "originalText": "proactive",
                },
                {
                    "characterOffsetBegin": 884,
                    "characterOffsetEnd": 892,
                    "word": "measures",
                    "originalText": "measures",
                },
                {
                    "characterOffsetBegin": 893,
                    "characterOffsetEnd": 895,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 896,
                    "characterOffsetEnd": 902,
                    "word": "detect",
                    "originalText": "detect",
                },
                {
                    "characterOffsetBegin": 903,
                    "characterOffsetEnd": 913,
                    "word": "fraudulent",
                    "originalText": "fraudulent",
                },
                {
                    "characterOffsetBegin": 914,
                    "characterOffsetEnd": 922,
                    "word": "activity",
                    "originalText": "activity",
                },
                {
                    "characterOffsetBegin": 923,
                    "characterOffsetEnd": 925,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 926,
                    "characterOffsetEnd": 930,
                    "word": "keep",
                    "originalText": "keep",
                },
                {
                    "characterOffsetBegin": 931,
                    "characterOffsetEnd": 934,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 935,
                    "characterOffsetEnd": 942,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 943,
                    "characterOffsetEnd": 950,
                    "word": "service",
                    "originalText": "service",
                },
                {
                    "characterOffsetBegin": 951,
                    "characterOffsetEnd": 954,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 955,
                    "characterOffsetEnd": 958,
                    "word": "our",
                    "originalText": "our",
                },
                {
                    "characterOffsetBegin": 959,
                    "characterOffsetEnd": 966,
                    "word": "members",
                    "originalText": "members",
                },
                {
                    "characterOffsetBegin": 966,
                    "characterOffsetEnd": 967,
                    "word": "’",
                    "originalText": "’",
                },
                {
                    "characterOffsetBegin": 968,
                    "characterOffsetEnd": 976,
                    "word": "accounts",
                    "originalText": "accounts",
                },
                {
                    "characterOffsetBegin": 977,
                    "characterOffsetEnd": 983,
                    "word": "secure",
                    "originalText": "secure",
                },
                {
                    "characterOffsetBegin": 983,
                    "characterOffsetEnd": 984,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [651, 984],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 985,
                    "characterOffsetEnd": 998,
                    "word": "Unfortunately",
                    "originalText": "Unfortunately",
                },
                {
                    "characterOffsetBegin": 998,
                    "characterOffsetEnd": 999,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 1000,
                    "characterOffsetEnd": 1005,
                    "word": "these",
                    "originalText": "these",
                },
                {
                    "characterOffsetBegin": 1006,
                    "characterOffsetEnd": 1011,
                    "word": "scams",
                    "originalText": "scams",
                },
                {
                    "characterOffsetBegin": 1012,
                    "characterOffsetEnd": 1015,
                    "word": "are",
                    "originalText": "are",
                },
                {
                    "characterOffsetBegin": 1016,
                    "characterOffsetEnd": 1022,
                    "word": "common",
                    "originalText": "common",
                },
                {
                    "characterOffsetBegin": 1023,
                    "characterOffsetEnd": 1025,
                    "word": "on",
                    "originalText": "on",
                },
                {
                    "characterOffsetBegin": 1026,
                    "characterOffsetEnd": 1029,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1030,
                    "characterOffsetEnd": 1038,
                    "word": "internet",
                    "originalText": "internet",
                },
                {
                    "characterOffsetBegin": 1039,
                    "characterOffsetEnd": 1042,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 1043,
                    "characterOffsetEnd": 1049,
                    "word": "target",
                    "originalText": "target",
                },
                {
                    "characterOffsetBegin": 1050,
                    "characterOffsetEnd": 1057,
                    "word": "popular",
                    "originalText": "popular",
                },
                {
                    "characterOffsetBegin": 1058,
                    "characterOffsetEnd": 1064,
                    "word": "brands",
                    "originalText": "brands",
                },
                {
                    "characterOffsetBegin": 1065,
                    "characterOffsetEnd": 1069,
                    "word": "such",
                    "originalText": "such",
                },
                {
                    "characterOffsetBegin": 1070,
                    "characterOffsetEnd": 1072,
                    "word": "as",
                    "originalText": "as",
                },
                {
                    "characterOffsetBegin": 1073,
                    "characterOffsetEnd": 1080,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 1081,
                    "characterOffsetEnd": 1084,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 1085,
                    "characterOffsetEnd": 1090,
                    "word": "other",
                    "originalText": "other",
                },
                {
                    "characterOffsetBegin": 1091,
                    "characterOffsetEnd": 1100,
                    "word": "companies",
                    "originalText": "companies",
                },
                {
                    "characterOffsetBegin": 1101,
                    "characterOffsetEnd": 1105,
                    "word": "with",
                    "originalText": "with",
                },
                {
                    "characterOffsetBegin": 1106,
                    "characterOffsetEnd": 1111,
                    "word": "large",
                    "originalText": "large",
                },
                {
                    "characterOffsetBegin": 1112,
                    "characterOffsetEnd": 1120,
                    "word": "customer",
                    "originalText": "customer",
                },
                {
                    "characterOffsetBegin": 1121,
                    "characterOffsetEnd": 1126,
                    "word": "bases",
                    "originalText": "bases",
                },
                {
                    "characterOffsetBegin": 1127,
                    "characterOffsetEnd": 1129,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 1130,
                    "characterOffsetEnd": 1134,
                    "word": "lure",
                    "originalText": "lure",
                },
                {
                    "characterOffsetBegin": 1135,
                    "characterOffsetEnd": 1140,
                    "word": "users",
                    "originalText": "users",
                },
                {
                    "characterOffsetBegin": 1141,
                    "characterOffsetEnd": 1145,
                    "word": "into",
                    "originalText": "into",
                },
                {
                    "characterOffsetBegin": 1146,
                    "characterOffsetEnd": 1152,
                    "word": "giving",
                    "originalText": "giving",
                },
                {
                    "characterOffsetBegin": 1153,
                    "characterOffsetEnd": 1156,
                    "word": "out",
                    "originalText": "out",
                },
                {
                    "characterOffsetBegin": 1157,
                    "characterOffsetEnd": 1165,
                    "word": "personal",
                    "originalText": "personal",
                },
                {
                    "characterOffsetBegin": 1166,
                    "characterOffsetEnd": 1177,
                    "word": "information",
                    "originalText": "information",
                },
                {
                    "characterOffsetBegin": 1177,
                    "characterOffsetEnd": 1178,
                    "word": ".",
                    "originalText": ".",
                },
                {
                    "characterOffsetBegin": 1178,
                    "characterOffsetEnd": 1179,
                    "word": "”",
                    "originalText": "”",
                },
            ],
            "span": [985, 1179],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 1180,
                    "characterOffsetEnd": 1189,
                    "word": "According",
                    "originalText": "According",
                },
                {
                    "characterOffsetBegin": 1190,
                    "characterOffsetEnd": 1192,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 1193,
                    "characterOffsetEnd": 1202,
                    "word": "Mailguard",
                    "originalText": "Mailguard",
                },
                {
                    "characterOffsetBegin": 1202,
                    "characterOffsetEnd": 1204,
                    "word": "’s",
                    "originalText": "’s",
                },
                {
                    "characterOffsetBegin": 1205,
                    "characterOffsetEnd": 1211,
                    "word": "report",
                    "originalText": "report",
                },
                {
                    "characterOffsetBegin": 1211,
                    "characterOffsetEnd": 1212,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 1213,
                    "characterOffsetEnd": 1216,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1217,
                    "characterOffsetEnd": 1221,
                    "word": "scam",
                    "originalText": "scam",
                },
                {
                    "characterOffsetBegin": 1222,
                    "characterOffsetEnd": 1225,
                    "word": "has",
                    "originalText": "has",
                },
                {
                    "characterOffsetBegin": 1226,
                    "characterOffsetEnd": 1234,
                    "word": "targeted",
                    "originalText": "targeted",
                },
                {
                    "characterOffsetBegin": 1235,
                    "characterOffsetEnd": 1241,
                    "word": "almost",
                    "originalText": "almost",
                },
                {
                    "characterOffsetBegin": 1242,
                    "characterOffsetEnd": 1245,
                    "word": "110",
                    "originalText": "110",
                },
                {
                    "characterOffsetBegin": 1246,
                    "characterOffsetEnd": 1253,
                    "word": "million",
                    "originalText": "million",
                },
                {
                    "characterOffsetBegin": 1254,
                    "characterOffsetEnd": 1265,
                    "word": "subscribers",
                    "originalText": "subscribers",
                },
                {
                    "characterOffsetBegin": 1265,
                    "characterOffsetEnd": 1266,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [1180, 1266],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 1267,
                    "characterOffsetEnd": 1270,
                    "word": "One",
                    "originalText": "One",
                },
                {
                    "characterOffsetBegin": 1271,
                    "characterOffsetEnd": 1280,
                    "word": "important",
                    "originalText": "important",
                },
                {
                    "characterOffsetBegin": 1281,
                    "characterOffsetEnd": 1286,
                    "word": "thing",
                    "originalText": "thing",
                },
                {
                    "characterOffsetBegin": 1287,
                    "characterOffsetEnd": 1289,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 1290,
                    "characterOffsetEnd": 1294,
                    "word": "note",
                    "originalText": "note",
                },
                {
                    "characterOffsetBegin": 1295,
                    "characterOffsetEnd": 1297,
                    "word": "is",
                    "originalText": "is",
                },
                {
                    "characterOffsetBegin": 1298,
                    "characterOffsetEnd": 1302,
                    "word": "that",
                    "originalText": "that",
                },
                {
                    "characterOffsetBegin": 1303,
                    "characterOffsetEnd": 1306,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1307,
                    "characterOffsetEnd": 1312,
                    "word": "email",
                    "originalText": "email",
                },
                {
                    "characterOffsetBegin": 1312,
                    "characterOffsetEnd": 1314,
                    "word": "’s",
                    "originalText": "’s",
                },
                {
                    "characterOffsetBegin": 1315,
                    "characterOffsetEnd": 1324,
                    "word": "recipient",
                    "originalText": "recipient",
                },
                {
                    "characterOffsetBegin": 1325,
                    "characterOffsetEnd": 1332,
                    "word": "appears",
                    "originalText": "appears",
                },
                {
                    "characterOffsetBegin": 1333,
                    "characterOffsetEnd": 1335,
                    "word": "as",
                    "originalText": "as",
                },
                {
                    "characterOffsetBegin": 1336,
                    "characterOffsetEnd": 1337,
                    "word": "“",
                    "originalText": "“",
                },
                {
                    "characterOffsetBegin": 1337,
                    "characterOffsetEnd": 1339,
                    "word": "no",
                    "originalText": "no",
                },
                {
                    "characterOffsetBegin": 1340,
                    "characterOffsetEnd": 1346,
                    "word": "sender",
                    "originalText": "sender",
                },
                {
                    "characterOffsetBegin": 1346,
                    "characterOffsetEnd": 1347,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 1347,
                    "characterOffsetEnd": 1348,
                    "word": "”",
                    "originalText": "”",
                },
                {
                    "characterOffsetBegin": 1349,
                    "characterOffsetEnd": 1353,
                    "word": "plus",
                    "originalText": "plus",
                },
                {
                    "characterOffsetBegin": 1354,
                    "characterOffsetEnd": 1357,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1358,
                    "characterOffsetEnd": 1364,
                    "word": "victim",
                    "originalText": "victim",
                },
                {
                    "characterOffsetBegin": 1364,
                    "characterOffsetEnd": 1366,
                    "word": "’s",
                    "originalText": "’s",
                },
                {
                    "characterOffsetBegin": 1367,
                    "characterOffsetEnd": 1371,
                    "word": "name",
                    "originalText": "name",
                },
                {
                    "characterOffsetBegin": 1372,
                    "characterOffsetEnd": 1379,
                    "word": "appears",
                    "originalText": "appears",
                },
                {
                    "characterOffsetBegin": 1380,
                    "characterOffsetEnd": 1382,
                    "word": "as",
                    "originalText": "as",
                },
                {
                    "characterOffsetBegin": 1383,
                    "characterOffsetEnd": 1384,
                    "word": "“",
                    "originalText": "“",
                },
                {
                    "characterOffsetBegin": 1384,
                    "characterOffsetEnd": 1389,
                    "word": "#name",
                    "originalText": "#name",
                },
                {
                    "characterOffsetBegin": 1389,
                    "characterOffsetEnd": 1390,
                    "word": "#",
                    "originalText": "#",
                },
                {
                    "characterOffsetBegin": 1390,
                    "characterOffsetEnd": 1391,
                    "word": ",",
                    "originalText": ",",
                },
                {
                    "characterOffsetBegin": 1391,
                    "characterOffsetEnd": 1392,
                    "word": "”",
                    "originalText": "”",
                },
                {
                    "characterOffsetBegin": 1393,
                    "characterOffsetEnd": 1395,
                    "word": "as",
                    "originalText": "as",
                },
                {
                    "characterOffsetBegin": 1396,
                    "characterOffsetEnd": 1401,
                    "word": "shown",
                    "originalText": "shown",
                },
                {
                    "characterOffsetBegin": 1402,
                    "characterOffsetEnd": 1404,
                    "word": "in",
                    "originalText": "in",
                },
                {
                    "characterOffsetBegin": 1405,
                    "characterOffsetEnd": 1408,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1409,
                    "characterOffsetEnd": 1419,
                    "word": "screenshot",
                    "originalText": "screenshot",
                },
                {
                    "characterOffsetBegin": 1419,
                    "characterOffsetEnd": 1420,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [1267, 1420],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 1421,
                    "characterOffsetEnd": 1428,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 1429,
                    "characterOffsetEnd": 1438,
                    "word": "customers",
                    "originalText": "customers",
                },
                {
                    "characterOffsetBegin": 1439,
                    "characterOffsetEnd": 1442,
                    "word": "who",
                    "originalText": "who",
                },
                {
                    "characterOffsetBegin": 1443,
                    "characterOffsetEnd": 1450,
                    "word": "receive",
                    "originalText": "receive",
                },
                {
                    "characterOffsetBegin": 1451,
                    "characterOffsetEnd": 1455,
                    "word": "this",
                    "originalText": "this",
                },
                {
                    "characterOffsetBegin": 1456,
                    "characterOffsetEnd": 1461,
                    "word": "email",
                    "originalText": "email",
                },
                {
                    "characterOffsetBegin": 1462,
                    "characterOffsetEnd": 1465,
                    "word": "are",
                    "originalText": "are",
                },
                {
                    "characterOffsetBegin": 1466,
                    "characterOffsetEnd": 1473,
                    "word": "advised",
                    "originalText": "advised",
                },
                {
                    "characterOffsetBegin": 1474,
                    "characterOffsetEnd": 1476,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 1477,
                    "characterOffsetEnd": 1484,
                    "word": "abstain",
                    "originalText": "abstain",
                },
                {
                    "characterOffsetBegin": 1485,
                    "characterOffsetEnd": 1489,
                    "word": "from",
                    "originalText": "from",
                },
                {
                    "characterOffsetBegin": 1490,
                    "characterOffsetEnd": 1497,
                    "word": "filling",
                    "originalText": "filling",
                },
                {
                    "characterOffsetBegin": 1498,
                    "characterOffsetEnd": 1501,
                    "word": "out",
                    "originalText": "out",
                },
                {
                    "characterOffsetBegin": 1502,
                    "characterOffsetEnd": 1505,
                    "word": "any",
                    "originalText": "any",
                },
                {
                    "characterOffsetBegin": 1506,
                    "characterOffsetEnd": 1517,
                    "word": "information",
                    "originalText": "information",
                },
                {
                    "characterOffsetBegin": 1518,
                    "characterOffsetEnd": 1526,
                    "word": "prompted",
                    "originalText": "prompted",
                },
                {
                    "characterOffsetBegin": 1527,
                    "characterOffsetEnd": 1529,
                    "word": "by",
                    "originalText": "by",
                },
                {
                    "characterOffsetBegin": 1530,
                    "characterOffsetEnd": 1533,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1534,
                    "characterOffsetEnd": 1541,
                    "word": "website",
                    "originalText": "website",
                },
                {
                    "characterOffsetBegin": 1541,
                    "characterOffsetEnd": 1542,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [1421, 1542],
        },
        {
            "tokens": [
                {
                    "characterOffsetBegin": 1543,
                    "characterOffsetEnd": 1550,
                    "word": "Netflix",
                    "originalText": "Netflix",
                },
                {
                    "characterOffsetBegin": 1550,
                    "characterOffsetEnd": 1552,
                    "word": "’s",
                    "originalText": "’s",
                },
                {
                    "characterOffsetBegin": 1553,
                    "characterOffsetEnd": 1565,
                    "word": "spokesperson",
                    "originalText": "spokesperson",
                },
                {
                    "characterOffsetBegin": 1566,
                    "characterOffsetEnd": 1570,
                    "word": "also",
                    "originalText": "also",
                },
                {
                    "characterOffsetBegin": 1571,
                    "characterOffsetEnd": 1580,
                    "word": "suggested",
                    "originalText": "suggested",
                },
                {
                    "characterOffsetBegin": 1581,
                    "characterOffsetEnd": 1585,
                    "word": "that",
                    "originalText": "that",
                },
                {
                    "characterOffsetBegin": 1586,
                    "characterOffsetEnd": 1593,
                    "word": "members",
                    "originalText": "members",
                },
                {
                    "characterOffsetBegin": 1594,
                    "characterOffsetEnd": 1596,
                    "word": "of",
                    "originalText": "of",
                },
                {
                    "characterOffsetBegin": 1597,
                    "characterOffsetEnd": 1600,
                    "word": "the",
                    "originalText": "the",
                },
                {
                    "characterOffsetBegin": 1601,
                    "characterOffsetEnd": 1610,
                    "word": "streaming",
                    "originalText": "streaming",
                },
                {
                    "characterOffsetBegin": 1611,
                    "characterOffsetEnd": 1618,
                    "word": "service",
                    "originalText": "service",
                },
                {
                    "characterOffsetBegin": 1619,
                    "characterOffsetEnd": 1624,
                    "word": "visit",
                    "originalText": "visit",
                },
                {
                    "characterOffsetBegin": 1625,
                    "characterOffsetEnd": 1645,
                    "word": "netflix.com/security",
                    "originalText": "netflix.com/security",
                },
                {
                    "characterOffsetBegin": 1646,
                    "characterOffsetEnd": 1648,
                    "word": "or",
                    "originalText": "or",
                },
                {
                    "characterOffsetBegin": 1649,
                    "characterOffsetEnd": 1656,
                    "word": "contact",
                    "originalText": "contact",
                },
                {
                    "characterOffsetBegin": 1657,
                    "characterOffsetEnd": 1665,
                    "word": "Customer",
                    "originalText": "Customer",
                },
                {
                    "characterOffsetBegin": 1666,
                    "characterOffsetEnd": 1673,
                    "word": "Service",
                    "originalText": "Service",
                },
                {
                    "characterOffsetBegin": 1674,
                    "characterOffsetEnd": 1682,
                    "word": "directly",
                    "originalText": "directly",
                },
                {
                    "characterOffsetBegin": 1683,
                    "characterOffsetEnd": 1685,
                    "word": "to",
                    "originalText": "to",
                },
                {
                    "characterOffsetBegin": 1686,
                    "characterOffsetEnd": 1691,
                    "word": "learn",
                    "originalText": "learn",
                },
                {
                    "characterOffsetBegin": 1692,
                    "characterOffsetEnd": 1696,
                    "word": "more",
                    "originalText": "more",
                },
                {
                    "characterOffsetBegin": 1697,
                    "characterOffsetEnd": 1708,
                    "word": "information",
                    "originalText": "information",
                },
                {
                    "characterOffsetBegin": 1709,
                    "characterOffsetEnd": 1714,
                    "word": "about",
                    "originalText": "about",
                },
                {
                    "characterOffsetBegin": 1715,
                    "characterOffsetEnd": 1720,
                    "word": "scams",
                    "originalText": "scams",
                },
                {
                    "characterOffsetBegin": 1721,
                    "characterOffsetEnd": 1724,
                    "word": "and",
                    "originalText": "and",
                },
                {
                    "characterOffsetBegin": 1725,
                    "characterOffsetEnd": 1730,
                    "word": "other",
                    "originalText": "other",
                },
                {
                    "characterOffsetBegin": 1731,
                    "characterOffsetEnd": 1740,
                    "word": "malicious",
                    "originalText": "malicious",
                },
                {
                    "characterOffsetBegin": 1741,
                    "characterOffsetEnd": 1749,
                    "word": "activity",
                    "originalText": "activity",
                },
                {
                    "characterOffsetBegin": 1749,
                    "characterOffsetEnd": 1750,
                    "word": ".",
                    "originalText": ".",
                },
            ],
            "span": [1543, 1750],
        },
    ],
    "text": "An email scam passing as a Netflix notification has been targeting subscribers of the streaming service. The “suspension notification” looks similar in design and format to other Netflix emails. It notifies and urges users to update their information to avoid the suspension of their account. The goal of the scam is to steal personal and credit card information, according to a report from Mailguard. The email contains a link to a fake Netflix website where users are required to enter log-in information and a credit card number. The faux website has the Netflix logo on display plus The Crown and House of Cards banners to further trick visitors. In a stament sent to EW, a Netflix spokesperson assured subscribers that the company takes the “security of our members’ accounts seriously,” also stating that these type of scams aren’t uncommon: “Netflix employs numerous proactive measures to detect fraudulent activity to keep the Netflix service and our members’ accounts secure. Unfortunately, these scams are common on the internet and target popular brands such as Netflix and other companies with large customer bases to lure users into giving out personal information.” According to Mailguard’s report, the scam has targeted almost 110 million subscribers. One important thing to note is that the email’s recipient appears as “no sender,” plus the victim’s name appears as “#name#,” as shown in the screenshot. Netflix customers who receive this email are advised to abstain from filling out any information prompted by the website. Netflix’s spokesperson also suggested that members of the streaming service visit netflix.com/security or contact Customer Service directly to learn more information about scams and other malicious activity.",
    "stanford_coref": {},
    "event": [
        {
            "id": "10231-0",
            "mentions": [
                {
                    "id": "10231-0-0",
                    "type": "Attack",
                    "subtype": "Phishing",
                    "realis": "Actual",
                    "nugget": {
                        "text": "email scam",
                        "span": [3, 12],
                        "tokens": [[0, 1], [0, 2]],
                    },
                    "arguments": [
                        {
                            "id": "10231-0-0-0",
                            "role": "Trusted-Entity",
                            "filler_type": "File",
                            "text": "a Netflix notification",
                            "span": [25, 46],
                            "tokens": [[0, 5], [0, 6], [0, 7]],
                        },
                        {
                            "id": "10231-0-0-1",
                            "role": "Trusted-Entity",
                            "filler_type": "System",
                            "text": "the streaming service",
                            "span": [82, 102],
                            "tokens": [[0, 13], [0, 14], [0, 15]],
                        },
                        {
                            "id": "10231-0-0-2",
                            "role": "Tool",
                            "filler_type": "File",
                            "text": "The “suspension notification”",
                            "span": [105, 133],
                            "tokens": [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
                        },
                        {
                            "id": "10231-0-0-3",
                            "role": "Trusted-Entity",
                            "filler_type": "Data",
                            "text": "Netflix emails",
                            "span": [179, 192],
                            "tokens": [[1, 13], [1, 14]],
                        },
                        {
                            "id": "10231-0-0-4",
                            "role": "Victim",
                            "filler_type": "Person",
                            "text": "subscribers",
                            "span": [67, 77],
                            "tokens": [[0, 11]],
                        },
                    ],
                },
                {
                    "id": "10231-0-1",
                    "type": "Attack",
                    "subtype": "Phishing",
                    "realis": "Actual",
                    "nugget": {
                        "text": "the scam",
                        "span": [305, 312],
                        "tokens": [[3, 3], [3, 4]],
                    },
                    "arguments": [
                        {
                            "id": "10231-0-1-0",
                            "role": "Purpose",
                            "filler_type": "Purpose",
                            "text": "steal personal and credit card information",
                            "span": [320, 361],
                            "tokens": [
                                [3, 7],
                                [3, 8],
                                [3, 9],
                                [3, 10],
                                [3, 11],
                                [3, 12],
                            ],
                        },
                        {
                            "id": "10231-0-1-1",
                            "role": "Tool",
                            "filler_type": "File",
                            "text": "The email",
                            "span": [402, 410],
                            "tokens": [[4, 0], [4, 1]],
                        },
                        {
                            "id": "10231-0-1-2",
                            "role": "Tool",
                            "filler_type": "Website",
                            "text": "a fake Netflix website",
                            "span": [431, 452],
                            "tokens": [[4, 6], [4, 7], [4, 8], [4, 9]],
                        },
                        {
                            "id": "10231-0-1-3",
                            "role": "Purpose",
                            "filler_type": "Purpose",
                            "text": "are required to enter log-in information and a credit card number",
                            "span": [466, 530],
                            "tokens": [
                                [4, 12],
                                [4, 13],
                                [4, 14],
                                [4, 15],
                                [4, 16],
                                [4, 17],
                                [4, 18],
                                [4, 19],
                                [4, 20],
                                [4, 21],
                                [4, 22],
                                [4, 23],
                                [4, 24],
                            ],
                        },
                        {
                            "id": "10231-0-1-4",
                            "role": "Victim",
                            "filler_type": "Person",
                            "text": "users",
                            "span": [460, 464],
                            "tokens": [[4, 11]],
                        },
                    ],
                },
            ],
        },
        {
            "id": "10231-1",
            "mentions": [
                {
                    "id": "10231-1-0",
                    "type": "Attack",
                    "subtype": "Phishing",
                    "realis": "Actual",
                    "nugget": {
                        "text": "further trick",
                        "span": [627, 639],
                        "tokens": [[5, 18], [5, 19]],
                    },
                    "arguments": [
                        {
                            "id": "10231-1-0-0",
                            "role": "Victim",
                            "filler_type": "Person",
                            "text": "visitors",
                            "span": [641, 648],
                            "tokens": [[5, 20]],
                        },
                        {
                            "id": "10231-1-0-1",
                            "role": "Tool",
                            "filler_type": "File",
                            "text": "The Crown and House of Cards banners",
                            "span": [587, 622],
                            "tokens": [
                                [5, 10],
                                [5, 11],
                                [5, 12],
                                [5, 13],
                                [5, 14],
                                [5, 15],
                                [5, 16],
                            ],
                        },
                        {
                            "id": "10231-1-0-2",
                            "role": "Trusted-Entity",
                            "filler_type": "File",
                            "text": "the Netflix logo",
                            "span": [554, 569],
                            "tokens": [[5, 4], [5, 5], [5, 6]],
                        },
                        {
                            "id": "10231-1-0-3",
                            "role": "Tool",
                            "filler_type": "Website",
                            "text": "The faux website",
                            "span": [533, 548],
                            "tokens": [[5, 0], [5, 1], [5, 2]],
                        },
                    ],
                }
            ],
        },
        {
            "id": "10231-2",
            "mentions": [
                {
                    "id": "10231-2-0",
                    "type": "Attack",
                    "subtype": "Phishing",
                    "realis": "Generic",
                    "nugget": {
                        "text": "lure",
                        "span": [1130, 1133],
                        "tokens": [[7, 24]],
                    },
                    "arguments": [
                        {
                            "id": "10231-2-0-0",
                            "role": "Trusted-Entity",
                            "filler_type": "Organization",
                            "text": "other companies",
                            "span": [1085, 1099],
                            "tokens": [[7, 17], [7, 18]],
                        },
                        {
                            "id": "10231-2-0-1",
                            "role": "Trusted-Entity",
                            "filler_type": "Organization",
                            "text": "Netflix",
                            "span": [1073, 1079],
                            "tokens": [[7, 15]],
                        },
                        {
                            "id": "10231-2-0-2",
                            "role": "Victim",
                            "filler_type": "Person",
                            "text": "users",
                            "span": [1135, 1139],
                            "tokens": [[7, 25]],
                        },
                        {
                            "id": "10231-2-0-3",
                            "role": "Purpose",
                            "filler_type": "Purpose",
                            "text": "giving out personal information",
                            "span": [1146, 1176],
                            "tokens": [[7, 27], [7, 28], [7, 29], [7, 30]],
                        },
                    ],
                }
            ],
        },
        {
            "id": "10231-3",
            "mentions": [
                {
                    "id": "10231-3-0",
                    "type": "Attack",
                    "subtype": "Phishing",
                    "realis": "Generic",
                    "nugget": {
                        "text": "these scams",
                        "span": [1000, 1010],
                        "tokens": [[7, 2], [7, 3]],
                    },
                    "arguments": [],
                }
            ],
        },
    ],
    "info": {
        "title": "Netflix subscribers targeted by scam email",
        "date": "2017_11_06",
        "type": "text",
        "link": "https://ew.com/tv/2017/11/06/netflix-subscribers-scam-email/",
    },
}
"""
