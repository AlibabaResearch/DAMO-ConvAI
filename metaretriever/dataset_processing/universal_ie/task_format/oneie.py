#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span


"""
{
  "doc_id": "AFP_ENG_20030427.0118",
  "sent_id": "AFP_ENG_20030427.0118-1",
  "tokens": ["A", "Pakistani", "court", "in", "central", "Punjab", "province", "has", "sentenced", "a", "Christian", "man", "to", "life", "imprisonment", "for", "a", "blasphemy", "conviction", ",", "police", "said", "Sunday", "."], "pieces": ["A", "Pakistani", "court", "in", "central", "Punjab", "province", "has", "sentenced", "a", "Christian", "man", "to", "life", "imprisonment", "for", "a", "b", "##lasp", "##hem", "##y", "conviction", ",", "police", "said", "Sunday", "."],
  "token_lens": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1],
  "sentence": "A Pakistani court in central Punjab province has sentenced a Christian man to life imprisonment for a blasphemy conviction, police said Sunday.",
  "entity_mentions": [
    {"id": "AFP_ENG_20030427.0118-E15-53", "text": "Pakistani", "entity_type": "GPE", "mention_type": "NAM", "entity_subtype": "Nation", "start": 1, "end": 2},
    {"id": "AFP_ENG_20030427.0118-E35-52", "text": "court", "entity_type": "ORG", "mention_type": "NOM", "entity_subtype": "Government", "start": 2, "end": 3},
    {"id": "AFP_ENG_20030427.0118-E37-54", "text": "province", "entity_type": "LOC", "mention_type": "NOM", "entity_subtype": "Region-General", "start": 6, "end": 7},
    {"id": "AFP_ENG_20030427.0118-E27-48", "text": "Christian", "entity_type": "PER", "mention_type": "NOM", "entity_subtype": "Group", "start": 10, "end": 11},
    {"id": "AFP_ENG_20030427.0118-E38-55", "text": "man", "entity_type": "PER", "mention_type": "NOM", "entity_subtype": "Individual", "start": 11, "end": 12},
    {"id": "AFP_ENG_20030427.0118-E39-56", "text": "police", "entity_type": "PER", "mention_type": "NOM", "entity_subtype": "Group", "start": 20, "end": 21}],
  "relation_mentions": [
    {"id": "AFP_ENG_20030427.0118-R1-1", "relation_type": "GEN-AFF", "relation_subtype": "GEN-AFF:Citizen-Resident-Religion-Ethnicity",
      "arguments": [
        {"entity_id": "AFP_ENG_20030427.0118-E38-55", "text": "man", "role": "Arg-1"}, 
        {"entity_id": "AFP_ENG_20030427.0118-E27-48", "text": "Christian", "role": "Arg-2"}
      ]
    },
    {"id": "AFP_ENG_20030427.0118-R3-1", "relation_type": "PART-WHOLE", "relation_subtype": "PART-WHOLE:Subsidiary",
      "arguments": [
        {"entity_id": "AFP_ENG_20030427.0118-E35-52", "text": "court", "role": "Arg-1"},
        {"entity_id": "AFP_ENG_20030427.0118-E15-53", "text": "Pakistani", "role": "Arg-2"}
      ]
    },
    {"id": "AFP_ENG_20030427.0118-R4-1", "relation_type": "GEN-AFF", "relation_subtype": "GEN-AFF:Org-Location",
      "arguments": [
        {"entity_id": "AFP_ENG_20030427.0118-E35-52", "text": "court", "role": "Arg-1"},
        {"entity_id": "AFP_ENG_20030427.0118-E37-54", "text": "province", "role": "Arg-2"}
      ]
    }
  ],
  "event_mentions": [
    {"id": "AFP_ENG_20030427.0118-EV1-1", "event_type": "Justice:Sentence",
      "trigger": {"text": "sentenced", "start": 8, "end": 9},
      "arguments": [
        {"entity_id": "AFP_ENG_20030427.0118-E35-52", "text": "court", "role": "Adjudicator"},
        {"entity_id": "AFP_ENG_20030427.0118-E38-55", "text": "man", "role": "Defendant"},
        {"entity_id": "AFP_ENG_20030427.0118-E37-54", "text": "province", "role": "Place"}
    ]},
    {"id": "AFP_ENG_20030427.0118-EV2-1", "event_type": "Justice:Convict",
      "trigger": {"text": "conviction", "start": 18, "end": 19},
      "arguments": [{"entity_id": "AFP_ENG_20030427.0118-E38-55", "text": "man", "role": "Defendant"}
    ]}
]}
"""


class OneIEEvent(TaskFormat):
    def __init__(self, doc_json, language='en'):
        super().__init__(
            language=language
        )
        self.doc_id = doc_json['doc_id']
        self.sent_id = doc_json['sent_id']
        self.tokens = doc_json['tokens']
        self.entities = doc_json['entity_mentions']
        self.relations = doc_json['relation_mentions']
        self.events = doc_json['event_mentions']

    def generate_instance(self):
        events = dict()
        entities = dict()

        for span_index, span in enumerate(self.entities):
            tokens = self.tokens[span['start']: span['end']]
            indexes = list(range(span['start'], span['end']))
            entities[span['id']] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(span['entity_type']),
                text_id=self.sent_id,
                record_id=span['id']
            )

        for event_index, event in enumerate(self.events):
            start = event['trigger']['start']
            end = event['trigger']['end']
            tokens = self.tokens[start:end]
            indexes = list(range(start, end))
            events[event['id']] = Event(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(event['event_type']),
                args=[(Label(x['role']), entities[x['entity_id']])
                      for x in event['arguments']],
                text_id=self.sent_id,
                record_id=event['id']
            )

        return Sentence(
            tokens=self.tokens,
            entities=list(),
            relations=list(),
            events=events.values(),
            text_id=self.sent_id
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        with open(filename) as fin:
            for line in fin:
                instance = OneIEEvent(
                    json.loads(line.strip()),
                    language=language
                ).generate_instance()
                sentence_list += [instance]
        return sentence_list
