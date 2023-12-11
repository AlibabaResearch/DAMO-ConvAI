#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import os
import json
from typing import Dict, List
from tqdm import tqdm
from universal_ie.generation_format.generation_format import GenerationFormat
from universal_ie.generation_format import generation_format_dict
from universal_ie.generation_format.structure_marker import BaseStructureMarker
from universal_ie.dataset import Dataset
from universal_ie.ie_format import Sentence


def convert_graph(
    generation_class: GenerationFormat,
    output_folder: str,
    datasets: Dict[str, List[Sentence]],
    language: str = "en",
    label_mapper: Dict = None,
):
    convertor = generation_class(
        structure_maker=BaseStructureMarker(),
        language=language,
        label_mapper=label_mapper,
    )

    counter = Counter()

    os.makedirs(output_folder, exist_ok=True)

    schema_counter = {
        "entity": list(),
        "relation": list(),
        "event": list(),
    }
    for data_type, instance_list in datasets.items():
        with open(os.path.join(output_folder, f"{data_type}.json"), "w") as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])
                converted_graph = convertor.annonote_graph(
                    tokens=instance.tokens,
                    entities=instance.entities,
                    relations=instance.relations,
                    events=instance.events,
                )
                src, tgt, spot_labels, asoc_labels = converted_graph[:4]
                spot_asoc = converted_graph[4]

                schema_counter["entity"] += instance.entities
                schema_counter["relation"] += instance.relations
                schema_counter["event"] += instance.events

                output.write(
                    "%s\n"
                    % json.dumps(
                        {
                            "text": src,
                            "tokens": instance.tokens,
                            "record": tgt,
                            "entity": [
                                entity.to_offset(label_mapper)
                                for entity in instance.entities
                            ],
                            "relation": [
                                relation.to_offset(
                                    ent_label_mapper=label_mapper,
                                    rel_label_mapper=label_mapper,
                                )
                                for relation in instance.relations
                            ],
                            "event": [
                                event.to_offset(evt_label_mapper=label_mapper)
                                for event in instance.events
                            ],
                            "spot": list(spot_labels),
                            "asoc": list(asoc_labels),
                            "spot_asoc": spot_asoc,
                        },
                        ensure_ascii=False,
                    )
                )
    convertor.output_schema(os.path.join(output_folder, "record.schema"))
    convertor.get_entity_schema(schema_counter["entity"]).write_to_file(
        os.path.join(output_folder, f"entity.schema")
    )
    convertor.get_relation_schema(schema_counter["relation"]).write_to_file(
        os.path.join(output_folder, f"relation.schema")
    )
    convertor.get_event_schema(schema_counter["event"]).write_to_file(
        os.path.join(output_folder, f"event.schema")
    )
    print(counter)
    print(output_folder)
    print("==========================")


def convert_to_oneie(output_folder: str, datasets: Dict[str, List[Sentence]]):
    os.makedirs(output_folder, exist_ok=True)
    counter = Counter()

    for data_type, instance_list in datasets.items():
        with open(
            os.path.join(output_folder, f"{data_type}.oneie.json"), "w"
        ) as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])
                entity_mentions = [
                    {
                        "id": entity.record_id,
                        "entity_type": str(entity.label),
                        "text": entity.span.text,
                        "start": entity.span.indexes[0],
                        "end": entity.span.indexes[-1] + 1,
                    }
                    for entity in instance.entities
                ]
                relation_mentions = [
                    {
                        "id": relation.record_id,
                        "relation_type": str(relation.label),
                        "argument": [
                            {
                                "entity_id": relation.arg1.record_id,
                                "text": relation.arg1.span.text,
                                "role": "Arg-1",
                            },
                            {
                                "entity_id": relation.arg2.record_id,
                                "text": relation.arg2.span.text,
                                "role": "Arg-2",
                            },
                        ],
                    }
                    for relation in instance.relations
                ]
                event_mentions = [
                    {
                        "id": event.record_id,
                        "event_type": str(event.label),
                        "trigger": {
                            "text": event.span.text,
                            "start": event.span.indexes[0],
                            "end": event.span.indexes[-1] + 1,
                        },
                        "argument": [
                            {
                                "id": arg[1].record_id,
                                "text": arg[1].span.text,
                                "role": str(arg[0]),
                            }
                            for arg in event.args
                        ],
                    }
                    for event in instance.events
                ]

                instance_dict = {
                    "tokens": instance.tokens,
                    "sent_id": instance.text_id,
                    "entity_mentions": entity_mentions,
                    "relation_mentions": relation_mentions,
                    "event_mentions": event_mentions,
                }
                instance_str = json.dumps(instance_dict, ensure_ascii=False)
                output.write(f"{instance_str}\n")

    print(counter)
    print(output_folder)
    print("==========================")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-format", dest="generation_format", default="spotasoc")
    parser.add_argument("-config", dest="config", default="data_config/relation")
    parser.add_argument("-output", dest="output", default="relation")
    options = parser.parse_args()

    generation_class = generation_format_dict.get(options.generation_format)

    if os.path.isfile(options.config):
        config_list = [options.config]
    else:
        config_list = [
            os.path.join(options.config, x) for x in os.listdir(options.config)
        ]

    for filename in config_list:
        dataset = Dataset.load_yaml_file(filename)

        datasets = dataset.load_dataset()
        label_mapper = dataset.mapper
        print(label_mapper)

        output_name = (
            f"converted_data/text2{options.generation_format}/{options.output}/"
            + dataset.name
        )

        if generation_class:
            convert_graph(
                generation_class,
                output_name,
                datasets=datasets,
                language=dataset.language,
                label_mapper=label_mapper,
            )
        elif options.generation_format == "oneie":
            convert_to_oneie(output_name, datasets=datasets)


if __name__ == "__main__":
    main()
