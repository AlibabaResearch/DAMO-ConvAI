import sys
import os
sys.path.append('/mnt/experiment/UnifiedData2TextPretrain/')
import argparse
import ast
import re
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from data_preprocess import utils

stopwords = set(stopwords.words('english'))


def tokenize(sent):
    return word_tokenize(sent)


def special_str_cleanout(sent):
    """
    for special_str in ['\u2013']:
        sent = sent.replace(special_str, '')
    """
    # sent = re.subn('(#U[0-9a-f]{4})', lambda cp: chr(int(cp.groups()[0][2:],16)), sent) [0]
    return sent


def two_dimension_relation_matrix_construct(n_entity, triples):
    """

    :param n_entity: int
    :param triples: (head_entity_id, relation, tail_entity_id) or (head_entity_id, tail_entity_id)
    :return:
    """
    relation_matrix = np.zeros([n_entity, n_entity])

    for triple in triples:
        head_entity_id = triple[0]
        tail_entity_id = triple[-1]

        relation_matrix[head_entity_id][tail_entity_id] += 1
        relation_matrix[tail_entity_id][head_entity_id] += 1

    return relation_matrix


def text_overlap_scorer(hyp, ref, lower=False):
    if lower:
        hyp = hyp.lower()
        ref = ref.lower()

    hyp_tokens = [token for token in hyp.split() if token not in stopwords]
    ref_tokens = [token for token in ref.split() if token not in stopwords]

    hyp_tokens = set(hyp_tokens)
    ref_tokens = set(ref_tokens)

    n_hit_token = 0
    for token in hyp_tokens:
        if token in ref_tokens:
            n_hit_token += 1

    overlap_score = n_hit_token / len(hyp_tokens) if n_hit_token > 0 else 0.0
    return overlap_score


def triple_cleanout(main_entities, entities, triples, target_text, node_remain_degree_lower_bound=2,
                    text_overlap_score_lower_bound=0.5, lower_text_overlap_score=False):
    """

    :param main_entities: list
    :param entities: list
    :param triples: list((head_entity, relation, tail_entity))
    :param target_text: list
    :param node_remain_degree_lower_bound
    :param text_overlap_score_lower_bound
    :param lower_text_overlap_score
    :return:
    """

    n_entity = len(entities)
    init_relation_matrix = two_dimension_relation_matrix_construct(n_entity=n_entity, triples=triples)

    node_degrees = init_relation_matrix.sum(-1)
    # print("n_entity", n_entity)
    # print("init_relation_matrix", init_relation_matrix)
    # print("node_degrees", node_degrees)
    # print("node_remain_degree_lower_bound", node_remain_degree_lower_bound)
    # if the degree of the node is smaller than the predefined lower bound, the node may be considered to remove
    init_removed_entity_ids = []
    for entity_idx, node_degree in enumerate(node_degrees):
        if entities[entity_idx] in main_entities:
            continue
        # print(node_degree, node_remain_degree_lower_bound)
        if node_degree < node_remain_degree_lower_bound:
            init_removed_entity_ids.append(entity_idx)
    # print("init_removed_entity_ids", len(init_removed_entity_ids))
    # if the init_removed_entity has no overlap with target text, then the node be consider removed

    text_overlap_removed_entity_ids = []
    for entity_idx in init_removed_entity_ids:
        entity = entities[entity_idx]

        text_overlap_score = text_overlap_scorer(hyp=entity, ref=target_text, lower=lower_text_overlap_score)
        # print("text_overlap_score", text_overlap_score, entity, target_text)
        if text_overlap_score < text_overlap_score_lower_bound:
            text_overlap_removed_entity_ids.append(entity_idx)

    # explore the neighbor of the node
    # print("text_overlap_removed_entity_ids", len(text_overlap_removed_entity_ids), text_overlap_removed_entity_ids)
    # filter the entity
    final_removed_entity_ids = text_overlap_removed_entity_ids

    final_entities = [entity for entity_idx, entity in enumerate(entities) if
                      entity_idx not in final_removed_entity_ids]

    return final_entities, final_removed_entity_ids


def convert_data_to_triples(kblinks, kg_knowledge):
    """

    :param kblinks: list[main_entity_id]
    :param kg_knowledge:
    dict(main_entity_knowledge: list(entity_name: str, description: str, relations: list(pair(relation, tail_entity)))
    :return:
    """

    main_entity_ids = []
    for main_entity_id in kblinks:
        if main_entity_id is not None and main_entity_id in kg_knowledge and main_entity_id not in main_entity_ids:
            main_entity_ids.append(main_entity_id)

    main_entities = []
    entities = []
    triples = []
    for main_entity_id in main_entity_ids:
        entity_knowledge = kg_knowledge[main_entity_id]
        main_entity = entity_knowledge[0]
        entity_description = entity_knowledge[1]

        main_entity = special_str_cleanout(main_entity)
        entity_description = special_str_cleanout(entity_description)

        main_entities.append(main_entity)
        if main_entity not in entities:
            head_entity_idx = len(entities)
            entities.append(main_entity)
        else:
            head_entity_idx = entities.index(main_entity)

        if len(entity_description) > 0:
            if entity_description not in entities:
                tail_entity_idx = len(entities)
                entities.append(entity_description)
            else:
                tail_entity_idx = entities.index(entity_description)

            # triples.append((main_entity, "'s description is", entity_description))
            triples.append((head_entity_idx, "that is", tail_entity_idx))

        entity_relations = entity_knowledge[2]
        for entity_relation in entity_relations:
            relation, tail_entity = entity_relation
            relation = special_str_cleanout(relation)
            tail_entity = special_str_cleanout(tail_entity)
            if tail_entity not in entities:
                tail_entity_idx = len(entities)
                entities.append(tail_entity)
            else:
                tail_entity_idx = entities.index(tail_entity)

            triples.append((head_entity_idx, relation, tail_entity_idx))
    return main_entities, entities, triples


def simplify_graph(entities, triples):
    linear_nodes = []
    node_dict = {}
    new_triples = []
    for entity_idx, entity in enumerate(entities):

        if entity not in linear_nodes:
            head_idx = len(linear_nodes)
            node_dict[entity] = head_idx
            linear_nodes.append(entity)
        else:
            head_idx = node_dict[entity]

        for triple in triples:
            ori_head_idx, relation, ori_tail_idx = triple

            if ori_head_idx == entity_idx:
                if relation not in linear_nodes:
                    relation_idx = len(linear_nodes)
                    node_dict[relation] = relation_idx
                    linear_nodes.append(relation)
                else:
                    relation_idx = node_dict[relation]

                ori_tail_entity = entities[ori_tail_idx]
                if ori_tail_entity not in linear_nodes:
                    final_tail_entity_idx = len(linear_nodes)
                    node_dict[ori_tail_entity] = final_tail_entity_idx
                    linear_nodes.append(ori_tail_entity)
                else:
                    final_tail_entity_idx = node_dict[ori_tail_entity]

                new_triples.append((head_idx, relation_idx))
                new_triples.append((relation_idx, final_tail_entity_idx))

    for entity in entities:
        assert entity in linear_nodes

    return linear_nodes, new_triples



def example_cleanout(data_item, kg_knowledge,
                     node_remain_degree_lower_bound=2,
                     text_overlap_score_lower_bound=0.5,
                     lower_text_overlap_score=False,
                     tokenized_the_sent_and_entities=False):
    """

    :param data_item: dict('id', 'url', 'title', 'title_kb_id', 'text', 'kblinks', 'score')
    :param kg_knowledge: dict(kblink: list(entity_name: str, description: str, relations: list(pair(relation, tail_entity)))
    :param node_remain_degree_lower_bound
    :param text_overlap_score_lower_bound
    :param lower_text_overlap_score
    :param tokenized_the_sent_and_entities
    :return:
    """

    data_id = data_item['id']
    title = data_item['title']
    title_kb_id = data_item['title_kb_id']
    target_sentence = data_item['text']
    target_sentence = [special_str_cleanout(word) for word in target_sentence]
    target_sentence = " ".join(target_sentence)
    score = data_item['score']
    kblinks = data_item['kblinks']

    metadata = []
    if title_kb_id not in kblinks:
        if tokenized_the_sent_and_entities:
            title = " ".join(tokenize(title))
        title = special_str_cleanout(title)
        title = "The data title is: {}".format(title)
        metadata.append(title)

    main_entities, init_entities, init_triples = convert_data_to_triples(kblinks=kblinks, kg_knowledge=kg_knowledge)

    entities, final_removed_entity_ids = triple_cleanout(main_entities=main_entities, entities=init_entities,
                                                         triples=init_triples, target_text=target_sentence,
                                                         node_remain_degree_lower_bound=node_remain_degree_lower_bound,
                                                         text_overlap_score_lower_bound=text_overlap_score_lower_bound,
                                                         lower_text_overlap_score=lower_text_overlap_score)

    # align the clean out entities and triple
    old_new_entity_idx_map = dict()
    for new_entity_idx, entity in enumerate(entities):
        old_entity_idx = init_entities.index(entity)
        assert old_entity_idx not in old_new_entity_idx_map, "{} : {}".format(old_entity_idx, old_new_entity_idx_map)

        old_new_entity_idx_map[old_entity_idx] = new_entity_idx

    if tokenized_the_sent_and_entities:
        entities = [" ".join(tokenize(entity)) for entity in entities]

    triples = []
    str_triples = []
    for triple in init_triples:
        head_entity_idx, relation, tail_entity_idx = triple

        if head_entity_idx in final_removed_entity_ids or tail_entity_idx in final_removed_entity_ids:
            continue
        new_head_entity_idx = old_new_entity_idx_map[head_entity_idx]
        new_tail_entity_idx = old_new_entity_idx_map[tail_entity_idx]
        if tokenized_the_sent_and_entities:
            relation = " ".join(tokenize(relation))
        triples.append((new_head_entity_idx, relation, new_tail_entity_idx))
        str_triples.append((entities[new_head_entity_idx], relation, entities[new_tail_entity_idx]))


    # print("entities", len(entities), entities)
    # print("triples", len(triples), triples)
    # print("str_triples", len(str_triples), str_triples)
    linear_nodes, new_triples = simplify_graph(entities, triples)
    example = dict()
    example['id'] = data_id
    example['score'] = score
    example['metadata'] = metadata
    example['linear_node'] = linear_nodes
    example['triple'] = new_triples
    example['target_sents'] = [target_sentence]
    example['node'] = entities
    example['ori_triple'] = triples
    example['str_triple'] = str_triples
    example['kblinks'] = kblinks
    # assert False
    return example


def cleanout(dataset_file_src, kg_knowledge_src, output_src,
             node_remain_degree_lower_bound=2, text_overlap_score_lower_bound=0.5, lower_text_overlap_score=False,
             tokenized_the_sent_and_entities=False, min_n_triple=2, min_n_triple_entity_rate=0.2, min_match_score=0.0):
    """

    :param dataset_file_src:
    :param kg_knowledge_src:
    :param output_src:
    :param node_remain_degree_lower_bound:
    :param text_overlap_score_lower_bound:
    :param lower_text_overlap_score:
    :param tokenized_the_sent_and_entities
    :param min_n_triple
    :param min_n_triple_entity_rate
    :param min_match_score
    :return:
    """
    data = utils.read_json_file(dataset_file_src)
    kg_knowledge = utils.read_json_file(kg_knowledge_src)

    cleanout_examples = []
    n_nodes = 0
    n_enc_tokens = 0
    n_dec_tokens = 0
    n_skip = 0
    n_triples = 0

    max_n_nodes = 0
    max_n_enc_tokens = 0
    max_n_dec_tokens = 0
    max_n_triples = 0

    for data_item in tqdm(data):
        # print("data_item", type(data_item), data_item)
        # print("kg_knowledge", type(kg_knowledge))
        # assert False
        if data_item['score'] < min_match_score:
            n_skip += 1
            continue

        cleanout_example = example_cleanout(data_item=data_item, kg_knowledge=kg_knowledge,
                                            node_remain_degree_lower_bound=node_remain_degree_lower_bound,
                                            text_overlap_score_lower_bound=text_overlap_score_lower_bound,
                                            lower_text_overlap_score=lower_text_overlap_score,
                                            tokenized_the_sent_and_entities=tokenized_the_sent_and_entities)
        if len(cleanout_example['triple']) < min_n_triple:
            n_skip += 1
            continue

        if len(cleanout_example['triple']) / len(cleanout_example['node']) < min_n_triple_entity_rate:
            n_skip += 1
            continue

        n_nodes += len(cleanout_example['node'])
        n_enc_tokens += len(" ".join(cleanout_example['node']).split())
        n_dec_tokens += len(cleanout_example['target_sents'][0].split())
        n_triples += len(cleanout_example['triple'])

        cleanout_examples.append(cleanout_example)

    utils.write_to_json_file_by_line(cleanout_examples, output_src)
    n_total_example = len(data)
    n_cleanout_example = len(cleanout_examples)
    print("Finished, total {} examples, cleanout {} examples, "
          "avg_node: {}, avg_enc_tokens: {}, avg_dec_tokens: {}, avg_triples: {}, "
          "cleanout data has been saved at {}".format(n_total_example, n_cleanout_example,
                                                      n_nodes / n_cleanout_example,
                                                      n_enc_tokens / n_cleanout_example,
                                                      n_dec_tokens / n_cleanout_example,
                                                      n_triples / n_cleanout_example,
                                                      output_src))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='KGText Cleanout of argparse')
    parser.add_argument("--kg_dataset_file_src", type=str, default='../../original_datasets/kgtext/test.json')
    parser.add_argument("--kg_knowledge_src", type=str, default='../../original_datasets/kgtext/knowledge-full.json')
    parser.add_argument("--output_cleanout_file_src", type=str, default='../../cleanout_datasets/kgtext/test.json')
    parser.add_argument("--node_remain_degree_lower_bound", type=float, default=3)
    parser.add_argument("--text_overlap_score_lower_bound", type=float, default=0.3)
    parser.add_argument("--lower_text_overlap_score", action='store_true')
    parser.add_argument("--min_n_triple", type=float, default=0)
    parser.add_argument("--min_n_triple_entity_rate", type=float, default=1.2, help='the min rate n_triple/n_entity')
    parser.add_argument("--min_match_score", type=float, default=0.3)

    args = parser.parse_args()

    cleanout(dataset_file_src=args.kg_dataset_file_src, kg_knowledge_src=args.kg_knowledge_src,
             output_src=args.output_cleanout_file_src,
             node_remain_degree_lower_bound=args.node_remain_degree_lower_bound,
             text_overlap_score_lower_bound=args.text_overlap_score_lower_bound,
             lower_text_overlap_score=args.lower_text_overlap_score,
             min_n_triple=args.min_n_triple,
             min_n_triple_entity_rate=args.min_n_triple_entity_rate,
             min_match_score=args.min_match_score)

