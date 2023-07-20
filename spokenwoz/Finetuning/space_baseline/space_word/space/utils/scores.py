from space.utils.decorators import ignore_nodes


def jaccard_dis_sim(x, y):
    """
    Jaccard Distance Similarity
    """
    res = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    if union_cardinality:
        return res / float(union_cardinality), 1
    else:
        return 0., 0


def clean_domain_frame(frame):
    cleaned_frame = {}
    for act, slot_values in frame.items():
        cleaned_act = act.strip().lower()
        cleaned_frame[cleaned_act] = {}
        for slot, values in slot_values.items():
            cleaned_slot = slot.strip().lower()
            cleaned_values = [str(value['value']).strip().lower() for value in values]
            cleaned_frame[cleaned_act][cleaned_slot] = cleaned_values
    return cleaned_frame


def clean_frame(frame):
    cleaned_frame = {}
    for domain, domain_frame in frame.items():
        cleaned_frame[domain.strip().lower()] = clean_domain_frame(frame=domain_frame)
    return cleaned_frame


def construct_domain_frame_graph(frame):
    act_nodes, slot_nodes, value_nodes = [], [], []  # 1-gram list (nodes)
    act_slot_edges, slot_value_edges = [], []  # 2-gram list (edges)
    act_slot_value_paths = []  # 3-gram list (paths)

    # construct
    act_nodes.extend(list(frame.keys()))
    for act, slot_values in frame.items():
        slot_nodes.extend(list(slot_values.keys()))
        values_list = list(slot_values.values())
        for values in values_list:
            value_nodes.extend(values)
        for slot, values in slot_values.items():
            act_slot_edges.append(f'{act}-{slot}')
            for value in values:
                slot_value_edges.append(f'{slot}-{value}')
                act_slot_value_paths.append(f'{act}-{slot}-{value}')

    return act_nodes, slot_nodes, value_nodes, act_slot_edges, slot_value_edges, act_slot_value_paths


@ignore_nodes(node_names=['DEFAULT_DOMAIN', 'DEFAULT_INTENT'])
def construct_frame_graph(frame):
    assert frame
    domain_nodes, act_nodes, slot_nodes, value_nodes = [], [], [], []  # 1-gram list (nodes)
    domain_act_edges, act_slot_edges, slot_value_edges = [], [], []  # 2-gram list (edges)
    domain_act_slot_paths, act_slot_value_paths = [], []  # 3-gram list (paths)
    domain_act_slot_value_paths = []  # 4-gram list (paths)

    domain_nodes.extend(list(frame.keys()))
    for domain, domain_frame in frame.items():
        single_act_nodes, single_slot_nodes, single_value_nodes, single_act_slot_edges, single_slot_value_edges, \
        single_act_slot_value_paths = construct_domain_frame_graph(frame=domain_frame)

        act_nodes.extend(single_act_nodes)
        slot_nodes.extend(single_slot_nodes)
        value_nodes.extend(single_value_nodes)
        act_slot_edges.extend(single_act_slot_edges)
        slot_value_edges.extend(single_slot_value_edges)
        act_slot_value_paths.extend(single_act_slot_value_paths)

        domain_act_edges.extend([f'{domain}-{act}' for act in single_act_nodes])
        domain_act_slot_paths.extend([f'{domain}-{act_slot}' for act_slot in single_act_slot_edges])
        domain_act_slot_value_paths.extend([f'{domain}-{act_slot_value}'
                                            for act_slot_value in single_act_slot_value_paths])

    return domain_nodes, act_nodes, slot_nodes, value_nodes, domain_act_edges, act_slot_edges, slot_value_edges, \
           domain_act_slot_paths, act_slot_value_paths, domain_act_slot_value_paths


def tree_edit_score(frame1, frame2):
    # deal with empty frame
    if not (frame1 and frame2):
        return 0.

    # clean frame
    frame1 = clean_frame(frame=frame1)
    frame2 = clean_frame(frame=frame2)
    if frame1 == frame2:
        return 1.

    # construct frame graph
    domain_nodes1, act_nodes1, slot_nodes1, value_nodes1, domain_act_edges1, act_slot_edges1, slot_value_edges1, \
    domain_act_slot_paths1, act_slot_value_paths1, domain_act_slot_value_paths1 = \
        construct_frame_graph(frame=frame1)

    domain_nodes2, act_nodes2, slot_nodes2, value_nodes2, domain_act_edges2, act_slot_edges2, slot_value_edges2, \
    domain_act_slot_paths2, act_slot_value_paths2, domain_act_slot_value_paths2 = \
        construct_frame_graph(frame=frame2)

    # compute individual score
    domain_score = jaccard_dis_sim(domain_nodes1, domain_nodes2)
    act_score = jaccard_dis_sim(act_nodes1, act_nodes2)
    slot_score = jaccard_dis_sim(slot_nodes1, slot_nodes2)
    value_score = jaccard_dis_sim(value_nodes1, value_nodes2)
    domain_act_score = jaccard_dis_sim(domain_act_edges1, domain_act_edges2)
    act_slot_score = jaccard_dis_sim(act_slot_edges1, act_slot_edges2)
    slot_value_score = jaccard_dis_sim(slot_value_edges1, slot_value_edges2)
    domain_act_slot_score = jaccard_dis_sim(domain_act_slot_paths1, domain_act_slot_paths2)
    act_slot_value_score = jaccard_dis_sim(act_slot_value_paths1, act_slot_value_paths2)
    domain_act_slot_value_score = jaccard_dis_sim(domain_act_slot_value_paths1, domain_act_slot_value_paths2)

    # compute combined score
    score, num_score = 0., 0
    for single_score in (domain_score, act_score, slot_score, value_score, domain_act_score, act_slot_score,
                         slot_value_score, domain_act_slot_score, act_slot_value_score,
                         domain_act_slot_value_score):
        score += single_score[0]
        num_score += single_score[1]
    score = score / num_score

    return score


if __name__ == '__main__':
    # Test 1: construct frame graph
    frame = {"hotel": {"DEFAULT_INTENT": {"name": [{"value": "the good shop"}]}, "bye": {}}}
    frame = clean_frame(frame)
    frame_graph = construct_frame_graph(frame=frame)
    print(frame_graph)
