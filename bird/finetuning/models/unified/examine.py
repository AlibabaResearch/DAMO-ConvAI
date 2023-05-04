import pickle
import dgl
import pdb
from collections import defaultdict


graph = pickle.load(open('data/graph_pedia_total.bin', 'rb'))
# db.set_trace()


def compute_relations(graph_pedia):
    relation_count = defaultdict()
    for idx, graph in graph_pedia.items():
        relation_lst = graph['edges']
        for e in relation_lst:
            r = e[-1]
            if r in relation_count:
                relation_count[r] += 1
            else:
                relation_count[r] = 1
    return relation_count



count = compute_relations(graph)
pdb.set_trace()

print(' ')