#coding=utf8
import math, dgl, torch
import numpy as np
from utils.constants import MAX_RELATIVE_DIST
from utils.graph_example import GraphExample

# mapping special column * as an ordinary column
special_column_mapping_dict = {
    'question-*-generic': 'question-column-nomatch',
    '*-question-generic': 'column-question-nomatch',
    'table-*-generic': 'table-column-has',
    '*-table-generic': 'column-table-has',
    '*-column-generic': 'column-column-generic',
    'column-*-generic': 'column-column-generic',
    '*-*-identity': 'column-column-identity'
}
nonlocal_relations = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
    '*-column-generic', 'column-*-generic', '*-*-identity', '*-table-generic',
    'question-question-identity', 'table-table-identity', 'column-column-identity'] + [
    'question-question-dist' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1) if i not in [-1, 0, 1]
]

class GraphProcessor():

    def process_rgatsql(self, ex: dict, db: dict, relation: list):
        graph = GraphExample()
        num_nodes = int(math.sqrt(len(relation)))
        local_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            for idx, r in enumerate(relation) if r not in nonlocal_relations]
        nonlocal_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            for idx, r in enumerate(relation) if r in nonlocal_relations]
        global_edges = local_edges + nonlocal_edges
        src_ids, dst_ids = list(map(lambda r: r[0], global_edges)), list(map(lambda r: r[1], global_edges))
        graph.global_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.global_edges = global_edges
        src_ids, dst_ids = list(map(lambda r: r[0], local_edges)), list(map(lambda r: r[1], local_edges))
        graph.local_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.local_edges = local_edges
        # graph pruning for nodes
        q_num = len(ex['processed_question_toks'])
        s_num = num_nodes - q_num
        graph.question_mask = [1] * q_num + [0] * s_num
        graph.schema_mask = [0] * q_num + [1] * s_num
        graph.gp = dgl.heterograph({
            ('question', 'to', 'schema'): (list(range(q_num)) * s_num,
            [i for i in range(s_num) for _ in range(q_num)])
            }, num_nodes_dict={'question': q_num, 'schema': s_num}, idtype=torch.int32
        )
        t_num = len(db['processed_table_toks'])
        def check_node(i):
            if i < t_num and i in ex['used_tables']:
                return 1.0
            elif i >= t_num and i - t_num in ex['used_columns']:
                return 1.0
            else: return 0.0
        graph.node_label = list(map(check_node, range(s_num)))
        ex['graph'] = graph
        return ex

    def process_lgesql(self, ex: dict, db: dict, relation: list):
        ex = self.process_rgatsql(ex, db, relation)
        graph = ex['graph']
        lg = graph.local_g.line_graph(backtracking=False)
        # prevent information propagate through matching edges
        match_ids = [idx for idx, r in enumerate(graph.global_edges) if 'match' in r[2]]
        src, dst, eids = lg.edges(form='all', order='eid')
        eids = [e for u, v, e in zip(src.tolist(), dst.tolist(), eids.tolist()) if not (u in match_ids and v in match_ids)]
        graph.lg = lg.edge_subgraph(eids, preserve_nodes=True).remove_self_loop().add_self_loop()
        ex['graph'] = graph
        return ex

    def process_graph_utils(self, ex: dict, db: dict, method: str = 'rgatsql'):
        """ Example should be preprocessed by self.pipeline
        """
        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')
        relation = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0)
        relation = relation.flatten().tolist()
        if method == 'rgatsql':
            ex = self.process_rgatsql(ex, db, relation)
        elif method == 'lgesql':
            ex = self.process_lgesql(ex, db, relation)
        return ex
