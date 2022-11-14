# coding=utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from asdl.sql.sql_transition_system import SelectColumnAction, SelectTableAction
from asdl.transition_system import ApplyRuleAction, ReduceAction
from asdl.action_info import ActionInfo
from asdl.hypothesis import Hypothesis
from asdl.decode_hypothesis import DecodeHypothesis
from utils.batch import Batch
from model.decoder.onlstm import LSTM, ONLSTM
from model.model_utils import Registrable, MultiHeadAttention

@Registrable.register('decoder_tranx')
class SqlParser(nn.Module):
    def __init__(self, args, transition_system):
        super(SqlParser, self).__init__()
        self.args = args
        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # the last entry is the embedding for Reduce action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)
        # embedding table for ASDL fields in constructors
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)
        # embedding table for ASDL types
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        # input of decoder lstm
        input_dim = args.action_embed_size  # previous action
        # parent node
        cxt_num = 2 if args.sep_cxt else 1
        input_dim += args.gnn_hidden_size * cxt_num * int(not args.no_context_feeding)
        input_dim += args.action_embed_size * int(not args.no_parent_production_embed)
        input_dim += args.field_embed_size * int(not args.no_parent_field_embed)
        input_dim += args.type_embed_size * int(not args.no_parent_field_type_embed)
        input_dim += args.lstm_hidden_size * int(not args.no_parent_state)
        cell_constructor = ONLSTM if args.lstm == 'onlstm' else LSTM
        self.decoder_lstm = cell_constructor(input_dim, args.lstm_hidden_size, num_layers=args.lstm_num_layers,
            chunk_num=args.chunk_size, dropout=args.dropout, dropconnect=args.drop_connect)

        # transform column embedding to production embedding space
        self.column_lstm_input = nn.Linear(args.gnn_hidden_size, args.action_embed_size)
        # transform table embedding to production embedding space
        self.table_lstm_input = self.column_lstm_input # nn.Linear(args.gnn_hidden_size, args.action_embed_size)

        self.context_attn = MultiHeadAttention(args.gnn_hidden_size, args.lstm_hidden_size, args.gnn_hidden_size, args.gnn_hidden_size,
            num_heads=args.num_heads, feat_drop=args.dropout)
        if args.sep_cxt: # calculate seperate context vector for question and database schema
            self.schema_attn = MultiHeadAttention(args.gnn_hidden_size, args.lstm_hidden_size, args.gnn_hidden_size, args.gnn_hidden_size,
                num_heads=args.num_heads, feat_drop=args.dropout)

        # feature vector before ApplyRule or SelectColumn/Table
        self.att_vec_linear = nn.Sequential(nn.Linear(args.lstm_hidden_size + cxt_num * args.gnn_hidden_size, args.att_vec_size), nn.Tanh())

        self.apply_rule_affine = nn.Linear(args.att_vec_size, args.action_embed_size, bias=False)
        self.apply_rule = lambda x: F.linear(x, self.production_embed.weight) # re-use the action embedding matrix
        self.select_column = MultiHeadAttention(args.gnn_hidden_size, args.att_vec_size, args.gnn_hidden_size, args.gnn_hidden_size,
            num_heads=args.num_heads, feat_drop=args.dropout)
        self.select_table = MultiHeadAttention(args.gnn_hidden_size, args.att_vec_size, args.gnn_hidden_size, args.gnn_hidden_size,
            num_heads=args.num_heads, feat_drop=args.dropout)

    def score(self, encodings, mask, h0, batch):
        """ Training function
            @input:
                encodings: encoded representations and mask matrix from encoder
                    bsize x seqlen x gnn_hidden_size
                batch: see utils.batch, we use fields
                    batch.examples, batch.get_frontier_prod_idx(t),
                    batch.get_frontier_field_idx(t), batch.get_frontier_field_type_idx(t),
                    batch.max_action_num, example.tgt_action (ActionInfo)
            output:
                loss: sum of loss for each training batch
        """
        args = self.args
        zero_action_embed = encodings.new_zeros(args.action_embed_size)
        split_len = [batch.max_question_len, batch.max_table_len, batch.max_column_len]
        q, tab, col = encodings.split(split_len, dim=1)
        q_mask, tab_mask, col_mask = mask.split(split_len, dim=1)
        if args.sep_cxt:
            encodings, mask = q, q_mask
            schema, schema_mask = torch.cat([tab, col], dim=1), torch.cat([tab_mask, col_mask], dim=1)
            schema_context, _ = self.schema_attn(schema, h0, schema_mask)
        context, _ = self.context_attn(encodings, h0, mask)
        h0 = h0.unsqueeze(0).repeat(args.lstm_num_layers, 1, 1)
        # h0 = h0.new_zeros(h0.size()) # init decoder with 0-vector
        h_c = (h0, h0.new_zeros(h0.size()))
        action_probs = [[] for _ in range(encodings.size(0))]
        history_states = []
        for t in range(batch.max_action_num):
            # x: [prev_action_embed, [prev_context, parent_production, parent_field, parent_type, parent_state]]
            if t == 0:
                x = encodings.new_zeros(encodings.size(0), self.decoder_lstm.input_size)
                offset = args.action_embed_size
                if not args.no_context_feeding:
                    x[:, offset: offset + args.gnn_hidden_size] = context
                    offset += args.gnn_hidden_size
                    if args.sep_cxt:
                        x[:, offset: offset + args.gnn_hidden_size] = schema_context
                        offset += args.gnn_hidden_size
                offset += args.action_embed_size * int(not args.no_parent_production_embed)
                offset += args.field_embed_size * int(not args.no_parent_field_embed)
                if not args.no_parent_field_type_embed:
                    start_rule = torch.tensor([self.grammar.type2id[self.grammar.root_type]] * x.size(0), dtype=torch.long, device=x.device)
                    x[:, offset: offset + args.type_embed_size] = self.type_embed(start_rule)
            else:
                prev_action_embed = []
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_action):
                        prev_action = example.tgt_action[t - 1].action
                        if isinstance(prev_action, ApplyRuleAction):
                            prev_action_embed.append(self.production_embed.weight[self.grammar.prod2id[prev_action.production]])
                        elif isinstance(prev_action, ReduceAction):
                            prev_action_embed.append(self.production_embed.weight[len(self.grammar)])
                        elif isinstance(prev_action, SelectColumnAction):
                            # map schema item into prod embed space
                            col_embed = self.column_lstm_input(col[e_id, prev_action.column_id])
                            prev_action_embed.append(col_embed)
                        elif isinstance(prev_action, SelectTableAction):
                            tab_embed = self.table_lstm_input(tab[e_id, prev_action.table_id])
                            prev_action_embed.append(tab_embed)
                        else:
                            raise ValueError('Unrecognized previous action object!')
                    else:
                        prev_action_embed.append(zero_action_embed)
                inputs = [torch.stack(prev_action_embed)]
                if not args.no_context_feeding:
                    inputs.append(context)
                    if args.sep_cxt:
                        inputs.append(schema_context)
                if not args.no_parent_production_embed:
                    inputs.append(self.production_embed(batch.get_frontier_prod_idx(t)))
                if not args.no_parent_field_embed:
                    inputs.append(self.field_embed(batch.get_frontier_field_idx(t)))
                if not args.no_parent_field_type_embed:
                    inputs.append(self.type_embed(batch.get_frontier_field_type_idx(t)))
                if not args.no_parent_state:
                    actions_t = [e.tgt_action[t] if t < len(e.tgt_action) else None for e in batch.examples]
                    parent_state = torch.stack([history_states[p_t][e_id]
                                                 for e_id, p_t in
                                                 enumerate([a_t.parent_t if a_t else 0 for a_t in actions_t])])
                    inputs.append(parent_state)
                x = torch.cat(inputs, dim=-1)

            # advance decoder lstm and attention calculation
            out, (h_t, c_t) = self.decoder_lstm(x.unsqueeze(1), h_c, start=(t==0))
            out = out.squeeze(1)
            context, _ = self.context_attn(encodings, out, mask)
            if args.sep_cxt:
                schema_context, _ = self.schema_attn(schema, out, schema_mask)
                att_vec = self.att_vec_linear(torch.cat([out, context, schema_context], dim=-1)) # bsize x args.att_vec_size
            else:
                att_vec = self.att_vec_linear(torch.cat([out, context], dim=-1))

            # action logprobs
            apply_rule_logprob = F.log_softmax(self.apply_rule(self.apply_rule_affine(att_vec)), dim=-1) # bsize x prod_num
            _, select_tab_prob = self.select_table(tab, att_vec, tab_mask)
            select_tab_logprob = torch.log(select_tab_prob + 1e-32)
            _, select_col_prob = self.select_column(col, att_vec, col_mask)
            select_col_logprob = torch.log(select_col_prob + 1e-32)

            for e_id, example in enumerate(batch.examples):
                if t < len(example.tgt_action):
                    action_t = example.tgt_action[t].action
                    if isinstance(action_t, ApplyRuleAction):
                        logprob_t = apply_rule_logprob[e_id, self.grammar.prod2id[action_t.production]]
                        # print('Rule %s with prob %s' %(action_t.production, logprob_t.item()))
                    elif isinstance(action_t, ReduceAction):
                        logprob_t = apply_rule_logprob[e_id, len(self.grammar)]
                        # print('Rule %s with prob %s' % ('Reduce', logprob_t.item()))
                    elif isinstance(action_t, SelectColumnAction):
                        logprob_t = select_col_logprob[e_id, action_t.column_id]
                        # print('SelectColumn %s with prob %s' % (action_t.column_id, logprob_t.item()))
                    elif isinstance(action_t, SelectTableAction):
                        logprob_t = select_tab_logprob[e_id, action_t.table_id]
                        # print('SelectTable %s with prob %s' % (action_t.table_id, logprob_t.item()))
                    else:
                        raise ValueError('Unrecognized action object!')
                    action_probs[e_id].append(logprob_t)

            h_c = (h_t, c_t)
            history_states.append(h_t[-1])

        # loss is negative sum of all the action probabilities
        loss = - torch.stack([torch.stack(logprob_i).sum() for label_index,logprob_i in enumerate(action_probs)]).sum()
        return loss

    def parse(self, encodings, mask, h0, batch, beam_size=5):
        """ Parse one by one, batch size for each args in encodings is 1
        """
        args = self.args
        zero_action_embed = encodings.new_zeros(args.action_embed_size)
        assert encodings.size(0) == 1 and mask.size(0) == 1
        encodings, mask = encodings.repeat(beam_size, 1, 1), mask.repeat(beam_size, 1)
        split_len = [batch.max_question_len, batch.max_table_len, batch.max_column_len]
        q, tab, col = encodings.split(split_len, dim=1)
        q_mask, tab_mask, col_mask = mask.split(split_len, dim=1)
        if args.sep_cxt:
            encodings, mask = q, q_mask
            schema, schema_mask = torch.cat([tab, col], dim=1), torch.cat([tab_mask, col_mask], dim=1)
            schema_context, _ = self.schema_attn(schema[:1], h0, schema_mask[:1])
        context, _ = self.context_attn(encodings[:1], h0, mask[:1])
        h0 = h0.unsqueeze(0).repeat(args.lstm_num_layers, 1, 1)
        # h0 = h0.new_zeros(h0.size()) # init decoder with 0-vector
        h_c = (h0, h0.new_zeros(h0.size()))
        hypotheses = [DecodeHypothesis()]
        hyp_states, completed_hypotheses, t = [[]], [], 0
        while t < args.decode_max_step:
            hyp_num = len(hypotheses)
            cur_encodings, cur_mask = encodings[:hyp_num], mask[:hyp_num]
            if args.sep_cxt:
                cur_schema, cur_schema_mask = schema[:hyp_num], schema_mask[:hyp_num]
            cur_tab, cur_tab_mask, cur_col, cur_col_mask = tab[:hyp_num], tab_mask[:hyp_num], col[:hyp_num], col_mask[:hyp_num]
            # x: [prev_action_embed, parent_production_embed, parent_field_embed, parent_field_type_embed, parent_state]
            if t == 0:
                x = encodings.new_zeros(hyp_num, self.decoder_lstm.input_size)
                offset = args.action_embed_size
                if not args.no_context_feeding:
                    x[:, offset: offset + args.gnn_hidden_size] = context
                    offset += args.gnn_hidden_size
                    if args.sep_cxt:
                        x[:, offset: offset + args.gnn_hidden_size] = schema_context
                        offset += args.gnn_hidden_size
                offset += args.action_embed_size * int(not args.no_parent_production_embed)
                offset += args.field_embed_size * int(not args.no_parent_field_embed)
                if not args.no_parent_field_type_embed:
                    start_rule = torch.tensor([self.grammar.type2id[self.grammar.root_type]] * hyp_num, dtype=torch.long, device=x.device)
                    x[:, offset: offset + args.type_embed_size] = self.type_embed(start_rule)
            else:
                prev_action_embed = []
                for e_id, hyp in enumerate(hypotheses):
                    prev_action = hyp.actions[-1]
                    if isinstance(prev_action, ApplyRuleAction):
                        prev_action_embed.append(self.production_embed.weight[self.grammar.prod2id[prev_action.production]])
                    elif isinstance(prev_action, ReduceAction):
                        prev_action_embed.append(self.production_embed.weight[len(self.grammar)])
                    elif isinstance(prev_action, SelectColumnAction): # need to first map schema item into prod embed space
                        col_embed = self.column_lstm_input(cur_col[e_id, prev_action.column_id])
                        prev_action_embed.append(col_embed)
                    elif isinstance(prev_action, SelectTableAction):
                        tab_embed = self.table_lstm_input(cur_tab[e_id, prev_action.table_id])
                        prev_action_embed.append(tab_embed)
                    else:
                        raise ValueError('Unrecognized previous action object!')
                inputs = [torch.stack(prev_action_embed)]
                if not args.no_context_feeding:
                    inputs.append(context)
                    if args.sep_cxt:
                        inputs.append(schema_context)
                if not args.no_parent_production_embed:
                    frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                    frontier_prod_embeds = self.production_embed(
                        torch.tensor([self.grammar.prod2id[prod] for prod in frontier_prods], dtype=torch.long, device=encodings.device))
                    inputs.append(frontier_prod_embeds)
                if not args.no_parent_field_embed:
                    frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                    frontier_field_embeds = self.field_embed(
                        torch.tensor([self.grammar.field2id[field] for field in frontier_fields], dtype=torch.long, device=encodings.device))
                    inputs.append(frontier_field_embeds)
                if not args.no_parent_field_type_embed:
                    frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                    frontier_field_type_embeds = self.type_embed(
                        torch.tensor([self.grammar.type2id[tp] for tp in frontier_field_types], dtype=torch.long, device=encodings.device))
                    inputs.append(frontier_field_type_embeds)
                if not args.no_parent_state:
                    p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                    parent_states = torch.stack([hyp_states[hyp_id][p_t] for hyp_id, p_t in enumerate(p_ts)])
                    inputs.append(parent_states)
                x = torch.cat(inputs, dim=-1) # hyp_num x input_size

            # advance decoder lstm and attention calculation
            out, (h_t, c_t) = self.decoder_lstm(x.unsqueeze(1), h_c)
            out = out.squeeze(1)
            context, _ = self.context_attn(cur_encodings, out, cur_mask)
            if args.sep_cxt:
                schema_context, _ = self.schema_attn(cur_schema, out, cur_schema_mask)
                att_vec = self.att_vec_linear(torch.cat([out, context, schema_context], dim=-1)) # hyp_num x args.att_vec_size
            else:
                att_vec = self.att_vec_linear(torch.cat([out, context], dim=-1))

            # action logprobs
            apply_rule_logprob = F.log_softmax(self.apply_rule(self.apply_rule_affine(att_vec)), dim=-1) # remaining_sents*beam x prod_num
            _, select_tab_prob = self.select_table(cur_tab, att_vec, cur_tab_mask)
            select_tab_logprob = torch.log(select_tab_prob + 1e-32)
            _, select_col_prob = self.select_column(cur_col, att_vec, cur_col_mask)
            select_col_logprob = torch.log(select_col_prob + 1e-32)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(hypotheses):
                action_types = self.transition_system.get_valid_continuation_types(hyp)
                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_logprob[hyp_id, prod_id]
                            new_hyp_score = hyp.score + prod_score
                            meta_entry = {'action_type': 'apply_rule', 'prod_id': prod_id,
                                          'score': prod_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_logprob[hyp_id, len(self.grammar)]
                        new_hyp_score = hyp.score + action_score
                        meta_entry = {'action_type': 'apply_rule', 'prod_id': len(self.grammar),
                                      'score': action_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                    elif action_type == SelectColumnAction:
                        col_num = cur_col_mask[hyp_id].int().sum().item()
                        for col_id in range(col_num):
                            col_sel_score = select_col_logprob[hyp_id, col_id]
                            new_hyp_score = hyp.score + col_sel_score
                            meta_entry = {'action_type': 'sel_col', 'col_id': col_id,
                                          'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                    elif action_type == SelectTableAction:
                        tab_num = cur_tab_mask[hyp_id].int().sum().item()
                        for tab_id in range(tab_num):
                            tab_sel_score = select_tab_logprob[hyp_id, tab_id]
                            new_hyp_score = hyp.score + tab_sel_score
                            meta_entry = {'action_type': 'sel_tab', 'tab_id': tab_id,
                                          'score': tab_sel_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                    else:
                        raise ValueError('Unrecognized action type while decoding!')
            if not new_hyp_meta: break

            # rank and pick
            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta])
            top_new_hyp_scores, meta_ids = new_hyp_scores.topk(min(beam_size, new_hyp_scores.size(0)))

            live_hyp_ids, new_hypotheses = [], []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.tolist(), meta_ids.tolist()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = hypotheses[prev_hyp_id]
                action_type_str = hyp_meta_entry['action_type']
                if action_type_str == 'apply_rule':
                    prod_id = hyp_meta_entry['prod_id']
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    else:
                        action = ReduceAction()
                elif action_type_str == 'sel_col':
                    action = SelectColumnAction(hyp_meta_entry['col_id'])
                elif action_type_str == 'sel_tab':
                    action = SelectTableAction(hyp_meta_entry['tab_id'])
                else:
                    raise ValueError('Unrecognized action type str!')

                action_info.action = action
                action_info.t = t
                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                if not args.no_parent_state:
                    hyp_states = [hyp_states[i] + [h_t[-1, i]] for i in live_hyp_ids]
                h_c = (h_t[:, live_hyp_ids], c_t[:, live_hyp_ids])
                if not args.no_context_feeding:
                    context = context[live_hyp_ids]
                    if args.sep_cxt:
                        schema_context = schema_context[live_hyp_ids]
                hypotheses = new_hypotheses
                t += 1
            else:
                break

            # for idx, hyp in enumerate(new_hypotheses):
                # print('Idx %d' % (idx))
                # print('Tree:', hyp.tree.to_string())
                # print('Action:', hyp.action_infos[-1])
                # print('============================================')
            # input('********************Next parse step ....************************')

        if len(completed_hypotheses) == 0: # no completed sql
            completed_hypotheses.append(DecodeHypothesis())
        else:
            completed_hypotheses.sort(key=lambda hyp: - hyp.score) # / hyp.tree.size
        return completed_hypotheses
