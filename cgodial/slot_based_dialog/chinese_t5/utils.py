import copy
import os, random, csv, time, logging, json, re
import pprint
from collections import Counter
import numpy as np
from itertools import chain
from copy import deepcopy
from collections import OrderedDict
import torch

import ontology
from config import global_config as cfg


class _ReaderBase(object):
    
    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None
        self.set_stats = {}
    
    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(copy.deepcopy(dial))
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))
    
    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch) % len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches
    
    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch
    
    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key == 'dial_id':
                    continue
                # if key == 'pointer' and self.db is not None:
                #     turn_domain = turn['turn_domain'][-1]
                #     value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs
    
    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    # if key == 'pointer' and self.db is not None:
                    #     turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                    #     value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs
    
    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        
        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)
        
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials
        
        return dial
    
    def get_batches(self, set_name):
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        # if cfg.low_resource and set_name == 'train':
        #     # dial = random.sample(dial, int(len(dial)*0.01))
        #     dial = random.sample(dial, 100)
        #     logging.info('Low Resource setting, finetuning size: {}'.format(len(dial)))
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        
        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        
        
        for k in turn_bucket:
            if set_name not in ['test', 'dev'] and (k == 1 or k >= 17):
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
            
        log_str += 'total batch num: %d\n' % len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches
        
        # log stats
        # logging.info(log_str)
        # cfg.num_training_steps = num_training_steps * cfg.epoch_num
        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials
        
        if set_name == 'train':
            random.shuffle(all_batches)
        return all_batches
        
    
    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch
    
    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)
    
    def save_result(self, write_mode, results, field, write_title=False):
        with open(cfg.result_path, write_mode) as rf:
            if write_title:
                rf.write(write_title + '\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None
    
    def save_result_report(self, results):
        # if 'joint_goal' in results[0]:
        #     with open(cfg.result_path[:-4] + '_report_dst.txt', 'w') as rf:
        #         rf.write('joint goal\tslot_acc\tslot_f1\tact_f1\n')
        #         for res in results:
        #             a,b,c,d = res['joint_goal'], res['slot_acc'], res['slot_f1'], res['act_f1']
        #             rf.write('%2.1f\t%2.1f\t%2.1f\t%2.1f\n'%(a,b,c,d))
        # elif 'joint_goal_delex' in results[0]:
        #     with open(cfg.result_path[:-4] + '_report_bsdx.txt', 'w') as rf:
        #         rf.write('joint goal\tslot_acc\tslot_f1\tact_f1\n')
        #         for res in results:
        #             a,b,c,d = res['joint_goal_delex'], res['slot_acc_delex'], res['slot_f1_delex'], res['act_f1']
        #             rf.write('%2.1f\t%2.1f\t%2.1f\t%2.1f\n'%(a,b,c,d))
        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv' % cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s' % str(cfg.beam_width)
            if cfg.beam_diverse_param > 0:
                setting += ', penalty=%s' % str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s' % str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s' % str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path, 'true_bspn': cfg.use_true_curr_bspn, 'true_aspn': cfg.use_true_curr_aspn,
               'decode': cfg.aspn_decode_mode, 'param': setting, 'nbest': cfg.nbest,
               'selection_sheme': cfg.act_selection_scheme,
               'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'],
               'act_f1': results[0]['act_f1'],
               'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class RisaWOZT5Reader(_ReaderBase):
    def __init__(self, tokenizer):
        super().__init__()

        with open(cfg.slot_value_set_path) as f:
            self.slot_value_set = json.load(f)

        self.tokenizer = tokenizer

        if cfg.mode == 'train':
            self.add_sepcial_tokens()

        self.vocab_size = self.tokenizer.vocab_size

        with open(cfg.test_list) as f:
            self.test_list = [n.lower() for n in json.load(f)]

        with open(cfg.val_list) as f:
            self.val_list = [n.lower() for n in json.load(f)]

        with open(cfg.train_list) as f:
            self.train_list = [n.lower() for n in json.load(f)]

        self.unk_id = self.tokenizer.encode('[UNK]')[1]
        self.pad_id = self.tokenizer.encode('[PAD]')[1]
        self.sos_u_id = self.tokenizer.encode('<sos_u>')[1]
        self.eos_u_id = self.tokenizer.encode('<eos_u>')[1]
        self.sos_b_id = self.tokenizer.encode('<sos_b>')[1]
        self.eos_b_id = self.tokenizer.encode('<eos_b>')[1]
        self.sos_db_id = self.tokenizer.encode('<sos_db>')[1]
        self.eos_db_id = self.tokenizer.encode('<eos_db>')[1]
        self.sos_a_id = self.tokenizer.encode('<sos_a>')[1]
        self.eos_a_id = self.tokenizer.encode('<eos_a>')[1]
        self.sos_r_id = self.tokenizer.encode('<sos_r>')[1]
        self.eos_r_id = self.tokenizer.encode('<eos_r>')[1]

        self.special_token_ids = self.tokenizer.convert_tokens_to_ids(ontology.special_tokens)

        self._load_data()

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        for word in ontology.DOMAIN_MAP_ch2en:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in ontology.ALL_DA:
            word = '[' + word + ']'
            special_tokens.append(word)
        for db_id in range(4):
            word = '[db_%d]' % db_id
            special_tokens.append(word)
        # for word in ontology.all_slots:
        # to be determine whether slot should be [slot]
        # if slot, tokenizer having trouble decoding.
        # special_tokens.append(word)
        for word_list in self.slot_value_set.values():
            for w in word_list:
                if '[v_%s]' % ontology.normalize_slot(w) not in special_tokens:
                    special_tokens.append('[v_%s]' % ontology.normalize_slot(w))
    
        special_tokens.extend(ontology.special_tokens)
        
        # delete
        special_tokens.append('<None>')
    
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        print(special_tokens_dict)
        logging.info(str(special_tokens_dict))
        self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to t5 tokenizer....')
        print('Added special tokens to t5 tokenizer....')

    def _load_data(self):
        self.data = json.loads(open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())

        # encoded_file = os.path.join(cfg.data_path, 'gpt_encode_data.json')
        encoded_file = 't5_encode_data.json'
        if os.path.exists(encoded_file):
            logging.info('Reading encoded data from {}'.format(encoded_file))
            with open(encoded_file) as f:
                encoded_data = json.load(f)
            self.train = encoded_data['train']
            self.dev = encoded_data['dev']
            self.test = encoded_data['test']

        else:
            self.train, self.dev, self.test = [] , [], []
            print(len(self.val_list))
            print(len(self.test_list))
            print(len(self.train_list))
            print(len(self.data))
            
            for fn, dial in self.data.items():
                fn = fn.lower()
                if fn in self.val_list:
                    self.dev.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log')))
                    self.dev.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr0')))
                    self.dev.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr1')))
                    self.dev.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr2')))
                elif fn in self.test_list:
                    self.test.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log')))
                    self.test.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr0')))
                    self.test.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr1')))
                    self.test.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr2')))
                elif fn in self.train_list:
                    self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log')))
                    # self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr0')))
                    # self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr1')))
                    # self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr2')))
                else:
                    print(fn)
            encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
            json.dump(encoded_data, open(encoded_file, 'w'), ensure_ascii=False)
        
        logging.info('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))
        print('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test))
)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
        
    def parse_bspn(self, bspn):
        if not bspn: return {}
        # [domain] slot1=value1|slot2=value2 [domain] slot1=value1
        bspn_spli = re.split(ontology.DOMAIN_RE, bspn)
        bs = {}
        cur_domain = None
        for w in bspn_spli:
            if not w: continue
            if w in ontology.DOMAIN_TOK and w not in bs:
                bs[w] = {}
                cur_domain = w
            elif '=' in w and cur_domain:
                for sv in w.split('|'):
                    try:
                        s, v = sv.split('=')
                        s = s.replace(' ', '')
                        v = v.replace(' ', '')
                        bs[cur_domain][s] = v
                    except:
                        pass
        return bs
    
    
    def check_update(self, prev_constraint, cur_constraint):
        # print('prev_constraint:', prev_constraint)
        # print('cur_constraint:', cur_constraint)
        prev_constraint_dic = self.parse_bspn(prev_constraint)
        cur_constraint_dic = self.parse_bspn(cur_constraint)
        
        output_s = ""
        for d in cur_constraint_dic:
            output = []
            for s, v in cur_constraint_dic[d].items():
                if d not in prev_constraint_dic or s not in prev_constraint_dic[d] or \
                    prev_constraint_dic[d][s] != v:
                    output.append("%s=%s"%(s,v))
            # delete
            if d in prev_constraint_dic:
                for s in prev_constraint_dic[d]:
                    if s not in cur_constraint_dic[d]:
                        output.append("%s=<None>"%s)
            if output:
                output_s += "%s " %d + '|'.join(output)
                
        for d in prev_constraint_dic:
            output = []
            if d not in cur_constraint_dic:
                for s,v in prev_constraint_dic[d].items():
                    output.append("%s=<None>" % s)
            
            if output:
                output_s += "%s " %d + '|'.join(output)

        return output_s
        

    def _get_encoded_data(self, fn, dial, log_key):
        encoded_dial = []
        prev_constraint = ""
        for idx, t in enumerate(dial[log_key]):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn
            enc['log_key'] = log_key
            # if fn != 'movie_tv_goal_2-40_v2###10199': continue
            if log_key != 'log':
                enc['dial_id'] = fn + '-' + log_key
            #enc['user'] = self.vocab.tokenizer.encode(t['user']) + self.vocab.tokenizer.encode(['<eos_u>'])
            
            enc['user'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_u> ' + t['user'] + ' <eos_u>'))

            enc['bspn'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_b> ' + t['constraint'] + ' <eos_b>'))

            update_bspn = self.check_update(prev_constraint, t['constraint'])

            enc['update_bspn'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_b> ' + update_bspn + ' <eos_b>'))

            enc['db'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_db> ' + t['db'] + ' <eos_db>'))

            enc['aspn'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_a> ' + t['sys_act'] + ' <eos_a>'))

            enc['resp'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_r> ' + t['resp'] + ' <eos_r>'))

            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']
            enc['turn_type'] = t['type']
            enc['turn_succ'] = t['turn_succ']
            
            encoded_dial.append(enc)

            prev_constraint = t['constraint']
        return encoded_dial


    def convert_batch(self, batch, prev, first_turn=False):
        """
        user: dialogue history ['user']
        input: previous dialogue state + dialogue history
        DB state: ['input_pointer']
        output1: dialogue state state_update:update ['update_bspn'] or current dialogue state ['bspn']
        output2: dialogue response ['resp']
        """
        inputs = {}
        pad_token = self.pad_id
        
        batch_size = len(batch['user'])
        
        # input: previous dialogue state + dialogue history
        input_ids = []
        if first_turn:
            for i in range(batch_size):
                input_ids.append(batch['user'][i])
        else:
            for i in range(batch_size):
                input_ids.append(prev['bspn'][i] + prev['resp'][i] + batch['user'][i])
          
        input_ids, masks = self.padInput(input_ids, pad_token)
        
        # context with DB
        input_ids_plus = []
        if first_turn:
            for i in range(batch_size):
                input_ids_plus.append(batch['user'][i] + batch['db'][i])
        else:
            for i in range(batch_size):
                input_ids_plus.append(prev['bspn'][i] + prev['resp'][i] + batch['user'][i] + batch['db'][i])

        input_ids_plus, masks_plus = self.padInput(input_ids_plus, pad_token)
        
        # print('input_ids:')
        # for ii in range(batch_size):
        #     print(self.tokenizer.decode(input_ids[ii]))
        # print(masks)
        
        inputs["input_ids"] = torch.tensor(input_ids,dtype=torch.long)
        inputs["masks"] = torch.tensor(masks,dtype=torch.long)
        
        state_update, state_input = self.padOutput(batch['update_bspn'], pad_token)

        # print('state_update:')
        # for ii in range(batch_size):
        #     print(self.tokenizer.decode(state_update[ii].tolist()))
        #
        # print('state_input:')
        # for ii in range(batch_size):
        #     print(self.tokenizer.decode(state_input[ii].tolist()))

        inputs["input_ids_plus"] = torch.tensor(input_ids_plus, dtype=torch.long)
        inputs["masks_plus"] = torch.tensor(masks_plus, dtype=torch.long)

        # print('input_ids_plus:')
        # for ii in range(batch_size):
        #     print(self.tokenizer.decode(input_ids_plus[ii]))
        # print(masks_plus)
        
        respose_ids = []
        for i in range(batch_size):
            respose_ids.append(batch['aspn'][i]+batch['resp'][i])
        
        response, response_input = self.padOutput(respose_ids, pad_token)

        # print('response:')
        # for ii in range(batch_size):
        #     print(self.tokenizer.decode(response[ii]))
        #
        # print('response_input:')
        # for ii in range(batch_size):
        #     print(self.tokenizer.decode(response_input[ii]))
            
        # print(input())
        
        inputs["state_update"] = torch.tensor(state_update,dtype=torch.long) # batch_size, seq_len
        inputs["response"] = torch.tensor(response,dtype=torch.long)

        inputs["state_input"] = torch.tensor(state_input,dtype=torch.long)
        inputs["response_input"] = torch.tensor(response_input,dtype=torch.long)
        inputs["turn_domain"] = batch["turn_domain"]
        
        return inputs

    def padOutput(self, sequences, pad_token):
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        max_len = max(lengths)
        # output_ids = np.ones((num_samples, max_len)) * (-100) #-100 ignore by cross entropy
        output_ids = np.ones((num_samples, max_len)) * pad_token #-100 ignore by cross entropy

        decoder_inputs = np.ones((num_samples, max_len)) * pad_token
        for idx, s in enumerate(sequences):
            output_ids[idx, :lengths[idx]-1] = s[1:lengths[idx]]
            decoder_inputs[idx, :lengths[idx]-1] = s[:lengths[idx]-1]
        return output_ids, decoder_inputs

    def padInput(self, sequences, pad_token):
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        max_len = max(lengths)
        input_ids = np.ones((num_samples, max_len)) * pad_token
        masks = np.zeros((num_samples, max_len))

        for idx, s in enumerate(sequences):
            trunc = s[-max_len:]
            input_ids[idx, :lengths[idx]] = trunc
            masks[idx, :lengths[idx]] = 1
        return input_ids, masks

    def parse_bspan(self, bspn):
        # [domain] slot1=value1|slot2=value2 [domain] slot1=value1
        bspn_spli = re.split(ontology.DOMAIN_RE, bspn)
        bs = {}
        cur_domain = None
        for w in bspn_spli:
            if not w: continue
            if w in ontology.DOMAIN_TOK and w not in bs:
                bs[w] = {}
                cur_domain = w
            elif '=' in w and cur_domain:
                for sv in w.split('|'):
                    try:
                        s, v = sv.split('=')
                        bs[cur_domain][s] = v
                    except:
                        pass
        return bs
    
    def bs2tok(self, bs_dic):
        # [domain] slot1=value1|slot2=value2 [domain] slot1=value1
        res = []
        # print(bs_dic)
        for d in bs_dic:
            if len(bs_dic[d]):
                res.append('%s' % d)
                res.append('|'.join(['%s=%s '%(s,v) for s, v in bs_dic[d].items()]))
        string = ' '.join(res)
        # print('return:', ' '.join(self.tokenizer.tokenize(string)))
        # print()
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string))
    
    def update_bspn(self, prev_bspn, bspn_update):
        if prev_bspn is None:
            bspn_update_toks = [tok for tok in self.tokenizer.decode(bspn_update).split() if
                                tok not in ['[PAD]', '<eos_b>', '<sos_b>']]
            # print('return:', bspn_update_toks)
            # print()
            return [self.sos_b_id] + \
                   self.tokenizer.convert_tokens_to_ids(
                       self.tokenizer.tokenize(' '.join(bspn_update_toks))) + \
                   [self.eos_b_id]
            

        prev_bspn_toks = [tok for tok in self.tokenizer.decode(prev_bspn).split() if tok not in ['[PAD]', '<eos_b>', '<sos_b>']]
        bspn_update_toks = [tok for tok in self.tokenizer.decode(bspn_update).split() if tok not in ['[PAD]', '<eos_b>', '<sos_b>']]
        # print('prev_bspn_toks:', ' '.join(prev_bspn_toks))
        # print('bspn_update_toks: ', ' '.join(bspn_update_toks))
        
        
        if not bspn_update_toks:
            # print('return:', prev_bspn_toks)
            # print()
            return [self.sos_b_id] + \
                   self.tokenizer.convert_tokens_to_ids(
                       self.tokenizer.tokenize(' '.join(prev_bspn_toks))) \
                   + [self.eos_b_id]
        

        
        prev_bs = self.parse_bspn(' '.join(prev_bspn_toks))
        update_bs = self.parse_bspn(' '.join(bspn_update_toks))
        # print(prev_bs)
        # print(update_bs)
        
        for d in update_bs:
            if d not in prev_bs:
                prev_bs[d] = {}
            for s, v in update_bs[d].items():
                if v == '<None>' and s in prev_bs[d]:
                    del prev_bs[d][s]
                else:
                    prev_bs[d][s] = v
                    # if s == '车型' and '坐席' in prev_bs['[火车]']:
                    #     del prev_bs['[火车]']['坐席']

        res_tok = self.bs2tok(prev_bs)
        return [self.sos_b_id] + res_tok + [self.eos_b_id]
    
    def parse_resp_gen(self, generateds):
        aspns, resps = [], []
        for generated in generateds:
            eos_a_id = self.eos_a_id
            sos_r_id = self.sos_r_id
            eos_r_id = self.eos_r_id
        
            # eos_r may not exists if gpt2 generated repetitive words.
            if eos_r_id in generated:
                eos_r_idx = generated.index(eos_r_id)
            else:
                eos_r_idx = len(generated) - 1
                logging.info('eos_r not in generated: ' + self.tokenizer.decode(generated))
            # eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated)-1
        
            if eos_a_id in generated:
                eos_a_idx = generated.index(eos_a_id)
            elif sos_r_id in generated:
                eos_a_idx = generated.index(sos_r_id) - 1
            else:
                logging.info('eos_a not in generated: ' + self.tokenizer.decode(generated))
                eos_a_idx = len(generated) // 2
    
            aspn = [self.sos_a_id] + [w for w in generated[: eos_a_idx + 1]
                                      if w not in self.special_token_ids] + [self.eos_a_id]
            resp = [self.sos_r_id] + [w for w in generated[eos_a_idx + 1: eos_r_idx + 1]
                                      if w not in self.special_token_ids] + [self.eos_r_id]
            
            aspns.append(aspn)
            resps.append(resp)
        return aspns, resps
        

    def wrap_result(self, result_dict):
        results = []
        eos_syntax = ontology.eos_tokens
        sos_syntax = ontology.sos_tokens

        field = ['dial_id', 'turn_num', 'turn_succ',
                 'user',
                 'bspn', 'bspn_gen',
                 'aspn_gen', 'aspn',
                 'resp_gen', 'resp',
                 'turn_type', 'turn_domain'
                 ]

        for dial_id, turns in result_dict.items():
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                # act , resp_gen = self.parse_resp_gen(turn['resp_gen'].numpy().tolist())
                # turn['aspn_gen'] = act
                # turn['resp_gen'] = resp_gen
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])
                        v = " ".join(v)
                    else:
                        pass  # v = v
                    entry[key] = v
                
                results.append(entry)

        return results, field

    # def restore(self, resp, domain, constraint_dict, mat_ents):
    #     restored = resp

    #     restored = restored.replace('[value_reference]', '53022')
    #     restored = restored.replace('[value_car]', 'BMW')

    #     # restored.replace('[value_phone]', '830-430-6666')
    #     for d in domain:
    #         constraint = constraint_dict.get(d,None)
    #         if constraint:
    #             if 'stay' in constraint:
    #                 restored = restored.replace('[value_stay]', constraint['stay'])
    #             if 'day' in constraint:
    #                 restored = restored.replace('[value_day]', constraint['day'])
    #             if 'people' in constraint:
    #                 restored = restored.replace('[value_people]', constraint['people'])
    #             if 'time' in constraint:
    #                 restored = restored.replace('[value_time]', constraint['time'])
    #             if 'type' in constraint:
    #                 restored = restored.replace('[value_type]', constraint['type'])
    #             if d in mat_ents and len(mat_ents[d])==0:
    #                 for s in constraint:
    #                     if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
    #                         restored = restored.replace('[value_price]', constraint['pricerange'])
    #                     if s+']' in restored:
    #                         restored = restored.replace('[value_%s]'%s, constraint[s])

    #         if '[value_choice' in restored and mat_ents.get(d):
    #             restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
    #     if '[value_choice' in restored:
    #         restored = restored.replace('[value_choice]', '3')


    #     # restored.replace('[value_car]', 'BMW')


    #     try:
    #         ent = mat_ents.get(domain[-1], [])
    #         if ent:
    #             ent = ent[0]

    #             for t in restored.split():
    #                 if '[value' in t:
    #                     slot = t[7:-1]
    #                     if ent.get(slot):
    #                         if domain[-1] == 'hotel' and slot == 'price':
    #                             slot = 'pricerange'
    #                         restored = restored.replace(t, ent[slot])
    #                     elif slot == 'price':
    #                         if ent.get('pricerange'):
    #                             restored = restored.replace(t, ent['pricerange'])
    #                         else:
    #                             print(restored, domain)
    #     except:
    #         print(resp)
    #         print(restored)
    #         quit()


    #     restored = restored.replace('[value_phone]', '62781111')
    #     restored = restored.replace('[value_postcode]', 'CG9566')
    #     restored = restored.replace('[value_address]', 'Parkside, Cambridge')

    #     return restored

    def restore(self, resp, domain, constraint_dict):
        restored = resp
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')

        mat_ents = self.db.get_match_num(constraint_dict, True)
        self.delex_refs = ["w29zp27k","qjtixk8c","wbjgaot8","wjxw4vrv","sa63gzjd","i4afi8et","u595dz8a","8ttxct27","vcmkko1k","a5litxvz","2gy5ulll","gethuntl","i76goxin","mq7amf1m","isyr3hnc","69srbpnj","pmhz3tjo","5vrjsmse","ie05gdqs","wpa3iy8c","lnk1guuk","bbg39tvv","73mseuiq","6knjsqxy","znl8d0eg","4rz5lydp","r9xjc41b","d77jcgj2","sw8ac8gh",]
        ref =  random.choice(self.delex_refs)
        restored = restored.replace('[value_reference]', ref.upper())
        restored = restored.replace('[value_car]', 'BMW')

        # restored.replace('[value_phone]', '830-430-6666')
        for d in domain:
            constraint = constraint_dict.get(d,None)
            if constraint:
                if 'stay' in constraint:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                if 'day' in constraint:
                    restored = restored.replace('[value_day]', constraint['day'])
                if 'people' in constraint:
                    restored = restored.replace('[value_people]', constraint['people'])
                if 'time' in constraint:
                    restored = restored.replace('[value_time]', constraint['time'])
                if 'type' in constraint:
                    restored = restored.replace('[value_type]', constraint['type'])
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        # restored.replace('[value_car]', 'BMW')
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

        ent = mat_ents.get(domain[-1], [])
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    slot = t[7:-1]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)

            # handle normal 1 entity case
            ent = ent[0]
            for t in restored.split():
                if '[value' in t:
                    slot = t[7:-1]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        restored = restored.replace(t, rep)
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        restored = restored.replace(t, ent['pricerange'])
                        # else:
                        #     print(restored, domain)
        restored = restored.replace('[value_phone]', '07338019809')#taxi number need to get from api call, which is not available
        for t in restored.split():
            if '[value' in t:
                restored = restored.replace(t, 'UNKNOWN')

        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)
        return restored


    def relex(self, result_path, output_path):
        data = []

        with open(result_path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 10: # skip statistic ressults
                    namelist = row
                elif i > 10:
                    data.append(row)

        bspn_index = namelist.index("bspn_gen")
        resp_index = namelist.index("resp_gen")
        dspn_index = namelist.index("dspn_gen")

        row_list = []
        row_list.append(namelist)

        for row in data:
            bspn = row[bspn_index]
            resp = row[resp_index]
            dspn = [row[dspn_index].replace("[","").replace("]","")]
            if bspn == "" or resp == "":
                row_list.append(row)
            else:
                constraint_dict = self.bspan_to_constraint_dict(bspn)
                new_resp_gen = self.restore(resp, dspn, constraint_dict)

                row[resp_index] = new_resp_gen
                row_list.append(row)

                
                print("resp", resp)
                #print("cons_dict: ", cons_dict)
                #print("dspn: ", dspn)
                print("new_resp_gen: ", new_resp_gen)

        with open(output_path, "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(row_list)


if __name__ == '__main__':
    from transformers import (AdamW, BertTokenizer, WEIGHTS_NAME, CONFIG_NAME, get_linear_schedule_with_warmup)
    
    tokenizer = BertTokenizer.from_pretrained('./t5_chinese_small')
    reader = RisaWOZT5Reader(tokenizer)
    
    
    print(tokenizer.convert_tokens_to_ids('[汽车] 车 系 = c - trek 蔚 领<None>'.split()))