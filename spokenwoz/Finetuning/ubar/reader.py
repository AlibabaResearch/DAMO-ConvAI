import numpy as np
import os
import csv
import random
import logging
import json
import spacy
import utils
import ontology
from copy import deepcopy
from collections import OrderedDict
from db_ops import MultiWozDB
from torch.utils.data import Dataset, DataLoader

from config import global_config as cfg
# from config21 import global_config as cfg

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
            turn_bucket[turn_len].append(dial)
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
                if key=='dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
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
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        # print(dial)
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
        if cfg.low_resource and set_name == 'train':
            # dial = random.sample(dial, int(len(dial)*0.01))
            dial = random.sample(dial, 100)
            logging.info('Low Resource setting, finetuning size: {}'.format(len(dial)))
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        
        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            # if set_name != 'test' and k == 1 or k >= 17:
                # continue
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
                rf.write(write_title+'\n')
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
               'decode': cfg.aspn_decode_mode, 'param': setting, 'nbest': cfg.nbest, 'selection_sheme': cfg.act_selection_scheme,
               'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'], 'act_f1': results[0]['act_f1'],
               'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class MultiWozReader(_ReaderBase):
    def __init__(self, tokenizer):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')

        self.db = MultiWozDB(cfg.dbs)
        self.vocab_size = self._build_vocab()

        # self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path) # add special tokens later
        self.tokenizer = tokenizer
        if cfg.mode=='train':
            self.add_sepcial_tokens()

        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(
            open(cfg.slot_value_set_path, 'r').read())
        if cfg.multi_acts_training:
            self.multi_acts = json.loads(open(cfg.multi_acts_path, 'r').read())

        test_list = [l.strip().lower()
                     for l in open(cfg.test_list, 'r').readlines()]
        # print('test')
        # print(test_list)
        dev_list = [l.strip().lower()
                    for l in open(cfg.dev_list, 'r').readlines()]
        # print('dev')
        # print(dev_list)
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        # for domain expanse aka. Cross domain
        self.exp_files = {}
        # if 'all' not in cfg.exp_domains:
        #     for domain in cfg.exp_domains:
        #         fn_list = self.domain_files.get(domain)
        #         if not fn_list:
        #             raise ValueError(
        #                 '[%s] is an invalid experiment setting' % domain)
        #         for fn in fn_list:
        #             self.exp_files[fn.replace('.json', '')] = 1
        all_domains_list = list(self.domain_files.keys())
        if 'all' not in cfg.exp_domains:
            domains = self.get_exp_domains(cfg.exp_domains, all_domains_list)
            logging.info(domains)
            for domain in domains:
                fn_list = self.domain_files.get(domain)
                if not fn_list:
                    raise ValueError(
                        '[%s] is an invalid experiment setting' % domain)
                for fn in fn_list:
                    self.exp_files[fn.replace('.json', '')] = 1
        #

        self._load_data()

        if cfg.limit_bspn_vocab:
            self.bspn_masks = self._construct_bspn_constraint()
        if cfg.limit_aspn_vocab:
            self.aspn_masks = self._construct_aspn_constraint()

        self.multi_acts_record = None

    def get_exp_domains(self, exp_domains, all_domains_list):
        if 'hotel' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'hotel']
                domains = [d for d in all_domains_list if 'hotel' not in d and 'multi' not in d]
            else:
                # ['hotel']
                domains = ['hotel_single', 'hotel_multi']
        if 'train' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'train']
                domains = [d for d in all_domains_list if 'train' not in d and 'multi' not in d]
            else:
                # ['train']
                domains = ['train_single', 'train_multi']
        if 'attraction' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'attraction']
                domains = [d for d in all_domains_list if 'attraction' not in d and 'multi' not in d]
            else:
                # ['attraction']
                domains = ['attraction_single', 'attraction_multi']
        if 'restaurant' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'restaurant']
                domains = [d for d in all_domains_list if 'restaurant' not in d and 'multi' not in d]
            else:
                # ['restaurant']
                domains = ['restaurant_single', 'restaurant_multi']
        if 'taxi' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'taxi']
                domains = [d for d in all_domains_list if 'taxi' not in d and 'multi' not in d]
            else:
                # ['taxi']
                domains = ['taxi_single', 'taxi_multi']
        return domains

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)
        # for word in ontology.all_slots:
            # to be determine whether slot should be [slot]
            # if slot, tokenizer having trouble decoding.
            # special_tokens.append(word)
        for word in self.vocab._word2idx.keys():
            if word.startswith('[value_') and word.endswith(']'):
                special_tokens.append(word)
        special_tokens.extend(ontology.special_tokens)

        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to gpt tokenizer.')

        cfg.pad_id = self.tokenizer.encode('<pad>')[0]

    def _build_vocab(self):
        self.vocab = utils.Vocab(cfg.vocab_size)
        vp = cfg.vocab_path_train if cfg.mode == 'train' or cfg.vocab_path_eval is None else cfg.vocab_path_eval
        # vp = cfg.vocab_path+'.json.freq.json'
        self.vocab.load_vocab(vp)
        return self.vocab.vocab_size

    def _construct_bspn_constraint(self):
        bspn_masks = {}
        valid_domains = ['restaurant', 'hotel',
                         'attraction', 'train', 'taxi', 'hospital']
        all_dom_codes = [self.vocab.encode('['+d+']') for d in valid_domains]
        all_slot_codes = [self.vocab.encode(s) for s in ontology.all_slots]
        bspn_masks[self.vocab.encode(
            '<go_b>')] = all_dom_codes + [self.vocab.encode('<eos_b>'), 0]
        bspn_masks[self.vocab.encode('<eos_b>')] = [self.vocab.encode('<pad>')]
        bspn_masks[self.vocab.encode('<pad>')] = [self.vocab.encode('<pad>')]
        for domain, slot_values in self.slot_value_set.items():
            if domain == 'police':
                continue
            dom_code = self.vocab.encode('['+domain+']')
            bspn_masks[dom_code] = []
            for slot, values in slot_values.items():
                slot_code = self.vocab.encode(slot)
                if slot_code not in bspn_masks:
                    bspn_masks[slot_code] = []
                if slot_code not in bspn_masks[dom_code]:
                    bspn_masks[dom_code].append(slot_code)
                for value in values:
                    for idx, v in enumerate(value.split()):
                        if not self.vocab.has_word(v):
                            continue
                        v_code = self.vocab.encode(v)
                        if v_code not in bspn_masks:
                            # print(self.vocab._word2idx)
                            bspn_masks[v_code] = []
                        if idx == 0 and v_code not in bspn_masks[slot_code]:
                            bspn_masks[slot_code].append(v_code)
                        if idx == (len(value.split()) - 1):
                            for w in all_dom_codes + all_slot_codes:
                                if self.vocab.encode('<eos_b>') not in bspn_masks[v_code]:
                                    bspn_masks[v_code].append(
                                        self.vocab.encode('<eos_b>'))
                                if w not in bspn_masks[v_code]:
                                    bspn_masks[v_code].append(w)
                            break
                        if not self.vocab.has_word(value.split()[idx + 1]):
                            continue
                        next_v_code = self.vocab.encode(value.split()[idx + 1])
                        if next_v_code not in bspn_masks[v_code]:
                            bspn_masks[v_code].append(next_v_code)
        bspn_masks[self.vocab.encode('<unk>')] = list(bspn_masks.keys())

        with open('data/multi-woz-processed/bspn_masks.txt', 'w') as f:
            for i, j in bspn_masks.items():
                f.write(self.vocab.decode(i) + ': ' +
                        ' '.join([self.vocab.decode(int(m)) for m in j])+'\n')
        return bspn_masks

    def _construct_aspn_constraint(self):
        aspn_masks = {}
        aspn_masks = {}
        all_dom_codes = [self.vocab.encode('['+d+']')
                         for d in ontology.dialog_acts.keys()]
        all_act_codes = [self.vocab.encode('['+a+']')
                         for a in ontology.dialog_act_params]
        all_slot_codes = [self.vocab.encode(s)
                          for s in ontology.dialog_act_all_slots]
        aspn_masks[self.vocab.encode(
            '<go_a>')] = all_dom_codes + [self.vocab.encode('<eos_a>'), 0]
        aspn_masks[self.vocab.encode('<eos_a>')] = [self.vocab.encode('<pad>')]
        aspn_masks[self.vocab.encode('<pad>')] = [self.vocab.encode('<pad>')]
        # for d in all_dom_codes:
        #     aspn_masks[d] = all_act_codes
        for a in all_act_codes:
            aspn_masks[a] = all_dom_codes + all_slot_codes + \
                [self.vocab.encode('<eos_a>')]
        for domain, acts in ontology.dialog_acts.items():
            dom_code = self.vocab.encode('['+domain+']')
            aspn_masks[dom_code] = []
            for a in acts:
                act_code = self.vocab.encode('['+a+']')
                if act_code not in aspn_masks[dom_code]:
                    aspn_masks[dom_code].append(act_code)
        # for a, slots in ontology.dialog_act_params.items():
        #     act_code = self.vocab.encode('['+a+']')
        #     slot_codes = [self.vocab.encode(s) for s in slots]
        #     aspn_masks[act_code] = all_dom_codes + slot_codes + [self.vocab.encode('<eos_a>')]
        for s in all_slot_codes:
            aspn_masks[s] = all_dom_codes + all_slot_codes + \
                [self.vocab.encode('<eos_a>')]
        aspn_masks[self.vocab.encode('<unk>')] = list(aspn_masks.keys())

        with open('data/multi-woz-processed/aspn_masks.txt', 'w') as f:
            for i, j in aspn_masks.items():
                f.write(self.vocab.decode(i) + ': ' +
                        ' '.join([self.vocab.decode(int(m)) for m in j])+'\n')
        return aspn_masks

    def _load_data(self, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """
        if save_temp: # save encoded data
            if 'all' in cfg.exp_domains:
                encoded_file = os.path.join(cfg.data_path, 'new_db_se_blank_encoded.data.json') 
                # encoded: no sos, se_encoded: sos and eos
                # db: add db results every turn
            else:
                xdomain_dir = './experiments_Xdomain/data'
                if not os.path.exists(xdomain_dir):
                    os.makedirs(xdomain_dir)
                encoded_file = os.path.join(xdomain_dir, '{}-encoded.data.json'.format('-'.join(cfg.exp_domains))) 

            if os.path.exists(encoded_file):
                logging.info('Reading encoded data from {}'.format(encoded_file))
                self.data = json.loads(
                    open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
                encoded_data = json.loads(open(encoded_file, 'r', encoding='utf-8').read())
                self.train = encoded_data['train']
                # print(self.train)
                self.dev = encoded_data['dev']
                # print(self.dev)
                self.test = encoded_data['test']
                # print(self.test)
            else:
                logging.info('Encoding data now and save the encoded data in {}'.format(encoded_file))
                # not exists, encode data and save
                self.data = json.loads(
                    open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
                self.train, self.dev, self.test = [], [], []
                for fn, dial in self.data.items():
                    # print('dev_files', self.dev_files)
                    # print('test_files', self.test_files)
                    # print(fn)
                    if '.json' in fn:
                        fn = fn.replace('.json', '')
                    if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                        if self.dev_files.get(fn):
                            self.dev.append(self._get_encoded_data(fn, dial))
                        elif self.test_files.get(fn):
                            # print(fn)
                            self.test.append(self._get_encoded_data(fn, dial))
                        else:
                            self.train.append(self._get_encoded_data(fn, dial))
                
                # save encoded data
                encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
        
        else: # directly read processed data and encode
            self.data = json.loads(
                open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
            self.train, self.dev, self.test = [], [], []
            for fn, dial in self.data.items():
                if '.json' in fn:
                    fn = fn.replace('.json', '')
                if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                    if self.dev_files.get(fn):
                        self.dev.append(self._get_encoded_data(fn, dial))
                    elif self.test_files.get(fn):
                        self.test.append(self._get_encoded_data(fn, dial))
                    else:
                        self.train.append(self._get_encoded_data(fn, dial))
        # if save_temp:
        #     json.dump(self.test, open(
        #         'data/multi-woz-analysis/test.encoded.json', 'w'), indent=2)
        #     self.vocab.save_vocab('data/multi-woz-analysis/vocab_temp')

        random.shuffle(self.train)
        # random.shuffle(self.dev)
        # random.shuffle(self.test)
        logging.info('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        for idx, t in enumerate(dial['log']):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn

            # enc['user'] = self.vocab.sentence_encode(t['user'].split() + ['<eos_u>'])
            # enc['usdx'] = self.vocab.sentence_encode(t['user_delex'].split() + ['<eos_u>'])
            # enc['resp'] = self.vocab.sentence_encode(t['resp'].split() + ['<eos_r>'])
            # enc['bspn'] = self.vocab.sentence_encode(t['constraint'].split() + ['<eos_b>'])
            # enc['bsdx'] = self.vocab.sentence_encode(t['cons_delex'].split() + ['<eos_b>'])
            # enc['aspn'] = self.vocab.sentence_encode(t['sys_act'].split() + ['<eos_a>'])
            # enc['dspn'] = self.vocab.sentence_encode(t['turn_domain'].split() + ['<eos_d>'])

            # use gpt tokenizer directly tokenize word list, prone to encode unknown words to |endoftext|
            # enc['user'] = self.tokenizer.encode(
            #     t['user'].split() + ['<eos_u>'])
            # enc['usdx'] = self.tokenizer.encode(
            #     t['user_delex'].split() + ['<eos_u>'])
            # enc['resp'] = self.tokenizer.encode(
            #     t['resp'].split() + ['<eos_r>'])
            # enc['bspn'] = self.tokenizer.encode(
            #     t['constraint'].split() + ['<eos_b>'])
            # enc['bsdx'] = self.tokenizer.encode(
            #     t['cons_delex'].split() + ['<eos_b>'])
            # enc['aspn'] = self.tokenizer.encode(
            #     t['sys_act'].split() + ['<eos_a>'])
            # enc['dspn'] = self.tokenizer.encode(
            #     t['turn_domain'].split() + ['<eos_d>'])


            # gpt use bpe to encode strings, very very slow. ~9min
            # in tokenization_utils.encode I find encode can pad_to_max_length, and reutrn tensor
            enc['user'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize( 
                '<sos_u> ' +
                t['user'] + ' <eos_u>'))
            enc['usdx'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_u> ' +
                t['user'] + ' <eos_u>'))
            enc['resp'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_r> ' +
                t['resp'] + ' <eos_r>'))
            enc['bspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_b> ' +
                t['constraint'] + ' <eos_b>'))
            enc['bsdx'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_b> ' +
                t['cons_delex'] + ' <eos_b>'))
            enc['aspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_a> ' +
                t['sys_act'] + ' <eos_a>'))
            enc['dspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_d> ' +
                t['turn_domain'] + ' <eos_d>'))


            enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']
            if cfg.multi_acts_training:
                enc['aspn_aug'] = []
                if fn in self.multi_acts:
                    turn_ma = self.multi_acts[fn].get(str(idx), {})
                    for act_type, act_spans in turn_ma.items():
                        enc['aspn_aug'].append([self.tokenizer.encode(
                            a.split()+['<eos_a>']) for a in act_spans])

            # add db results to enc, at every turn
            db_pointer = self.bspan_to_DBpointer(t['constraint'], t['turn_domain'].split())
            # db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']
            enc['db'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_db> ' +
                db_pointer + ' <eos_db>'))

            encoded_dial.append(enc)
        return encoded_dial

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx+1]
                        ns = self.vocab.decode(ns) if type(
                            ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx+1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        # vector = self.db.addDBPointer(match_dom, match)
        vector = self.db.addDBIndicator(match_dom, match)
        return vector
    
    def aspan_to_act_list(self, aspan):
        aspan = aspan.split() if isinstance(aspan, str) else aspan
        acts = []
        domain = None
        conslen = len(aspan)
        for idx, cons in enumerate(aspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_a>':
                break
            if '[' in cons and cons[1:-1] in ontology.dialog_acts:
                domain = cons[1:-1]

            elif '[' in cons and cons[1:-1] in ontology.dialog_act_params:
                if domain is None:
                    continue
                vidx = idx+1
                if vidx == conslen:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
                    break
                vt = aspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                no_param_act = True
                while vidx < conslen and vt != '<eos_a>' and '[' not in vt:
                    no_param_act = False
                    acts.append(domain+'-'+cons[1:-1]+'-'+vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = aspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if no_param_act:
                    acts.append(domain+'-'+cons[1:-1]+'-none')

        return acts

    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            dom = self.vocab.decode(d) if type(d) is not str else d
            if dom != '<eos_d>':
                domains[dom] = 1
            else:
                break
        return domains


    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous ubar, U_t, B_t, A_t] predict R_t
            firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        context_list = []
        # predict_list = []
        prompt = ''
        if cfg.use_true_curr_bspn:
            if cfg.use_true_curr_aspn: # only predict resp
                context_list = ['user', 'bspn', 'db','aspn']
                # context_list = ['user','aspn'] # predict resp based on current aspn and bspn
                # predict_list = ['resp']
                prompt = '<sos_r>'
            else: # predicted aspn
                context_list = ['user', 'bspn', 'db']
                # predict_list = ['aspn', 'resp']
                prompt = '<sos_a>'
        else: # predict bspn aspn resp. db are not predicted. this part tbd.
            context_list = ['user']
            # predict_list = ['bspn', 'db','aspn', 'resp']
            prompt = '<sos_b>'
        
        if first_turn:
            context = []
            for c in context_list:
                context += turn[c]

            inputs['context'] = context + self.tokenizer.encode([prompt])
            inputs['labels'] = context
            # e43 with BABAU
            # inputs['labels'] = []
            
        else:
            context = []
            for c in context_list:
                context += turn[c]

            pv_context = pv_turn['labels'] + pv_turn['bspn'] + pv_turn['db'] + pv_turn['aspn'] + pv_turn['resp']
            # e43 with BABAU
            # pv_context = pv_turn['labels'] + pv_turn['bspn'] + pv_turn['db'] + pv_turn['aspn']
                
            # prompt response, add sos_r
            inputs['context'] = pv_context + context + self.tokenizer.encode([prompt])
            # context just the current turn
            # inputs['context'] = context + self.tokenizer.encode([prompt])
            # context just the current action

            if cfg.use_all_previous_context:
                inputs['labels'] = pv_context + context # use all previous ubar history
            else:
                inputs['labels'] = context# use privosu trun

        if len(inputs['context']) > 900:
            # print('len exceeds 900')
            inputs['context'] = inputs['context'][-900:]
         
        return inputs

    def convert_turn_eval_URURU(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous U_t, R_t] predict R_t
            firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        context_list = []
        predict_list = []
        prompt = ''
        if cfg.use_true_curr_bspn:
            if cfg.use_true_curr_aspn: # only predict resp
                context_list = ['user', 'bspn', 'db','aspn']
                # context_list = ['user','aspn'] # predict resp based on current aspn and bspn
                predict_list = ['resp']
                prompt = '<sos_r>'
            else: # predicted aspn
                context_list = ['user', 'bspn', 'db']
                predict_list = ['aspn', 'resp']
                prompt = '<sos_a>'
        else: # predict bspn aspn resp. db are not predicted. this part tbd.
            context_list = ['user']
            predict_list = ['bspn', 'db','aspn', 'resp']
            prompt = '<sos_b>'
        
        if first_turn:
            context = []
            for c in context_list:
                context += turn[c]

            # prompt response, add sos_r
            inputs['context'] = context + self.tokenizer.encode([prompt])

            # labels = []
            # for p in predict_list:
            #     labels += turn[p]
            # inputs['labels'] = context + labels # or just labels
            inputs['labels'] = turn['user']
        else:
            context = []
            for c in context_list:
                context += turn[c]

            pv_context = pv_turn['labels'] + pv_turn['resp']
                
            # prompt response, add sos_r
            inputs['context'] = pv_context + context + self.tokenizer.encode([prompt])
            # context just the current turn
            # inputs['context'] = context + self.tokenizer.encode([prompt])
            # context just the current action

            if cfg.use_all_previous_context:
                inputs['labels'] = pv_context + context # use all previous ubar history
            else:
                inputs['labels'] = context# use privosu trun

        if len(inputs['context']) > 900:
            # print('len exceeds 900')
            inputs['context'] = inputs['context'][-900:]
         
        return inputs
    

    def convert_batch_session(self, dial_batch):
        """
        convert the whole session for training
        concat [U_0, B_0, A_0, R_0, ... , U_n, B_n, A_n, R_n]

        try: [user, bspn, aspn, resp]
        or
        try: [user, bspn, db, aspn, resp]
        """
        inputs = {}
        contexts = []
        cell_list = ['user', 'bspn', 'db', 'aspn', 'resp']
        for idx, dial in enumerate(dial_batch):
            context = []
            for turn_num, turn in enumerate(dial):
                for cell in cell_list:
                    context.extend(turn[cell])
            contexts.append(context)
        
        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        return inputs

    def convert_batch_turn(self, turn_batch, pv_batch, first_turn=False):
        """
        URURU
        convert the current and the last turn
        concat [U_0,R_0,...,U_{t-1}, R_{t-1}, U_t, B_t, A_t, R_t]
        firts turn: [U_t, B_t, A_t, R_t]
        try: [user, bspn, db, aspn, resp]

        """
        inputs = {}
        if first_turn:
            contexts = []
            labels = []
            batch_zipped = zip(
                turn_batch['user'], turn_batch['bspn'], turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
            for u, b, db, a, r in batch_zipped:
                context = u+b+db+a+r
                contexts.append(context)
                label = u + r
                labels.append(label)
            inputs['contexts'] = contexts
            inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)

            inputs['labels'] = labels
        else:
            contexts = []
            labels = []
            batch_zipped = zip(pv_batch,
                               turn_batch['user'], turn_batch['bspn'], turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
            for ur, u, b, db, a, r in batch_zipped:
                context = ur + u + b + db + a + r
                contexts.append(context)
                label = ur + u + r
                labels.append(label)
            inputs['contexts'] = contexts
            contexts_np, lengths = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
            inputs['contexts_np'] = contexts_np
            inputs['lengths'] = lengths

            inputs['labels'] = labels
        return inputs

    def convert_batch_gpt(self, turn_batch, pv_batch, first_turn=False):
        """
        convert the current and the last turn
        concat [U_{t-1}, B_{t-1}, A_{t-1}, R_{t-1}, U_t, B_t, A_t, R_t]
        firts turn: [U_t, B_t, A_t, R_t]
        try: [usdx, bspn, aspn, resp]

        """
        inputs = {}
        if first_turn:
            contexts = []
            batch_zipped = zip(
                turn_batch['usdx'], turn_batch['bspn'], turn_batch['aspn'], turn_batch['resp'])
            for u, b, a, r in batch_zipped:
                context = u+b+a+r
                contexts.append(context)
            inputs['contexts'] = contexts
            # padSeqs to make [UBAR] the same length
            inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        else:
            contexts = []
            batch_zipped = zip(pv_batch['pv_usdx'], pv_batch['pv_bspn'], pv_batch['pv_aspn'], pv_batch['pv_resp'],
                               turn_batch['usdx'], turn_batch['bspn'], turn_batch['aspn'], turn_batch['resp'])
            for pu, pb, pa, pr, u, b, a, r in batch_zipped:
                context = pu + pb + pa + pr + u + b + a + r
                contexts.append(context)
            inputs['contexts'] = contexts
            contexts_np, lengths = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
            inputs['contexts_np'] = contexts_np
            inputs['lengths'] = lengths
        return inputs

    def convert_batch(self, py_batch, py_prev, first_turn=False):
        inputs = {}
        if first_turn:
            for item, py_list in py_prev.items():
                batch_size = len(py_batch['user'])
                inputs[item+'_np'] = np.array([[1]] * batch_size)
                inputs[item+'_unk_np'] = np.array([[1]] * batch_size)
        else:
            for item, py_list in py_prev.items():
                if py_list is None:
                    continue
                if not cfg.enable_aspn and 'aspn' in item:
                    continue
                if not cfg.enable_bspn and 'bspn' in item:
                    continue
                if not cfg.enable_dspn and 'dspn' in item:
                    continue
                prev_np = utils.padSeqs(
                    py_list, truncated=cfg.truncated, trunc_method='pre')
                inputs[item+'_np'] = prev_np
                if item in ['pv_resp', 'pv_bspn']:
                    inputs[item+'_unk_np'] = deepcopy(inputs[item+'_np'])
                    # <unk>, restrict vocab size to 3k, map ids>3k to <unk>
                    inputs[item+'_unk_np'][inputs[item+'_unk_np']
                                           >= self.vocab_size] = 2
                else:
                    inputs[item+'_unk_np'] = inputs[item+'_np']

        for item in ['user', 'usdx', 'resp', 'bspn', 'aspn', 'bsdx', 'dspn']:
            if not cfg.enable_aspn and item == 'aspn':
                continue
            if not cfg.enable_bspn and item == 'bspn':
                continue

            if not cfg.enable_dspn and item == 'dspn':
                continue
            py_list = py_batch[item]
            trunc_method = 'post' if item == 'resp' else 'pre'
            # max_length = cfg.max_nl_length if item in ['user', 'usdx', 'resp'] else cfg.max_span_length
            inputs[item+'_np'] = utils.padSeqs(
                py_list, truncated=cfg.truncated, trunc_method=trunc_method)
            if item in ['user', 'usdx', 'resp', 'bspn']:
                inputs[item+'_unk_np'] = deepcopy(inputs[item+'_np'])
                inputs[item+'_unk_np'][inputs[item+'_unk_np']
                                       >= self.vocab_size] = 2   # <unk>
            else:
                inputs[item+'_unk_np'] = inputs[item+'_np']

        if cfg.multi_acts_training and cfg.mode == 'train':
            inputs['aspn_bidx'], multi_aspn = [], []
            for bidx, aspn_type_list in enumerate(py_batch['aspn_aug']):
                if aspn_type_list:
                    for aspn_list in aspn_type_list:
                        random.shuffle(aspn_list)
                        # choose one random act span in each act type
                        aspn = aspn_list[0]
                        multi_aspn.append(aspn)
                        inputs['aspn_bidx'].append(bidx)
                        if cfg.multi_act_sampling_num > 1:
                            for i in range(cfg.multi_act_sampling_num):
                                if len(aspn_list) >= i+2:
                                    # choose one random act span in each act type
                                    aspn = aspn_list[i+1]
                                    multi_aspn.append(aspn)
                                    inputs['aspn_bidx'].append(bidx)

            if multi_aspn:
                inputs['aspn_aug_np'] = utils.padSeqs(
                    multi_aspn, truncated=cfg.truncated, trunc_method='pre')
                # [all available aspn num in the batch, T]
                inputs['aspn_aug_unk_np'] = inputs['aspn_aug_np']

        inputs['db_np'] = np.array(py_batch['pointer'])
        inputs['turn_domain'] = py_batch['turn_domain']

        return inputs

    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                     'dspn_gen', 'dspn', 'bspn', 'pointer']

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'trun_num': len(turns)}
            for f in field[2:]:
                entry[f] = '' # ???
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)

                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])
                        # if key != 'resp_gen':
                        #     # remove eos/sos in span
                        #     if eos_syntax[key] in v:
                        #         v.remove(eos_syntax[key])
                        #     if sos_syntax[key] in v:
                        #         v.remove(sos_syntax[key])
                        # else: # 'resp_gen'
                        #     sos_index = 0
                        #     eos_index = -1
                        #     if sos_syntax[key] in v:
                        #         sos_index = v.index(sos_syntax[key])
                        #     if eos_syntax[key] in v:
                        #         eos_index = v.index(eos_syntax[key])
                        #     else:
                        #         pass # take too long
                        #         # no <eos_r> found, stop at any eos_tokens
                        #         # for i in range(sos_index+1, len(v)):
                        #         #     if v[i] in sos_syntax.values() or v[i] in eos_syntax.values():
                        #         #         eos_index = i
                        #     v = v[sos_index+1: eos_index]


                        # v = self.tokenizer.convert_tokens_to_string(v)
                        v = " ".join(v)
                    else: 
                        pass # v = v
                    entry[key] = v

                results.append(entry)

        return results, field

    def wrap_result(self, result_dict, eos_syntax=None):
        decode_fn = self.vocab.sentence_decode
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax

        if cfg.bspn_mode == 'bspn':
            field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bspn', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                     'dspn_gen', 'dspn', 'pointer']
        elif not cfg.enable_dst: # this
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                     'dspn_gen', 'dspn', 'bspn', 'pointer']
        else:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                     'dspn_gen', 'dspn', 'bspn_gen', 'bspn', 'pointer']
        if self.multi_acts_record is not None:
            field.insert(7, 'multi_act_gen')

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            for prop in field[2:]:
                entry[prop] = ''
            results.append(entry)
            for turn_no, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)
                    entry[key] = decode_fn(
                        v, eos=eos_syntax[key]) if key in eos_syntax and v != '' else v
                results.append(entry)
        return results, field

    def restore(self, resp, domain, constraint_dict, mat_ents):
        restored = resp

        restored = restored.replace('[value_reference]', '53022')
        restored = restored.replace('[value_car]', 'BMW')

        # restored.replace('[value_phone]', '830-430-6666')
        for d in domain:
            constraint = constraint_dict.get(d, None)
            if constraint:
                if 'stay' in constraint:
                    restored = restored.replace(
                        '[value_stay]', constraint['stay'])
                if 'day' in constraint:
                    restored = restored.replace(
                        '[value_day]', constraint['day'])
                if 'people' in constraint:
                    restored = restored.replace(
                        '[value_people]', constraint['people'])
                if 'time' in constraint:
                    restored = restored.replace(
                        '[value_time]', constraint['time'])
                if 'type' in constraint:
                    restored = restored.replace(
                        '[value_type]', constraint['type'])
                if d in mat_ents and len(mat_ents[d]) == 0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace(
                                '[value_price]', constraint['pricerange'])
                        if s+']' in restored:
                            restored = restored.replace(
                                '[value_%s]' % s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace(
                    '[value_choice]', str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', '3')

        # restored.replace('[value_car]', 'BMW')

        try:
            ent = mat_ents.get(domain[-1], [])
            if ent:
                ent = ent[0]

                for t in restored.split():
                    if '[value' in t:
                        slot = t[7:-1]
                        if ent.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            restored = restored.replace(t, ent[slot])
                        elif slot == 'price':
                            if ent.get('pricerange'):
                                restored = restored.replace(
                                    t, ent['pricerange'])
                            else:
                                print(restored, domain)
        except:
            print(resp)
            print(restored)
            quit()

        restored = restored.replace('[value_phone]', '62781111')
        restored = restored.replace('[value_postcode]', 'CG9566')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')

        # if '[value_' in restored:

        #     print(domain)
        #     # print(mat_ents)
        #     print(resp)
        #     print(restored)
        return restored

    def record_utterance(self, result_dict):
        decode_fn = self.vocab.sentence_decode

        ordered_dial = {}
        for dial_id, turns in result_dict.items():
            diverse = 0
            turn_count = 0
            for turn_no, turn in enumerate(turns):
                act_collect = {}
                act_type_collect = {}
                slot_score = 0
                for i in range(cfg.nbest):
                    aspn = decode_fn(turn['multi_act'][i],
                                     eos=ontology.eos_tokens['aspn'])
                    pred_acts = self.aspan_to_act_list(' '.join(aspn))
                    act_type = ''
                    for act in pred_acts:
                        d, a, s = act.split('-')
                        if d + '-' + a not in act_collect:
                            act_collect[d + '-' + a] = {s: 1}
                            slot_score += 1
                            act_type += d + '-' + a + ';'
                        elif s not in act_collect:
                            act_collect[d + '-' + a][s] = 1
                            slot_score += 1
                    act_type_collect[act_type] = 1
                turn_count += 1
                diverse += len(act_collect) * 3 + slot_score
            ordered_dial[dial_id] = diverse/turn_count

        ordered_dial = sorted(ordered_dial.keys(),
                              key=lambda x: -ordered_dial[x])


        dialog_record = {}

        with open(cfg.eval_load_path + '/dialogue_record.csv', 'w') as rf:
            writer = csv.writer(rf)

            for dial_id in ordered_dial:
                dialog_record[dial_id] = []
                turns = result_dict[dial_id]
                writer.writerow([dial_id])
                for turn_no, turn in enumerate(turns):
                    user = decode_fn(
                        turn['user'], eos=ontology.eos_tokens['user'])
                    bspn = decode_fn(
                        turn['bspn'], eos=ontology.eos_tokens['bspn'])
                    aspn = decode_fn(
                        turn['aspn'], eos=ontology.eos_tokens['aspn'])
                    resp = decode_fn(
                        turn['resp'], eos=ontology.eos_tokens['resp'])
                    constraint_dict = self.bspan_to_constraint_dict(bspn)
                    # print(constraint_dict)
                    mat_ents = self.db.get_match_num(constraint_dict, True)
                    domain = [i[1:-1]
                              for i in self.dspan_to_domain(turn['dspn']).keys()]
                    restored = self.restore(
                        resp, domain, constraint_dict, mat_ents)
                    writer.writerow(
                        [turn_no, user, turn['pointer'], domain, restored, resp])
                    turn_record = {'user': user, 'bspn': bspn, 'aspn': aspn,
                                   'dom': domain, 'resp': resp, 'resp_res': restored}

                    resp_col = []
                    aspn_col = []
                    resp_restore_col = []
                    for i in range(cfg.nbest):
                        aspn = decode_fn(
                            turn['multi_act'][i], eos=ontology.eos_tokens['aspn'])
                        resp = decode_fn(
                            turn['multi_resp'][i], eos=ontology.eos_tokens['resp'])

                        restored = self.restore(
                            resp, domain, constraint_dict, mat_ents)
                        resp_col.append(resp)
                        resp_restore_col.append(restored)
                        aspn_col.append(aspn)

                    zipped = list(zip(resp_restore_col, resp_col, aspn_col))
                    zipped.sort(key=lambda s: len(s[0]))
                    resp_restore_col = list(list(zip(*zipped))[0])
                    aspn_col = list(list(zip(*zipped))[2])
                    resp_col = list(list(zip(*zipped))[1])
                    turn_record['aspn_col'] = aspn_col
                    turn_record['resp_col'] = resp_col
                    turn_record['resp_res_col'] = resp_restore_col
                    for i in range(cfg.nbest):
                        # aspn = decode_fn(turn['multi_act'][i], eos=ontology.eos_tokens['aspn'])
                        resp = resp_col[i]
                        aspn = aspn_col[i]
                        resp_restore = resp_restore_col[i]

                        writer.writerow(['', resp_restore, resp, aspn])

                    dialog_record[dial_id].append(turn_record)

            # json.dump(dialog_record, open(cfg.eval_load_path + '/resultdict.json','w'))


if __name__ == '__main__':
    reader = MultiWozReader()
    # for aspan in ["[general] [bye] [welcome] <eos_a>","[train] [inform] trainid destination arrive leave [offerbook] [general] [reqmore] <eos_a>",]:
    #     act = reader.aspan_to_constraint_dict(aspan.split())
    #     print('！！！')
    #     print(act)

    for bspan in ["[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday", "[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday <eos_b>"]:
        encoded = reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bspn')
        print(cons)
    for bspan in ["[taxi] destination departure leave [hotel] name [attraction] name people day", "[taxi] destination departure leave [hotel] name [attraction] name people day <eos_b>"]:
        encoded = reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bsdx')
        print(cons)
