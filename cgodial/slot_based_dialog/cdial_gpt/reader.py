import copy
import os
import csv
import random
import logging
import json
import utils
import ontology as ontology
from collections import OrderedDict

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
                if key=='dial_id':
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
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
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
            if set_name != 'test' and k == 1 or k >= 17:
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

        # if set_name == 'train':
        #     random.shuffle(all_batches)
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



class RisaWOZReader(_ReaderBase):
    def __init__(self, tokenizer):
        super().__init__()

        with open(cfg.slot_value_set_path) as f:
            self.slot_value_set = json.load(f)

        self.tokenizer = tokenizer
        
        if cfg.mode=='train':
            self.add_sepcial_tokens()
        # self.add_sepcial_tokens()
        
        self.vocab_size = self.tokenizer.vocab_size

        with open(cfg.test_list) as f:
            self.test_list = [n.lower() for n in json.load(f)]
            
        with open(cfg.val_list) as f:
            self.val_list = [n.lower() for n in json.load(f)]
            
        with open(cfg.train_list) as f:
            self.train_list = [n.lower() for n in json.load(f)]

        self._load_data()
        self.get_eval_data('train')
        self.get_batches('train')
        self.get_eval_data('dev')
        self.get_batches('dev')
        self.get_eval_data('test')
        self.get_batches('test')


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
            word = '[db_%d]' %db_id
            special_tokens.append(word)
        # for word in ontology.all_slots:
            # to be determine whether slot should be [slot]
            # if slot, tokenizer having trouble decoding.
            # special_tokens.append(word)
        for word_list in self.slot_value_set.values():
            for w in word_list:
                if '[v_%s]'% ontology.normalize_slot(w) not in special_tokens:
                    special_tokens.append('[v_%s]' % ontology.normalize_slot(w))
        
        special_tokens.extend(ontology.special_tokens)

        special_tokens_dict = {'additional_special_tokens': special_tokens}
        print(special_tokens_dict)
        logging.info(str(special_tokens_dict))
        self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to gpt tokenizer....')
        print('Added special tokens to gpt tokenizer....')

        cfg.pad_id = self.tokenizer.convert_tokens_to_ids([ontology.PAD_token])[0]

    def _load_data(self):
        """
        load processed data and encode, or load already encoded data
        """
        encoded_file = os.path.join('gpt_encode_data.json')
        if os.path.exists(encoded_file):
            logging.info('Reading encoded data from {}'.format(encoded_file))
            with open(cfg.data_path+cfg.data_file) as f:
                self.data = json.load(f)
            with open(encoded_file) as f:
                encoded_data = json.load(f)
            self.train = encoded_data['train']
            self.dev = encoded_data['dev']
            self.test = encoded_data['test']
            
        else:
            logging.info('Encoding data now and save the encoded data in {}'.format(encoded_file))
            # not exists, encode data and save
            with open(cfg.data_path+cfg.data_file) as f:
                self.data = json.load(f)
                
            self.train, self.dev, self.test = [], [], []
            
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
                else:
                    self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log')))
                    # Do not load augmented training data to save training/loading time
                    # self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr0')))
                    # self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr1')))
                    # self.train.append(copy.deepcopy(self._get_encoded_data(fn, dial, 'log_asr2')))
                    
            # save encoded data
            encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
            json.dump(encoded_data, open(encoded_file, 'w'), ensure_ascii=False)

        random.shuffle(self.train)
        # random.shuffle(self.dev)
        # random.shuffle(self.test)
        logging.info('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))
        print('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def _get_encoded_data(self, fn, dial, log_key):
        encoded_dial = []
        for idx, t in enumerate(dial[log_key]):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn
            enc['log_key'] = log_key
            if log_key != 'log':
                enc['dial_id'] = fn + '-' +log_key
            # in tokenization_utils.encode I find encode can pad_to_max_length, and reutrn tensor
            
            enc['user'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_u> ' +t['user'] + ' <eos_u>'))
            
            enc['bspn'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_b> ' +t['constraint'] + ' <eos_b>'))

            enc['db'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_db> ' +t['db'] + ' <eos_db>'))
            
            enc['aspn'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_a> ' +t['sys_act'] + ' <eos_a>'))

            enc['resp'] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize('<sos_r> ' + t['resp'] + ' <eos_r>'))
            
            # print(self.tokenizer.tokenize('<sos_u> ' +t['user'] + ' <eos_u>'))
            # print(self.tokenizer.tokenize('<sos_b> ' +t['constraint'] + ' <eos_b>'))
            # print(self.tokenizer.tokenize('<sos_db> ' +t['db'] + ' <eos_db>'))
            # print(self.tokenizer.tokenize('<sos_a> ' +t['sys_act'] + ' <eos_a>'))
            # print(self.tokenizer.tokenize('<sos_r> ' + t['resp'] + ' <eos_r>'))
            # print()
            
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']
            enc['turn_type'] = t['type']
            enc['turn_succ'] = t['turn_succ']
            encoded_dial.append(copy.deepcopy(enc))
        return encoded_dial

    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous ubar, U_t, B_t, A_t] predict R_t
            firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        context_list = []
        # predict_list = []
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

            inputs['context'] = context + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
            inputs['labels'] = context

            # inputs['generated_bs'] = turn['user'] + turn['bspn']
            # inputs['generated_ar'] = turn['user'] + turn['bspn'] + turn['db'] + turn['aspn'] + turn['resp']
            
        else:
            context = []
            for c in context_list:
                context += turn[c]

            pv_context = pv_turn['labels'] + pv_turn['bspn'] + pv_turn['db'] + pv_turn['aspn'] + pv_turn['resp']
            # e43 with BABAU
            # pv_context = pv_turn['labels'] + pv_turn['bspn'] + pv_turn['db'] + pv_turn['aspn']
                
            # prompt response, add sos_r
            inputs['context'] = pv_context + context + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
            # context just the current turn
            # inputs['context'] = context + self.tokenizer.encode([prompt])
            # context just the current action

            # inputs['generated_bs'] = pv_context + turn['user'] + turn['bspn']
            # inputs['generated_ar'] = pv_context + turn['user'] + turn['bspn'] + turn['db'] + turn['aspn'] + turn['resp']

            if cfg.use_all_previous_context:
                inputs['labels'] = pv_context + context # use all previous ubar history
            else:
                inputs['labels'] = context# use privosu trun

        if len(inputs['context']) > 850:
            # print('len exceeds 850')
            diff = len(inputs['context']) - 850
            # inputs['generated_bs'] = inputs['generated_bs'][diff:]
            # inputs['generated_ar'] = inputs['generated_ar'][diff:]
            inputs['context'] = inputs['context'][-850:]
         
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
    
    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'turn_succ',
                 'user',
                 'bspn','bspn_gen',
                 'aspn_gen', 'aspn',
                 'resp_gen', 'resp',
                 'turn_type', 'turn_domain'
                 ]

        for dial_id, turns in result_dict.items():
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    
                    # if key == 'turn_domain':
                    #     v = ' '.join(v)

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



if __name__ == '__main__':
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("./gpt_chinese")
    model = GPT2LMHeadModel.from_pretrained("./gpt_chinese")
    reader = RisaWOZReader(tokenizer)

    # sys_act = "[旅游景点] 区域=工业园区|消费=便宜|名称=独墅湖教堂 [酒店] 房型=大床房|星级=5|区域=工业园区|名称=苏州金鸡湖新罗酒店 [餐厅] 区域=工业园区|菜系=江浙菜|名称=得月楼 [通用] number=17837575842|name=银"
    # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sys_act))))
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<sos_a> ' + sys_act + ' <eos_a>')))
    #
    #
    #
    # tokens = reader.tokenizer.tokenize('[旅游景点] [v_游戏性能] [v_最适合人群] [旅游景点] 景点类型=水乡古镇|消费=中等|名称=山塘街[affirm]<sos_db> <eos_db> <pad> <unk> <eos_r>')
    # print(tokens)
    # tokens_id = reader.tokenizer.convert_tokens_to_ids(tokens)
    # print(tokens_id)
    # print()
    # print(reader.tokenizer.convert_ids_to_tokens(tokens_id))
    # vocab_size = reader.tokenizer.vocab_size
    # print(reader.tokenizer.convert_ids_to_tokens(list(range(21129, 21329))))
    #
    # all_batches = reader.get_batches('test')
    # data_iterator = reader.get_nontranspose_data_iterator(
    #     all_batches)
    # for batch_idx, dial_batch in enumerate(data_iterator):
    #     inputs = reader.convert_batch_session(dial_batch)
    #     for context in inputs['contexts']:
    #         print(reader.tokenizer.decode(context))
    #     print(input())
    
    
    # model = GPT2Tokenizer.from_pretrained("/Users/daiyp/Documents/AdvResProj/CGoDial_Exp/risawoz/CDial_GPT")
    
    
    # for aspan in ["[general] [bye] [welcome] <eos_a>","[train] [inform] trainid destination arrive leave [offerbook] [general] [reqmore] <eos_a>",]:
    #     act = reader.aspan_to_constraint_dict(aspan.split())
    #     print('！！！')
    #     print(act)
    #
    # for bspan in ["[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday", "[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday <eos_b>"]:
    #     encoded = reader.vocab.sentence_encode(bspan.split())
    #     print(encoded)
    #     cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bspn')
    #     print(cons)
    # for bspan in ["[taxi] destination departure leave [hotel] name [attraction] name people day", "[taxi] destination departure leave [hotel] name [attraction] name people day <eos_b>"]:
    #     encoded = reader.vocab.sentence_encode(bspan.split())
    #     print(encoded)
    #     cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bsdx')
    #     print(cons)
