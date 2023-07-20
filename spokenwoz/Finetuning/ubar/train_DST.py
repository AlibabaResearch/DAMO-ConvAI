
from eval import MultiWozEvaluator
from damd_net import DAMD, cuda_, get_one_hot_input
# from reader import MultiWozReader
from dst_reader import MultiWozReader
from config import global_config as cfg  # or config21
# from dst import ignore_none_dontcare, default_cleaning, IGNORE_TURNS_TYPE2, paser_bs
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from compute_joint_acc import compute_jacc
import warnings
import collections
from dst import default_cleaning, IGNORE_TURNS_TYPE2, paser_bs
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, AutoTokenizer,AutoModel,AutoConfig
warnings.filterwarnings("ignore")

class Modal(object):
    def __init__(self, device):
        self.device = device
        # initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        # self.tokenizer = AutoTokenizer.from_pretrained("./model/Bert/", local_files_only=True) 
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)

        # create model: gpt2
        self.model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        # config = AutoConfig.from_pretrained('bert-base-cased')
        # model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        # self.model = AutoModel.from_pretrained("./model/SPACE.model", local_files_only=True, config=config)
        
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))
        # print(self.device)
        self.model.to(self.device)  # single gpu

        #
        self.evaluator = MultiWozEvaluator(self.reader)
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log')
        else:
            self.tb_writer = None

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] * cfg.epoch_num // cfg.gradient_accumulation_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:4]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor(
            [inputs['context']]).to(self.device)
        return inputs

    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def train(self):
        """

        """
        all_batches = self.reader.get_batches('train')
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()
        if cfg.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O1")
        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)
            data_iterator = self.reader.get_nontranspose_data_iterator(
                all_batches)
            # pbar = tqdm(data_iterator)
            pbar = data_iterator
            # 69 batches for bs of 128
            for batch_idx, dial_batch in enumerate(pbar):
                inputs = self.reader.convert_batch_session(dial_batch)
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    # to tensor
                    inputs = self.add_torch_input(inputs)
                    # loss
                    outputs = self.model(inputs['contexts_tensor'])
                    # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                    loss = self.calculate_loss_and_accuracy(
                        outputs, labels=inputs['contexts_tensor'])
                    if cfg.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                            # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
                    else:
                        loss.backward()
                    tr_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(
                    #     self.model.parameters(), 5.0)
                    epoch_step += 1

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                        # end of an epoch
                        (epoch_step + \
                         1) == set_stats['num_training_steps_per_epoch']
                    ):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # global_step: actual step the optimizer took
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                            loss_scalar = (tr_loss - logging_loss) / \
                                cfg.report_interval
                            logging_loss = tr_loss
                            logs['loss'] = loss_scalar
                            logging.info(
                                'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                    global_step, epoch_step, loss_scalar
                                ))
                            # validate
                            # add to tensorboard...
                            if cfg.evaluate_during_training and loss_scalar < 10:
                                results = self.validate()
                                for k, v in results.items():
                                    eval_key = "eval_{}".format(k)
                                    logs[eval_key] = v

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(
                                        k, v, global_step)
                            # save model... 

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Epoch {} Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(epoch,(time.time()-btm)/60, tr_loss))
            # save model after every epoch
            avg_loss = tr_loss/epoch_step
            logging.info("Average loss {:.4}".format(avg_loss))
            # if avg_loss < 0.7:
            #     self.save_model(epoch, avg_loss)
            self.save_model(epoch, avg_loss)

    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2'.format(epoch+1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg

    def validate(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()

        # all_batches = self.reader.get_batches('dev')
        # data_iterator = self.reader.get_data_iterator(all_batches)
        eval_data = self.reader.get_eval_data(data)
        print(data, eval_data)
        # print(eval_data)
        model_output = {}
        set_stats = self.reader.set_stats[data]
        # print(set_stats)
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            eval_pbar = eval_data
            count = 0
            for dial_idx, dialog in enumerate(eval_pbar):
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    if count % 1000 == 0:
                        logging.info("Decoded turn: {}".format(count))
                    count += 1
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval(
                        turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    max_len=60
                    if cfg.use_true_curr_bspn:
                        if not cfg.use_true_curr_aspn:
                            max_len = 80
                        # if not cfg.use_true_curr_bspn:
                        #     max_len = 100
                        outputs = self.model.generate(
                            input_ids=inputs['context_tensor'],
                            max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
                            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0]
                        )#   no_repeat_ngram_size=4
                        # turn['generated'] = self.tokenizer.decode(outputs[0])

                        # resp_gen, nedd to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length-1:]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                        # check DB result
                        db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                        # error in pretraining data, db special tokens are just two <eos_db>.
                        db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
                    else: # predict bspn, access db, then generate act and resp
                        decoded = {'resp': [], 'bspn': [], 'aspn': []}
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+80,# top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
                        if cfg.fix_bs:
                            bspn_gen = self.rule_based_bs_fix(inputs['context'],bspn_gen)
                        # check DB result
                        if cfg.use_true_db_pointer:
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                        else:
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
                        # error in pretraining data, db special tokens are just two <eos_db>.
                        db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
                        if True:
                            context = inputs['context'][:-1] + bspn_gen + db
                            if len(context) > cfg.max_context_length:
                                context = context[-cfg.max_context_length:]
                            inputs['context_tensor_db'] = torch.tensor([context]).to(self.device)
                            context_length = len(inputs['context_tensor_db'][0])
                            outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
                                                        max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                        pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
                            generated_ar = outputs_db[0].cpu().numpy().tolist()
                            generated_ar = generated_ar[context_length-1:]
                            try:
                                decoded = self.decode_generated_act_resp(generated_ar)
                                decoded['bspn'] = bspn_gen
                            except ValueError as exception:
                                logging.info(str(exception))
                                logging.info(self.tokenizer.decode(generated_ar))
                                decoded = {'resp': [], 'bspn': [], 'aspn': []}
                        else:
                            decoded['bspn'] = bspn_gen

                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']
                    turn['context'] = self.tokenizer.decode(inputs['context'])
                    file_name = turn['dial_id']
                    if file_name not in model_output:
                        model_output[file_name] = {}
                    decoded_turn = {}
                    decoded_turn['context'] = self.tokenizer.decode(inputs['context'])
                    decoded_turn['bspn_gen'] = self.tokenizer.decode(turn['bspn_gen'])
                    decoded_turn['aspn_gen'] = self.tokenizer.decode(turn['aspn_gen'])
                    decoded_turn['resp_gen'] = self.tokenizer.decode(turn['resp_gen'])
                    decoded_turn['resp'] = self.tokenizer.decode(turn['resp'])
                    decoded_turn['bspn'] = self.tokenizer.decode(turn['bspn'])
                    decoded_turn['aspn'] = self.tokenizer.decode(turn['aspn'])
                    model_output[file_name][turn_idx] = decoded_turn

                    # check DB results
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    pv_turn['labels'] = inputs['labels'] # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    pv_turn['db'] = db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                
                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))

        logging.info("inference time: {:.2f} min".format((time.time()-btm)/60))
        # print(model_output)
        joint_acc, joint_acc_wo_cross, dict_rate = compute_jacc(data=model_output,path=cfg.eval_load_path)
        with open(cfg.gpt_path+cfg.model_output+'BS.json',"w") as f:
            json.dump(model_output,f,indent=2)
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
        # bleu, success, match,dials = self.evaluator.validation_metric(results)
        # bleu, success, match = self.evaluator.validation_metric(results)
        logging.info('Jonit ACC: {:.4f}'.format(joint_acc))
        logging.info('Jonit ACC w/o cross slot: {:.4f}'.format(joint_acc_wo_cross))
        logging.info(dict_rate)
        
        logging.info("Saving model output ot {}".format(cfg.gpt_path))
        # with open(cfg.gpt_path+cfg.model_output+'.json',"w") as f:
        #     json.dump(dials,f,indent=2)

        logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        # score = 0.5 * (success + match) + bleu
        # valid_loss = 130 - score
        # logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f    score: %.1f' % (
        #     match, success, bleu, score))
        eval_results = {}
        # eval_results['score'] = score
        # eval_results['bleu'] = bleu
        # eval_results['success'] = success
        # eval_results['match'] = match
        eval_results['joint_acc'] = joint_acc
        eval_results['joint_acc_wo_cross'] = joint_acc_wo_cross,

        model_setting, epoch_setting = cfg.eval_load_path.split('/')[1], cfg.eval_load_path.split('/')[2]
        epoch_setting += str(cfg.max_context_length)
        eval_on = '-'.join(cfg.exp_domains)
        log_file_name = os.path.join(cfg.log_path, cfg.exp_no+model_setting+'-'+eval_on+'.json')
        if os.path.exists(log_file_name):
            eval_to_json = json.load(open(log_file_name, 'r'))
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        else:
            eval_to_json = {}
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        logging.info('update eval results to {}'.format(log_file_name))
        
        return eval_results

    def validate_URURU(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()

        # all_batches = self.reader.get_batches('dev')
        # data_iterator = self.reader.get_data_iterator(all_batches)
        eval_data = self.reader.get_eval_data(data)

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            eval_pbar = eval_data
            for dial_idx, dialog in enumerate(eval_pbar):
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval_URURU(
                        turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    if cfg.use_true_curr_bspn: # generate act, response
                        max_len=60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
                                                    #   no_repeat_ngram_size=4
                        # turn['generated'] = self.tokenizer.decode(outputs[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length-1:]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else: # predict bspn, access db, then generate act and resp
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
                        # check DB result
                        if cfg.use_true_db_pointer:
                            # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            db = turn['db']
                        else:
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
                            db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
                        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db]).to(self.device)
                        context_length = len(inputs['context_tensor_db'][0])
                        outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
                                                    max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        generated_ar = generated_ar[context_length-1:]
                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated_ar))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                    
                    turn['resp_gen'] = decoded['resp']
                    # print(turn)
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']
                    turn['context'] = self.tokenizer.decode(inputs['context'])
                    # check DB results
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    pv_turn['labels'] = inputs['labels'] # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    # pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    # pv_turn['db'] = db
                    # pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                    # pv_turn = inputs['labels']
                    
                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))

        logging.info("inference time: {:.2f} min".format((time.time()-btm)/60))
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        # bleu, success, match,dials = self.evaluator.validation_metric(results)
        logging.info("Saving model output ot {}".format(cfg.gpt_path))
        # with open(cfg.gpt_path+cfg.model_output,"w") as f:
        #     json.dump(dials,f,indent=2)
        logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f    score: %.1f' % (
            match, success, bleu, score))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match

        return eval_results

    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated)-1
            logging.info('eos_r not in generated: ' + self.tokenizer.decode(generated))
        # eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated)-1
        
        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx+1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx+1]
            decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        # if cfg.use_true_curr_bspn:
            
        # else:  # predict bspn aspn resp
        #     eos_b_idx = generated.index(eos_b_id)
        #     eos_a_idx = generated.index(eos_a_id)
        #     decoded['bspn'] = generated[: eos_b_idx+1]
        #     decoded['aspn'] = generated[eos_b_idx+1: eos_a_idx+1]
        #     decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated)-1
        return generated[: eos_b_idx+1]

    def rule_based_bs_fix(self, context, bspn):
        context = self.tokenizer.decode(context)
        bspn = self.tokenizer.decode(bspn)
        triple_bs = paser_bs(bspn)
        flag = False
        for bs in triple_bs:
            value = bs.split()[2:]
            slot = bs.split()[1]
            for v in value:
                if v not in context and v != 'centre' and slot == 'name':
                    flag = True
                    triple_bs.remove(bs)
                    break

        bspn2 = []
        for bs in triple_bs:
            bs = bs.split()
            d = bs[0]
            sv = ' '.join(bs[1:])
            if d not in bspn2:
                bspn2.append(d)
            bspn2.append(sv)

        bspn2 = ['<sos_b>'] + bspn2 + ['<eos_b>']
        bspn2 = ' '.join(bspn2)
        if flag:
            # logging.info("Context {}".format(context))
            # logging.info("Bspn {}".format(bspn))
            # logging.info("Bspn2 {}".format(bspn2))
            # logging.info("")
        # if self.tokenizer.encode(bspn) != self.tokenizer.encode(bspn2):
        #     logging.info("Context {}".format(context))
        #     logging.info("Bspn1 {}".format(bspn))
        #     logging.info("Bspn1　ids {}".format(str(self.tokenizer.encode(bspn))))
        #     logging.info("Bspn2 {}".format(bspn2))
        #     logging.info("Bspn2　ids {}".format(str(self.tokenizer.encode(bspn2))))
            return self.tokenizer.encode(bspn2)
        else:
            return self.tokenizer.encode(bspn)
            
def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'adjust':
        parse_arg_cfg(args)
        # cfg.model_path = cfg.eval_load_path
        cfg.gpt_path = cfg.eval_load_path
    else:  # train
        parse_arg_cfg(args)
        if cfg.exp_path in ['', 'to be generated']:
            # log file path, control the factors: seed, learning_rate, batch_size, early_stop_count, weight decay...
            # cfg.exp_path = 'experiments/{}_{}_sd{}_lr{}_bs{}_sp{}_dc{}/'.format('-'.join(cfg.exp_domains),
            #                                                                     cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
            #                                                                     cfg.early_stop_count, cfg.weight_decay_count)
            cfg.exp_path = 'experiments/{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device) == 1:
            cfg.multi_gpu = False
            # torch.cuda.set_device(cfg.cuda_device[0])
            # print(cfg.cuda_device[0])
            # device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
            device = torch.device('cuda:0')
            print(device)
        else:
            pass  # multi-gpu
    else:
        device = torch.device('cpu')
        logging.info('Device: {}'.format(torch.cuda.current_device()))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Modal(device)
    os.makedirs(cfg.log_path, exist_ok=True)
    if args.mode == 'train':    # train
        if cfg.save_log:  # save cfg details.
            pass
        m.train()
    elif cfg.context_scheme=='URURU':  # test
        # m.validate_URURU()
        logging.info('Running eavl on valid')
        m.validate_URURU()
    elif cfg.context_scheme=='UBARU':
        # m.validate()
        logging.info('Running eavl on valid')
        m.validate()


#  testing:  CUDA_VISIBLE_DEVICES=0 nohup python train.py -mode test -cfg eval_load_path=experiments/all_0729_sd11_lr0.0001_bs2_ga16/epoch43_trloss0.56_gpt2/ >test.file  &


if __name__ == "__main__":
    main()
