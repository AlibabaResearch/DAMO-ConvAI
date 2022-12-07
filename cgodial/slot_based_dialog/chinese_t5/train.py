'''
This repo is built based upon https://github.com/zlinao/MinTL
Thanks for the author to share the code.
Please refer to the original repo for more details
'''

import copy
import os, random, argparse, time, logging, json, tqdm
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import pprint

from utils import RisaWOZT5Reader
from config import global_config as cfg
from transformers import (AdamW, BertTokenizer, WEIGHTS_NAME,CONFIG_NAME, get_linear_schedule_with_warmup)
from T5 import MiniT5
from torch.utils.tensorboard import SummaryWriter
from eval import RisaWOZEvaluator


class Model(object):
    def __init__(self, device):
        self.device = device
        self.evaluator = RisaWOZEvaluator()
        self.tokenizer = BertTokenizer.from_pretrained(cfg.t5_path)
        self.model = MiniT5.from_pretrained(cfg.t5_path)
        self.reader = RisaWOZT5Reader(self.tokenizer)
        # word_list = ['this', 'is', '[PAD]', '[CLS]', '[旅游景点]', '[汽车]', '[辅导班]', '[飞机]', '[医院]', '[酒店]', '[电影]', '[电脑]', '[餐厅]', '[火车]', '[电视剧]', '[天气]', '[通用]', '[request]', '[inform]',
        #              '[recommend]', '[affirm]', '[negate]', '[nooffer]', '[general]', '[bye]', '[fallback]', '[db_0]', '[db_1]', '[db_2]', '[db_3]', '[v_名称]', '[v_区域]', '[v_景点类型]',
        #              '[v_最适合人群]', '[v_消费]', '[v_是否地铁直达]', '[v_门票价格]', '[v_电话号码]', '[v_地址]', '[v_评分]', '[v_开放时间]', '[v_特点]', '[v_车型]', '[v_级别]', '[v_座位数]', '[v_车身尺寸]', '[v_厂商]',
        #              '[v_能源类型]', '[v_发动机排量]', '[v_发动机马力]', '[v_驱动方式]', '[v_综合油耗]', '[v_环保标准]', '[v_驾驶辅助影像]', '[v_巡航系统]', '[v_价格]', '[v_车系]', '[v_动力水平]',
        #              '[v_油耗水平]', '[v_倒车影像]', '[v_定速巡航]', '[v_座椅加热]', '[v_座椅通风]', '[v_所属价格区间]', '[v_班号]', '[v_难度]', '[v_科目]', '[v_年级]', '[v_校区]',
        #              '[v_上课方式]', '[v_开始日期]', '[v_结束日期]', '[v_每周]', '[v_上课时间]', '[v_下课时间]', '[v_时段]', '[v_课次]', '[v_课时]', '[v_教室地点]', '[v_教师]',
        #              '[v_课程网址]', '[v_教师网址]', '[v_出发地]', '[v_目的地]', '[v_日期]', '[v_舱位档次]', '[v_航班信息]', '[v_起飞时间]', '[v_到达时间]', '[v_票价]', '[v_准点率]', '[v_等级]', '[v_类别]', '[v_性质]',
        #              '[v_电话]', '[v_挂号时间]', '[v_门诊时间]', '[v_公交线路]', '[v_地铁可达]', '[v_地铁线路]', '[v_重点科室]', '[v_ct]', '[v_mri]', '[v_dsa]', '[v_星级]', '[v_价位]', '[v_酒店类型]', '[v_房型]', '[v_停车场]', '[v_房费]',
        #              '[v_制片国家地区]', '[v_类型]', '[v_年代]', '[v_主演]', '[v_导演]', '[v_片名]', '[v_主演名单]', '[v_具体上映时间]', '[v_片长]', '[v_豆瓣评分]', '[v_品牌]', '[v_产品类别]', '[v_分类]', '[v_内存容量]', '[v_屏幕尺寸]', '[v_cpu]', '[v_价格区间]',
        #              '[v_系列]', '[v_商品名称]', '[v_系统]', '[v_游戏性能]', '[v_cpu型号]',
        #              '[v_裸机重量]', '[v_显卡类别]', '[v_显卡型号]', '[v_特性]', '[v_色系]',
        #              '[v_待机时长]', '[v_硬盘容量]', '[v_菜系]', '[v_人均消费]', '[v_营业时间]',
        #              '[v_推荐菜]', '[v_坐席]', '[v_车次信息]', '[v_时长]', '[v_出发时间]', '[v_首播时间]', '[v_集数]', '[v_单集片长]', '[v_城市]', '[v_天气]', '[v_温度]',
        #              '[v_风力风向]', '[v_紫外线强度]', '[PAD]', '[UNK]', '<sos_u>', '<eos_u>', '<sos_b>', '<eos_b>', '<sos_db>', '<eos_db>', '<sos_a>', '<eos_a>', '<sos_r>',
        #              '<eos_r>', '<sos_d>', '<eos_d>', '<go_r>', '<go_b>', '<go_a>', '<go_d>']
        # print(self.tokenizer.encode(" ".join(word_list)))
        
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.tie_decoder()
            self.model.config.decoder_start_token_id = self.reader.sos_b_id
        print('dst_start_id:', self.model.config.decoder_start_token_id)
        self.model.to(self.device)  # single gpu
        # print(input())
        # self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        self.optim = AdamW(self.model.parameters(), lr=cfg.lr)
        self.args = cfg

        # self.evaluator = None
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log')
        else:
            self.tb_writer = None
        

    def load_model(self):
        # model_state_dict = torch.load(checkpoint)
        # start_model.load_state_dict(model_state_dict)
        self.model = type(self.model).from_pretrained(self.args.model_path)
        self.model.to(self.args.device)

    def train(self):
        btm = time.time()
        step = 0
        # log info
        _ = self.reader.get_batches('train')
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
                     set_stats['num_dials']*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size))
        
        
        self.model.train()
        # lr scheduler
        lr_lambda = lambda epoch: self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)

        for epoch in range(cfg.epoch_num):
            log_loss = 0
            log_dst = 0
            log_resp = 0
            log_cnt = 0
            sw = time.time()
            data_iterator = self.reader.get_data_iterator(self.reader.get_batches('train'))
            for iter_num, dial_batch in enumerate(data_iterator):
                py_prev = {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    # pprint.pprint(turn_batch, width=500)
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                    
                    for k in inputs:
                        if k != "turn_domain":
                            inputs[k] = inputs[k].to(cfg.device)
                    
                    # batch_size = inputs["input_ids"].shape[0]
                    # input_seq_len = inputs["input_ids"].shape[1]
                    # dst_seq_len = inputs["state_input"].shape[1]
                    # resp_seq_len = inputs["response"].shape[1]
                    # print(f"batch_size:{batch_size},seq_len:{input_seq_len}, dst:{dst_seq_len}, resp:{resp_seq_len}")
                    
                    # print(inputs['input_ids'].size())
                    # print(inputs['masks'].size())
                    # print(inputs["state_input"].size())
                    # print(inputs["state_update"].size())
                    
                    outputs = self.model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["state_input"],
                                        lm_labels=inputs["state_update"],
                                        ignore_index=self.reader.pad_id,
                                        )
                    dst_loss = outputs[0]

                    # outputs = self.model(input_ids=inputs["input_ids"],
                    #                     attention_mask=inputs["masks"],
                    #                     decoder_input_ids=inputs["response_input"],
                    #                     lm_labels=inputs["response"]
                    #                     )

                    # print(inputs["response_input"])
                    # print(inputs["response"])
                    
                    # print(inputs['input_ids_plus'].size())
                    # print(inputs['masks_plus'].size())
                    # print(inputs["response_input"].size())
                    # print(inputs["response"].size())

                    outputs = self.model(input_ids=inputs['input_ids_plus'], #skip loss and logits
                                         attention_mask=inputs["masks_plus"],
                                         decoder_input_ids=inputs["response_input"],
                                         lm_labels=inputs["response"],
                                         ignore_index=self.reader.pad_id,
                                        )
                    
                    resp_loss = outputs[0]

                    py_prev['bspn'] = copy.deepcopy(turn_batch['bspn'])
                    py_prev['resp'] = copy.deepcopy(turn_batch['resp'])

                    total_loss = (dst_loss + resp_loss) / self.args.gradient_accumulation_steps
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_norm)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    
                    step+=1
                    log_loss += float(total_loss.item())
                    log_dst +=float(dst_loss.item())
                    log_resp +=float(resp_loss.item())
                    log_cnt += 1

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [total|bspn|resp] loss: {:.2f} {:.2f} {:.2f} time: {:.1f}'.format(iter_num+1,
                                                                           log_loss/(log_cnt+ 1e-8),
                                                                           log_dst/(log_cnt+ 1e-8),log_resp/(log_cnt+ 1e-8),
                                                                           time.time()-btm))
                    if self.tb_writer:
                        self.tb_writer.add_scalar(
                                'loss', log_loss/(log_cnt+ 1e-8), step+1)
                            
            # epoch_sup_loss = log_loss/(log_cnt+ 1e-8)
            # do_test = False
            # valid_loss = self.validate(do_test=do_test)
            # logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
            #         valid_loss, (time.time()-sw)/60))

            epoch_sup_loss = log_loss / (log_cnt + 1e-8)
            logging.info(
                'epoch: %d, train loss: %.3f, total time: %.1fmin' % (epoch + 1, epoch_sup_loss,
                                                                                        (time.time() - sw) / 60))

            self.save_model(epoch, epoch_sup_loss)
    
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
        self.model.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_data_iterator(self.reader.get_batches(data))
        result_collection = {}
        ctt, hit = 0 ,0
        btm = time.time()
        for batch_num, dial_batch in enumerate(data_iterator):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(
                    turn_batch, py_prev, first_turn=first_turn)
                for k in inputs:
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(cfg.device)
                # print()
                bspn_gen, aspn_gen, resp_gen = self.model.inference(
                        reader=self.reader,
                        prev=py_prev,
                        inputs=inputs)

                turn_batch['resp_gen'] = copy.deepcopy(resp_gen)
                turn_batch['bspn_gen'] = copy.deepcopy(bspn_gen)
                turn_batch['aspn_gen'] = copy.deepcopy(aspn_gen)
                py_prev['bspn'] = copy.deepcopy(bspn_gen)
                py_prev['resp'] = copy.deepcopy(resp_gen)
                
                # for ii in range(len(bspn_gen)):
                #     ctt += 1
                #     print(' --- ')
                #     print(self.tokenizer.decode(bspn_gen[ii]))
                #     print(self.tokenizer.decode(turn_batch['bspn'][ii]))
                #     print(self.tokenizer.decode(turn_batch['update_bspn'][ii]))
                #     print(' --- ')
                #     print(self.tokenizer.decode(aspn_gen[ii]))
                #     print(self.tokenizer.decode(turn_batch['aspn'][ii]))
                #     print(' --- ')
                #     print(self.tokenizer.decode(resp_gen[ii]))
                #     print(self.tokenizer.decode(turn_batch['resp'][ii]))
                #     hit += 1
                #     input('>>>')
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))
        logging.info("Inference time: {:.2f} min".format((time.time() - btm) / 60))
        print("Inference time: {:.2f} min".format((time.time() - btm) / 60))

        results, _ = self.reader.wrap_result(result_collection)
        
        with open('eval_test_results.json', 'w') as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

        bleu, success, match = self.evaluator.validation_metric(results)

        if bleu * success * match > 0:
            score = pow(bleu * success * match, 1 / 3)
        else:
            score = 0
        logging.info('bleu: %0.2f, success: %0.2f, match: %0.2f, score: %0.2f' %
                     (bleu, success, match, score))
        print('bleu: %0.2f, success: %0.2f, match: %0.2f, score: %0.2f' %
              (bleu, success, match, score))


    def lexicalize(self, result_path,output_path):
        self.reader.relex(result_path,output_path)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True


    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt


def parse_arg_cfg(args):
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
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default="t5-small", help="t5-small, t5-base, bart-large")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--context_window", type=int, default=5, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--noupdate_dst", action='store_true', help="dont use update base DST")
    parser.add_argument("--back_bone", type=str, default="t5", help="choose t5 or bart")
    #parser.add_argument("--dst_weight", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'relex':
        parse_arg_cfg(args)
        cfg.t5_path = cfg.eval_load_path
    else:
        parse_arg_cfg(args)
        cfg.exp_path = 'experiments/cw{}_sd{}_lr{}_bs{}/'.format(
            cfg.exp_no,
                cfg.seed, args.lr, cfg.batch_size,
                args.context_window)
        logging.info('save path:', cfg.exp_path)
        if not os.path.exists(cfg.exp_path):
            os.makedirs(cfg.exp_path)
        cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
        cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
        cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
        cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    #cfg.model_parameters = m.count_params()
    logging.info(str(cfg))
    
    if cfg.cuda:
        if len(cfg.cuda_device) == 1:
            cfg.multi_gpu = False
            # torch.cuda.set_device(cfg.cuda_device[0])
            device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
        else:
            pass  # multi-gpu
    else:
        device = torch.device('cpu')
        logging.info('Device: CPU')

    cfg.device = device

    m = Model(device)
    
    if args.mode == 'train':
        m.train()
    elif args.mode == 'test':
        m.validate('test')
    elif args.mode == 'relex':
        output_path = os.path.join(args.model_path, 'generation.csv')
        m.lexicalize(cfg.result_path,output_path)


if __name__ == '__main__':
    main()
