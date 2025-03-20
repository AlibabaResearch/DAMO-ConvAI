import logging
import os
import time
import torch
import json
import numpy as np

from tools.logger import init_logger
from metrics.metric_computor import compute_metrics
from uda import utils
from data import init_dataset
from data.dataloader import init_dataloader
from data.noise_processor import NoiseProcessor
import string
import re

class Evaluator(object):
    def __init__(self, model_name, eval_data_loader, tokenizer, num_beams,
                 eval_max_target_length, logger,
                 generated_text_dir=None,
                 num_return_sequence=1):
        super(Evaluator, self).__init__()

        self.model_name = model_name
        self.eval_data_loader = eval_data_loader
        self.tokenizer = tokenizer
        self.num_beams = num_beams

        self.eval_max_target_length = eval_max_target_length
        self.generated_text_dir = generated_text_dir

        # tokeizer
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self.logger = logger

        self.num_return_sequence = num_return_sequence

    @classmethod
    def init_evaluator(self, args, model_name, tokenizer, special_tokens,
                       generated_text_dir):
        logger = init_logger(__name__)
        task_source_prefix = getattr(args, 'eval_task_source_prefix', args.task_source_prefix)
        # task_source_prefix = "describe the following data: "
        noise_processor = None
        eval_noise_data = getattr(args, 'eval_noise_data', False)
        if eval_noise_data:
            noise_processor = NoiseProcessor.init_noise_processor(
                extra_tokens=tokenizer.additional_special_tokens,
                args=args,
                random_delete_rate=args.eval_random_delete_rate,
                noise_types=args.eval_noise_types,
                noise_type_rates=args.eval_noise_type_rates,
                noise_task_source_prefix=args.eval_noise_task_source_prefix,
                random_allocation_mask=args.random_allocation_mask)
        eval_dataset = init_dataset(
            data_dir=args.eval_file_src,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            datatype=args.datatype,
            max_inp_len=args.eval_max_source_length,
            n_example=args.eval_n_example,
            enable_uda_relative_pos=args.enable_uda_relative_pos,
            data_processor=args.data_processor,
            task_source_prefix=task_source_prefix,
            noise_processor=noise_processor
        )
        eval_dataloader = init_dataloader(
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            dist_train=False,
            shuffle=False
        )

        return self(model_name=model_name, eval_data_loader=eval_dataloader, tokenizer=tokenizer,
                    num_beams=args.num_beams,
                    eval_max_target_length=args.eval_max_target_length, logger=logger,
                    generated_text_dir=generated_text_dir,
                    num_return_sequence=1)

    # def lower_set_process(self, source_text, target_text, mode):
    #     read_file = open(source_text, "r")
    #     t = read_file.readlines()
    #     if mode == "webnlg":
    #         list0 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference0.lex", "r")
    #         list1 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference1.lex", "r")
    #         list2 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference2.lex", "r")
    #         list3 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference3.lex", "r")
    #         list4 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference4.lex", "r")
    #         list5 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference5.lex", "r")
    #         list6 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference6.lex", "r")
    #         list7 = open("/root/code/UnifiedData2TextPretrain/evaluate/webnlg/gold-all-cat-reference7.lex", "r")
    #
    #         lis0 = list0.readlines()
    #         lis1 = list1.readlines()
    #         lis2 = list2.readlines()
    #         lis3 = list3.readlines()
    #         lis4 = list4.readlines()
    #         lis5 = list5.readlines()
    #         lis6 = list6.readlines()
    #         lis7 = list7.readlines()
    #
    #
    #         k = 0
    #         new_set = []
    #         i = 0
    #         while i < len(t):
    #             if lis1[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 1
    #             elif lis2[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 2
    #             elif lis3[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 3
    #             elif lis4[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 4
    #             elif lis5[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 5
    #             elif lis6[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 6
    #             elif lis7[k] == '\n':
    #                 new_set.append(t[i])
    #                 i += 7
    #             else:
    #                 new_set.append(t[i])
    #                 i += 8
    #             k += 1
    #     elif mode == "dart":
    #         new_set = t
    #
    #     write_file = open(target_text, 'w')
    #
    #     for i in new_set:
    #         s = ''
    #         for j in i:
    #             j = j.lower()
    #             if j in string.punctuation:
    #                 s += ' ' + j + ' '
    #             else:
    #                 s += j
    #         s = re.sub(' +', ' ', s)
    #         s = s.strip() + '\n'
    #
    #         write_file.write(s.lower())

    def lower_set_process(self, test_json, generate_text, target_text, mode):
        def convert_text(text):
            text = text.lower()
            text = " ".join(re.split("(\W)", text))
            text = " ".join(text.split())
            return text + "\n"
        if mode == "webnlg":
            test = open(test_json, "r")
            test_l = test.readlines()

            generate = open(generate_text, "r")
            generate_l = generate.readlines()

            write_file = open(target_text, 'w')

            target_l = []
            k = json.loads(test_l[0])
            old_flag = k["eid"]

            for i in range(len(generate_l)):
                kk = json.loads(test_l[i])
                new_flag = kk["eid"]
                if new_flag != old_flag or i == 0:
                    target_l.append(generate_l[i])
                    # with open(target_text, "a") as f:
                    #    f.write(generate_l[i])
                    if len(target_l) <= 1779:
                        write_file.write(convert_text(generate_l[i]))
                old_flag = new_flag

        elif mode == "dart":

            read_file = open(generate_text, "r")
            t = read_file.readlines()

            write_file = open(target_text, 'w')

            for i in t:
                write_file.write(convert_text(i))

    def eval_officle_bleu(self, folder_data, pred_file, dataset,mode):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if mode == "webnlg":
            cmd_string = (
                "perl "
                + dir_path
                + "/multi-bleu.perl -lc "
                + folder_data
                + "/"
                + dataset
                + "0.lex "
                + folder_data
                + "/"
                + dataset
                + "1.lex "
                + folder_data
                + "/"
                + dataset
                + "2.lex <"
                # + folder_data
                # + "/"
                # + dataset
                # + "3.lex "
                # + folder_data
                # + "/"
                # + dataset
                # + "4.lex "
                # + folder_data
                # + "/"
                # + dataset
                # + "5.lex "
                # + folder_data
                # + "/"
                # + dataset
                # + "6.lex "
                # + folder_data
                # + "/"
                # + dataset
                # + "7.lex < "
                + pred_file
                + " > "
                + pred_file.replace("txt", "bleu")
        )
        elif mode == "dart":
            cmd_string = (
                    "perl "
                    + dir_path
                    + "/multi-bleu.perl -lc "
                    + folder_data
                    + "/"
                    + dataset
                    + "0.lex "
                    + folder_data
                    + "/"
                    + dataset
                    + "1.lex "
                    + folder_data
                    + "/"
                    + dataset
                    + "2.lex "
                    + folder_data
                    + "/"
                    + dataset
                    + "3.lex <"
                    + pred_file
                    + " > "
                    + pred_file.replace("txt", "bleu")
            )

        os.system(cmd_string)

        try:
            bleu_info = open(pred_file.replace("txt", "bleu"), "r").readlines()[0].strip()
        except:
            bleu_info = -1
        # print("@@@@",bleu_info)
        return bleu_info

    def evaluate(self, model, device, args, prefix=None):

        start_time = time.time()
        logging.info("--- Starting Evaluating ---")
        model.eval()
        generated_hyps = []
        refs = []
        total_steps = len(self.eval_data_loader)

        with torch.no_grad():
            for i, batch in enumerate(self.eval_data_loader):
                _, hyps = self.step(batch, model, device=device)
                if i % 100 == 0:
                    logging.info("- evaluate {} / {} examples, spending: {}".format(i, total_steps, "%.4f" % (
                            time.time() - start_time)))
                generated_hyps += hyps

                if len(batch['target_text']) > 0:
                    refs.extend(batch['target_text'])
        hyps = [self.tokenizer.decode(sents[0]) for sents in generated_hyps]  # only calculate

        self.logger.info("hyp: {}".format(hyps[0]))
        if len(refs) > 0:
            self.logger.info("ref: {}".format(refs[0]))

            assert len(refs) == len(hyps), "{}: {}".format(len(refs), len(hyps))

            # print("refs", refs)
            # print("hyps", hyps)
            metrics = compute_metrics(refs=refs, hyps=hyps)
            metrics['avg_length'] = np.mean([len(_.split()) for _ in hyps])

            for m, s in metrics.items():
                self.logger.info(f'{m}:  {s:.4f}')

            output_file_name = "bleu_%.4f.txt" % metrics['bleu']
        else:
            metrics = None
            output_file_name = 'no_ground_truth.txt'

        if prefix is not None:
            output_file_name = prefix + '_' + output_file_name

        if self.generated_text_dir is not None:
            output_file_src = os.path.join(self.generated_text_dir, output_file_name)

        # if 'webnlg' in args.eval_file_src:
        #     utils.write_text_file(lines=hyps[2433:], file_src=output_file_src)
        # else:
        utils.write_text_file(lines=hyps, file_src=output_file_src)
            # utils.write_text_file(lines=refs, file_src=os.path.join(self.generated_text_dir, 'val_ground_truth_3000.txt'))
        if len(refs) > 0:
            utils.write_text_file(lines=[ref[0] for ref in refs],
                                  file_src=os.path.join(self.generated_text_dir, 'ground_truth.txt'))
        metrics['officle_bleu'] = 0.0
        if (args.train_type != "pretrain") and ('webnlg' or 'dart' in args.eval_file_src):
            mode = "webnlg" if 'webnlg' in args.eval_file_src else "dart"
            output_file_name = 'ls_' + prefix + '_' + output_file_name
            ls_output_file_src = os.path.join(self.generated_text_dir, output_file_name)
            self.lower_set_process(args.eval_file_src, output_file_src, ls_output_file_src, mode = mode)

            if mode == "webnlg":
                metrics['officle_bleu'] = self.eval_officle_bleu('/root/code/UnifiedData2TextPretrain/evaluate/webnlg3.0',
                                                             ls_output_file_src, 'test_both.target_eval',mode = mode)
            elif mode == "dart":
                metrics['officle_bleu'] = self.eval_officle_bleu('/root/code/UnifiedData2TextPretrain/evaluate/dart',
                                                                 ls_output_file_src, 'all-delex-reference',
                                                                 mode=mode)
            metrics['officle_bleu'] = re.search("[0-9]{2}.[0-9]{2}", metrics['officle_bleu'])
            metrics['officle_bleu'] = float(metrics['officle_bleu'].group()) if metrics['officle_bleu'] else 0.0
            best_obleu = best_obleu if best_obleu > obleu else obleu
            print(metrics['officle_bleu'])
        else:
            metrics['officle_bleu'] = 0.0


        return metrics, output_file_name

    def step(self, batch, model, device):

        input_ids = batch['enc_inp'].to(device)
        enc_attention_mask = batch['enc_attention_mask'].to(device)

        if self.model_name == 'uda':
            struct_attention = batch['struct_attention'].to(device)
            linear_relative_position_matrix = batch['linear_relative_position_matrix']
            if linear_relative_position_matrix is not None:
                linear_relative_position_matrix = linear_relative_position_matrix.to(device)

            output_sequence = model.generate(input_ids=input_ids,
                                             attention_mask=enc_attention_mask,
                                             max_length=self.eval_max_target_length,
                                             do_sample=False,
                                             num_beams=self.num_beams,
                                             bos_token_id=self.bos_token_id,
                                             pad_token_id=self.pad_token_id,
                                             eos_token_id=self.eos_token_id,
                                             decoder_start_token_id=self.bos_token_id,
                                             num_return_sequences=self.num_return_sequence,
                                             relative_position=linear_relative_position_matrix,
                                             struct_attention=struct_attention)

        else:
            output_sequence = model.generate(input_ids=input_ids,
                                             attention_mask=enc_attention_mask,
                                             max_length=self.eval_max_target_length,
                                             do_sample=False,
                                             num_beams=self.num_beams,
                                             bos_token_id=self.bos_token_id,
                                             pad_token_id=self.pad_token_id,
                                             eos_token_id=self.eos_token_id,
                                             decoder_start_token_id=self.bos_token_id,
                                             num_return_sequences=self.num_return_sequence, )

        assert input_ids.size(0) * self.num_return_sequence == output_sequence.size(0)

        tgt_ids = output_sequence[:, 1:].view(input_ids.size(0), self.num_return_sequence, -1)
        tgt_tensor_list = [_ for _ in tgt_ids]

        tgt_ids_list = [[self.cut_seq_to_eos(_, self.eos_token_id) for _ in l.tolist()] for l in tgt_tensor_list]
        # print("tgt_ids_list:", tgt_ids_list)
        return tgt_tensor_list, tgt_ids_list

    @staticmethod
    def cut_seq_to_eos(seq_ids, eos_token_id):
        new_seq_ids = []
        for idx in seq_ids:
            if idx == eos_token_id:
                break
            new_seq_ids.append(idx)
        return new_seq_ids
