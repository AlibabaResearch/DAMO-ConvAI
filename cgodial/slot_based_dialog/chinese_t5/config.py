import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        # self.vocab_path_train = os.path.join(os.path.dirname(__file__), 'data/multi-woz-processed/vocab')
        # self.vocab_path_eval = None
        self.t5_path = './t5_chinese_small'
        self.data_path = '../data/processed/'
        self.data_file = 'data_for_damd.json'
        self.val_list = '../data/risawoz/valListFile.json'
        self.test_list = '../data/risawoz/testListFile.json'
        self.train_list = '../data/risawoz/trainListFile.json'

        self.slot_value_set_path = '..//data/processed/slot_value_set.json'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # experiment settings
        self.mode = 'train'
        self.cuda = False
        self.cuda_device = [1]
        self.exp_no = ''
        self.seed = 11
        self.exp_domains = ['all']
        self.save_log = True
        self.report_interval = 10
        self.max_nl_length = 60
        self.max_span_length = 30
        self.truncated = False

        # model settings
        self.vocab_size = 3000
        self.embed_size = 50
        self.hidden_size = 100
        self.pointer_dim = 6 # fixed
        self.enc_layer_num = 1
        self.dec_layer_num = 1
        self.dropout = 0
        self.layer_norm = False
        self.skip_connect = False
        self.encoder_share = False
        self.attn_param_share = False
        self.copy_param_share = False
        self.enable_aspn = False
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = True

        # training settings
        self.lr = 1e-4
        self.label_smoothing = .0
        self.lr_decay = 0.5
        self.max_norm = 1.0
        self.batch_size = 128
        self.gradient_accumulation_steps = 1
        self.epoch_num = 60
        self.early_stop_count = 5
        self.weight_decay_count = 3
        self.teacher_force = 100
        self.multi_acts_training = False
        self.multi_act_sampling_num = 1
        self.valid_loss = 'score'

        # evaluation settings
        self.eval_load_path ='experiments/all_multi_acts_sample3_sd777_lr0.005_bs80_sp5_dc3'
        self.eval_per_domain = False
        self.use_true_pv_resp = True
        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_prev_dspn = False
        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False
        self.use_true_bspn_for_ctr_eval = False
        self.use_true_domain_for_ctr_eval = False
        self.use_true_db_pointer = False
        self.limit_bspn_vocab = False
        self.limit_aspn_vocab = False
        self.same_eval_as_cambridge = True
        self.same_eval_act_f1_as_hdsa = False
        self.aspn_decode_mode = 'greedy'  #beam, greedy, nucleur_sampling, topk_sampling
        self.beam_width = 5
        self.nbest = 5
        self.beam_diverse_param=0.2
        self.act_selection_scheme = 'high_test_act_f1'
        self.topk_num = 1
        self.nucleur_p = 0.
        self.record_mode = False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and self.mode == 'train':
            file_handler = logging.FileHandler('./log/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif self.mode == 'test':
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

