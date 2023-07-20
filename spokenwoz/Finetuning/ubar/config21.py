import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        self.gpt_path = '/data/yangyy/BERT-models/huggingface/distilgpt2/'

        self.vocab_path_train = './data/multi-woz-2.1-processed/vocab'
        self.vocab_path_eval = None
        self.data_path = './data/multi-woz-2.1-processed/'
        self.data_file = 'data_for_damd.json'
        self.dev_list = 'data/multi-woz/valListFile.json'
        self.test_list = 'data/multi-woz/testListFile.json'
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.domain_file_path = 'data/multi-woz-2.1-processed/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'
        self.multi_acts_path = 'data/multi-woz-2.1-processed/multi_act_mapping_train.json'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # experiment settings
        self.mode = 'unknown'
        self.cuda = True
        self.cuda_device = [1]
        self.exp_no = ''
        self.seed = 11
        self.exp_domains = ['all']
        self.save_log = True # tensorboard 
        self.evaluate_during_training = False # evaluate during training
        self.report_interval = 200 # 485 for bs 128
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
        self.enable_aspn = True
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = 'bsdx' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False

        # training settings
        self.lr = 5e-4
        self.warmup_steps = 2000 # gpt tbd
        self.weight_decay = 0.0 # gpt tbd
        self.gradient_accumulation_steps = 16
        self.batch_size = 2

        self.label_smoothing = .0
        self.lr_decay = 0.5
        self.epoch_num = 60
        self.early_stop_count = 5
        self.weight_decay_count = 3
        self.teacher_force = 100
        self.multi_acts_training = False
        self.multi_act_sampling_num = 1
        self.valid_loss = 'score'

        # evaluation settings
        self.eval_load_path = 'experiments/all_0729_sd11_lr0.0001_bs2_ga16/epoch43_trloss0.56_gpt2'
        self.model_output = 'model_output_e2e_FFFT_fix_bs.json'
        self.eval_per_domain = False

        ### generation setting
        self.use_true_prev_bspn = True
        self.use_true_prev_aspn = True
        self.use_true_db_pointer = False
        self.use_true_prev_resp = True

        self.use_true_curr_bspn = True
        self.use_true_curr_aspn = False
        self.use_all_previous_context = True

        self.context_scheme = 'UBARU' # UBARU or URURU
        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi
        self.log_path = 'logs2.1'
        self.low_resource = False
        ###
        
        ## dst setting
        self.fix_bs = True
        self.use_nodelex_resp = True
        self.max_context_length = 900
        ##

        self.use_true_bspn_for_ctr_eval = True        
        self.use_true_domain_for_ctr_eval = True
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

