#photochat intent
python run.py with env_8 task_finetune_photochat_intent per_gpu_batchsize=4 load_path=[pretrained.ckpt]

#photochat retrieval
python run.py with env_8 task_finetune_irtr_photochat_randaug per_gpu_batchsize=4 load_path=[pretrained.ckpt]

#mmdial intent
python run.py with env_8 task_finetune_mmdial_intent per_gpu_batchsize=8 load_path=[pretrained.ckpt]

#mmconv dst
python run.py with env_8 task_finetune_mmconvdst_randaug per_gpu_batchsize=4 load_path=[pretrained.ckpt]

#mmconv rg
python run.py with env_yzc task_finetune_rg_mmconv max_epoch=50 learning_rate=1e-4 end_lr=1e-6  num_gpus=8 batch_size=128 per_gpu_batchsize=16 max_text_len=512 max_source_len=362 max_pred_len=150 mlm_prob=0.25 mask_source_words=True vocab_size=30681 special_tokens_file=../pace/datamodules/vocabs/mmconv_special_tokens3.json use_segment_ids=True load_path=[pretrained.ckpt]
python run.py with env_yzc task_finetune_rg_mmconv max_epoch=120 learning_rate=1e-4 end_lr=0  num_gpus=8 batch_size=128 per_gpu_batchsize=16 max_text_len=512 max_source_len=362 max_pred_len=150 mlm_prob=0.2 mask_source_words=False vocab_size=30681 special_tokens_file=../pace/datamodules/vocabs/mmconv_special_tokens3.json use_segment_ids=True load_path=[pretrained.ckpt]

#simmc dst
python run.py with env_yzc task_finetune_dst_simmc2 max_epoch=50 learning_rate=1e-4 end_lr=1e-6  num_gpus=8 batch_size=128 per_gpu_batchsize=16 max_text_len=512 max_source_len=412 mlm_prob=0.25 mask_source_words=True vocab_size=31205 special_tokens_file=../pace/datamodules/vocabs/simmc2_special_tokens.json use_segment_ids=True load_path=[pretrained.ckpt]
python run.py with env_yzc task_finetune_dst_simmc2 max_epoch=120 learning_rate=1e-4 end_lr=0  num_gpus=8 batch_size=128 per_gpu_batchsize=16 max_text_len=512 max_source_len=412 mlm_prob=0.25 mask_source_words=False vocab_size=31205 special_tokens_file=../pace/datamodules/vocabs/simmc2_special_tokens.json use_segment_ids=True load_path=[pretrained.ckpt]

#simmc rg
python run.py with env_yzc task_finetune_rg_simmc2 max_epoch=40 learning_rate=1e-4 end_lr=1e-6 num_gpus=8 batch_size=128 per_gpu_batchsize=16 max_text_len=512 max_source_len=362 mlm_prob=0.25 mask_source_words=True vocab_size=31205 special_tokens_file=../pace/datamodules/vocabs/simmc2_special_tokens.json use_segment_ids=True load_path=[pretrained.ckpt]
python run.py with env_yzc task_finetune_rg_simmc2 max_epoch=160 learning_rate=1e-4 end_lr=0 warmup_steps=0.1 num_gpus=8 batch_size=128 per_gpu_batchsize=16 max_text_len=512 max_source_len=362 max_pred_len=150 mlm_prob=0.25 mask_source_words=False vocab_size=31205 special_tokens_file=../pace/datamodules/vocabs/simmc2_special_tokens.json use_segment_ids=True load_path=[pretrained.ckpt]

#imagechat text retrieval
python run.py with env_yzc task_finetune_tr_imagechat max_epoch=10 learning_rate=1e-4 end_lr=0 warmup_steps=0.1 num_gpus=8 batch_size=32 per_gpu_batchsize=4 draw_false_text=15 max_text_len=80 load_path=[pretrained.ckpt]