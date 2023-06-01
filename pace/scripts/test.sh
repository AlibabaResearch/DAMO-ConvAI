#photochat intent
python run.py with env_debug task_finetune_photochat_intent per_gpu_batchsize=4 test_only=True precision=32 load_path=[finetuned.ckpt]

#photochat retrieval
python run.py with env_debug task_finetune_irtr_photochat_randaug per_gpu_batchsize=4 test_only=True precision=32 load_path=[finetuned.ckpt]

#mmdial intent
python run.py with env_debug task_finetune_mmdial_intent per_gpu_batchsize=8 test_only=True precision=32 load_path=[finetuned.ckpt]

#mmconv dst
python run.py with env_debug task_finetune_mmconvdst_randaug per_gpu_batchsize=4 test_only=True precision=32 load_path=[finetuned.ckpt]

#mmconv rg
python eval_model.py with env_yzc task_finetune_rg_mmconv max_text_len=512 max_source_len=362 max_pred_len=150 vocab_size=30681 \
 special_tokens_file=../pace/datamodules/vocabs/mmconv_special_tokens3.json use_segment_ids=True load_path=[finetuned.ckpt]

#simmc dst
python eval_model.py with env_yzc task_finetune_dst_simmc2 max_text_len=512 max_source_len=412 max_pred_len=100 vocab_size=31205 \
special_tokens_file=../pace/datamodules/vocabs/simmc2_special_tokens.json use_segment_ids=True load_path=[finetuned.ckpt]

#simmc rg
python eval_model.py with env_yzc task_finetune_rg_simmc2 max_text_len=512 max_source_len=362 max_pred_len=150 vocab_size=31205 \ 
special_tokens_file=../pace/datamodules/vocabs/simmc2_special_tokens.json use_segment_ids=True data_root=[data_dir] load_path=[finetuned.ckpt]

#imagechat
python eval_model.py with env_yzc task_finetune_tr_imagechat max_text_len=80 data_root=[data_dir] load_path=[finetuned.ckpt]