BASE_DIR := $(shell pwd)

.PHONY: pull-train-image
pull-train-image:
        # docker pull tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF)
        docker pull eyuansu62/graphix-text-to-sql:v2


.PHONY: pull-eval-image
pull-eval-image:
        # docker pull tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF)
        docker pull eyuansu62/graphix-text-to-sql:v2

.PHONY: train
train: 
	mkdir -p -m 777 train_db_id
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	chmod 777 data_all_in/*
	docker run \
			-it \
			--rm \
			--gpus all\
			--user 13011:13011 \
			--mount type=bind,source=$(BASE_DIR)/train_db_id,target=/train_db_id \
			--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
			--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
			--mount type=bind,source=$(BASE_DIR)/configs/train.json,target=/app/configs/train.json \
			--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
			--mount type=bind,source=$(BASE_DIR)/data_all_in/,target=/app/data_all_in \
			--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
			eyuansu62/graphix-text-to-sql:v2 \
			/bin/bash -c "CUDA_VISIBLE_DEVICES=0 python seq2seq/run_seq2seq_train.py configs/train.json"

.PHONY: pre_process
pre_process: 
	mkdir -p -m 777 train
	mkdir -p -m 777 train_data
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	chmod 777 data_all_in/*
	chmod 777 seq2seq/*
	chmod 777 data_all_in/data/*
	docker run \
			-it \
			--rm \
			--gpus all\
			--user 13011:13011 \
			--mount type=bind,source=$(BASE_DIR)/train,target=/train \
			--mount type=bind,source=$(BASE_DIR)/train_data,target=/train_data \
			--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
			--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
			--mount type=bind,source=$(BASE_DIR)/configs/train.json,target=/app/configs/train.json \
			--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
			--mount type=bind,source=$(BASE_DIR)/data_all_in,target=/app/data_all_in \
			--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
			eyuansu62/graphix-text-to-sql:v2 \
			/bin/bash -c "sh ./data_all_in/run/run_pre.sh"
		

.PHONY: eval
eval: 
	mkdir -p -m 777 eval
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	chmod 777 data_all_in/*
	chmod 777 train_db_id/*
	docker run \
			-it \
			--rm \
			--gpus all\
			--user 13011:13011 \
			--mount type=bind,source=$(BASE_DIR)/eval,target=/eval \
			--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
			--mount type=bind,source=$(BASE_DIR)/train_db_id,target=/train_db_id \
			--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
			--mount type=bind,source=$(BASE_DIR)/configs/eval.json,target=/app/configs/eval.json \
			--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
			--mount type=bind,source=$(BASE_DIR)/data_all_in/,target=/app/data_all_in \
			--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
			eyuansu62/graphix-text-to-sql:v2 \
			/bin/bash -c "CUDA_VISIBLE_DEVICES=0 python seq2seq/run_seq2seq_eval.py configs/eval.json"