#!/usr/bin/env bash
# -*- coding:utf-8 -*-

export DEVICE=0
export model_path=uie_models

CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14lap --model ${model_path}/absa_14lap_65.25 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14res --model ${model_path}/absa_14res_74.59 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/15res --model ${model_path}/absa_15res_68.30 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/16res --model ${model_path}/absa_16res_76.57 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14lap --model ${model_path}/absa_14lap_base_63.95 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14res --model ${model_path}/absa_14res_base_73.63 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/15res --model ${model_path}/absa_15res_base_64.68 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/16res --model ${model_path}/absa_16res_base_73.23 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/entity/mrc_ace04 --model ${model_path}/ent_ace04ent_86.87 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/entity/mrc_ace05 --model ${model_path}/ent_ace05ent_85.89 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/ace05-rel --model ${model_path}/rel_ace05-rel_66.22 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/conll04 --model ${model_path}/rel_conll04_large_74.97 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/NYT --model ${model_path}/rel_nyt_93.53 --batch_size 64 --match_mode set
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/scierc --model ${model_path}/rel_scierc_large_37.05 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/event/oneie_ace05_en_event --model ${model_path}/evt_ace05evt_74.06_55.97 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/event/casie --model ${model_path}/evt_casie_69.97_61.24 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/entity/conll03 --model ${model_path}/ent_conll03_92.97 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/NYT --model ${model_path}/rel_nyt_base_92.46 --batch_size 64 --match_mode set
