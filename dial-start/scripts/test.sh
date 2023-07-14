# trained model
python test.py --model model_711 --dataset 711
python test.py --model model_doc --dataset doc

# reproduce our paper result with our checkpoint
python test.py --model model_711 --dataset 711 --single_ckpt --ckpt ./model/model_711 --dataset 711
python test.py --model model_doc --dataset doc --single_ckpt --ckpt ./model/model_doc --dataset doc
