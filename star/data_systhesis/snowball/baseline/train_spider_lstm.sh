save_dir=checkpoints/spider_lstm
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=1 fairseq-train data-bin/spider.sql-en \
    --optimizer adam --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch lstm --save-dir $save_dir --save-interval 100 --max-epoch 1000
