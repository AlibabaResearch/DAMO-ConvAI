python -m torch.distributed.launch --nproc_per_node=8 ../main.py \
    --apex_level=1 \
    --audio_length=10 \
    --audio_path=/PATH/OF/YOUR/WAVLM \
    --batch_size=24 \
    --data_path /PATH/OF/YOUR/PRETRAINED/DATA \
    --epochs=5 \
    --grad_acc=4 \
    --lr=1e-4 \
    --model_save_path=/PATH/TO/SAVE/YOUR/MODEL \
    --model_name=spectra \
    --save_interval=25 \
    --text_path=/PATH/OF/YOUR/ROBERTA \
    --transcripts=/PATH/OF/DOWNLOADED/PRETRINED/DATASET/transcripts.pkl \
    > train.log 2>&1