. /mnt/coai-nas/yibai/anaconda3/bin/activate
conda activate soto
cd /mnt/coai-nas/yibai/LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/dpo.yaml