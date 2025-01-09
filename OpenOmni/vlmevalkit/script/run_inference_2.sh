export PATH=/usr/local/cuda/bin:$PATH

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
export PYTHONPATH=$(dirname $SELF_DIR):$PYTHONPATH

# gpu consumed
# fp16 17-18G
# int4 7-8G

# model to be used
# Example: MODELNAME=MiniCPM-Llama3-V-2_5
MODELNAME=$1
# datasets to be tested
# Example: DATALIST="POPE ScienceQA_TEST ChartQA_TEST"
DATALIST=$2
# test mode, all or infer
MODE=$3

echo "Starting inference with model $MODELNAME on datasets $DATALIST"
# run on multi gpus with torchrun command
# remember to run twice, the first run may fail
torchrun --nproc_per_node=1 --master_port=28881 run.py --data $DATALIST --model $MODELNAME --mode $MODE --rerun
# torchrun --nproc_per_node=4 run.py --data $DATALIST --model $MODELNAME --mode $MODE --rerun
# run on single gpu with python command
# python run.py --data $DATALIST --model $MODELNAME --verbose --mode $MODE
# python run.py --data $DATALIST --model $MODELNAME --verbose --mode $MODE
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./script/run_inference.sh LLaVA-Llama3-V-1_6_ablation_final_80k_evol "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench" all
