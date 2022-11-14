# STAR

This is the project containing source code for the paper [*STAR: SQL Guided Pre-Training for Context-dependent Text-to-SQL Parsing*]

## Create conda environment
The following commands.

Create conda environment `star`:
  - In our experiments, we use **torch==1.7.0** with CUDA version 11.0
  - We use four GeForce A-100 for our pre-trained experiments.
    
        conda create -n star python=3.6
        conda activate star
        pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

## Unzip pretraining dataset

Download and move the pretaining data file [pretrain_data.txt](https://drive.google.com/file/d/1YF7Kx0TZMyS_5BJ8GmsFXfuraiogxBID/view?usp=sharing) into the directory `datasets`.

## Training
(It may takes two days on four Tesla V100-PCIE-32GB GPU.)

        python pretain_inbatch.py

## Saving STAR model

        python save_model.py
    
Then you can get the trained model and its configuration (at least containing `model.bin` and `config.json`) under `pretrained/sss` direction.