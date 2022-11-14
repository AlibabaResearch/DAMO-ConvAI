1. Firstly, create conda environment `data_construction`:
    
        conda create -n text2sql python=3.6
        source activate text2sql
        pip install grakel
        python -c "import nltk; nltk.download('punkt')"
        cd snowball
        pip install -r requirements.txt

2. Download [raw data](https://drive.google.com/file/d/10C7MeYyZvoj8j3VIabzMeuMaM_KuQH-3/view?usp=sharing) and unzip it into the `raw_data` directory. Make sure the datasets are correctly located as:
```
data
├── database
├── tables.json
└── text_to_sql_data.json
```
        
        
3. Execute the command in the file `preprocess.ipynb` to generate three data files `alldata.json`,`logic.json`,`question_sql.json` in the `preprocessed` directory.

4. Follow the paper `Logic-Consistency Text Generation from Semantic Parses`, train a snowball model from scratch or just download our pre-trained checkpoint [snollball](https://drive.google.com/file/d/1etmQtCSzd__Pl8G1LxjB71pv0OGjzmzL/view?usp=sharing) and unzip it into the `saves/checkpoint-epoch-10.0` directory. Then run the following command to generate the `final_generation.json` file:

        cd snowball
        python eval.py

5. Run the command in the file `convert2id.ipynb` to generate final pre-train data `alltask_final.txt` in the `final_data` directory.