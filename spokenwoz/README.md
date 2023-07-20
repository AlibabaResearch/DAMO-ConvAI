# SpokenWOZ: A Large-Scale Speech-Text Dataset for Spoken Task-Oriented Dialogue in Multiple Domains

Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang and Yongbin Li

Arxive:  [Link](https://arxiv.org/abs/2305.13040)  

## Abstract

Task-oriented dialogue (TOD) models have made significant progress in recent years. However, previous studies primarily focus on datasets written by annotators, which has resulted in a gap between academic research and real-world spoken conversation scenarios. While several small-scale spoken TOD datasets are proposed to address robustness issues such as ASR errors, they ignore the unique challenges in spoken conversation. To tackle the limitations, we introduce SpokenWOZ, a large-scale speech-text dataset for spoken TOD, containing 8 domains, 203k turns, 5.7k dialogues and 249 hours of audios from human-to-human spoken conversations. SpokenWOZ further incorporates common spoken characteristics such as word-by-word processing and reasoning in spoken language. Based on these characteristics, we present cross-turn slot and reasoning slot detection as new challenges. We conduct experiments on various baselines, including text-modal models, newly proposed dual-modal models, and LLMs, e.g., ChatGPT. The results show that the current models still have substantial room for improvement in spoken conversation, where the most advanced dialogue state tracker only achieves 25.65% in joint goal accuracy and the SOTA end-to-end model only correctly completes the user request in 52.1% of dialogues.

## Getting Started

The data is split into training, development, and unreleased test sets. 

You can download a copy of the dataset (distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license): https://spokenwoz.github.io/SpokenWOZ-github.io/

Once you have built a model that works to your expectations on the dev set, you can submit it to [Wentao Ma](https://spokenwoz.github.io/SpokenWOZ-github.io/mawentao.mwt@alibaba-inc.com) to get official scores on a hidden test set. To preserve the integrity of test results, we do not release the test set to the public. Instead, we request you to submit your model so that we can run it on the test set for you.

## Data Structure

There are 5,700 dialogues ranging form single-domain to multi-domain in SpokenWOZ. The test sets contain 1k examples.

Dialogues with MUL in the name refers to multi-domain dialogues. Dialogues with SNG refers to single-domain dialogues. Each dialogue consists of a goal, multiple user and system utterances, dialogue state, dialogue act, corresponding audio and ASR transcription. 

The file name of the audio is consistent with the id of the dialogue, for example, the corresponding audio file for MUL0032 is MUL0032.wav.

The dialogue goal for each dialogue is recorded in the "goal" field. The dialogue goal holds the fields involved in the dialogue as well as the slots involved and the corresponding values.

The dialogue state for each dialogue is recorded in the "metadata" field in every turn the same as MultiWOZ 2.1.  The  state have two sections: semi, book. Semi refers to slots from a particular domain. Book refers to booking slots for a particular domain. The joint accuracy metrics includes ALL slots.

The dialogue act for each dialogue is recorded in the "dialogue_act" and "span_info" field in every turn:

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "dialogue_act": {
        "$act_name": [
          [
            "$slot_name",
            "$action_value"
          ]
        ]
      },
      "span_info": [
        [
          "$act_name"
          "$slot_name",
          "$action_value"
          "$start_charater_index",
          "$exclusive_end_character_index"
        ]
  }
}
```

The ASR transcription for each dialogue is recorded in the "words" field in every turn.  

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "words": [
        {
        "$word_context": "$word",
        "$begin_time": "$begintime",
        "end_time": "$endtime",
        "channel_id": "$channel",
        "word_index": "$index",
        }
  }
}
```

## Baselines

The baseline contains the main following resources:

- `LLM`: It contains source codes to convert texts to SQLs by calling APIs from LLMs, such as `text-davinci-003`, `gpt-3.5-turbo`.

- `Finetuning`: It contains the codes for supervised models, including: UBAR, GALAXY, SPACE, SPACE+TripPy, SPACE+WavLM+TripPy, SPACE+WavLM, SPACE+WavLM$_{align}$.

  - `audio`: Place the corresponding audio files in this folder.
  - `audio_npy`: This folder contains the audio processed into numpy format.
  - `trippy`: This folder contains the SPACE+TripPy and SPACE+WavLM+TripPy implementations. Meanwhile, you can easily modify it to BERT+TripPy.
  - `ubar`: This folder contains the UBAR implementations.
  - `space_baseline`: This folder contains the GALAXY, SPACE, SPACE+WavLM and SPACE+WavLM$_{align}$ implementations.

  

### LLMs

**Environment Setup**

```
pip install openai
```

#### DST

Please place the text data of SpokenWOZ in  `./LLM/dst/` 

```
python dst_data.py
(modify the openai key in the python file) sh run_chatgpt.sh / run_text003.sh
(modify the file dir in the python file) python eval_JPA.py
```

#### Response Generation

Please place the text data of SpokenWOZ in  `./LLM/response/`. Make sure you get the dst results, then place the prediction file in this folder.

```
sh chatgpt.sh
sh text_003.sh
```




### Finetuning

Fo supervised models, please kindly find more details from their original improvements.

#### SPACE+TripPy and SPACE+WavLM+TripPy 

**Environment Setup**

```
pip install -r requirement.txt
```

**Training & Evaluation**

Here `data.sh` does the processing of the audio file and processes it to`./trippy/../audio/` (will be used by subsequent models) 

Meanwhile, please place the text data of SpokenWOZ in  `./trippy/data` 

```
cd ./trippy
sh scripts/data.sh
sh scripts/train.sh
```



#### UBAR 

**Environment Setup**

```
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

**Training**

Please place the text data of SpokenWOZ in `./ubar/data/multi-woz` and move db folder and ontology.json to  `./ubar/db`

```
cd ./ubar
sh data_process.sh
sh train_dst.sh
sh train_response.sh
```

**Evaluation**

**Dialog State Tracking**

```
path='YOUR_EXPERIMENT_PATH'
python train_DST.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=True use_true_prev_resp=True use_true_db_pointer=False
```

**Policy Optimization (Act and Response Generation)**

```
path='YOUR_EXPERIMENT_PATH'

python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=True use_true_prev_aspn=False use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=True use_true_curr_aspn=False use_all_previous_context=True cuda_device=0
```

**End-to-end Modeling (Belief state, Act and Response Generation)**

```
path='YOUR_EXPERIMENT_PATH'
python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=False use_true_prev_resp=False use_true_curr_bspn=False use_true_curr_aspn=False use_all_previous_context=True cuda_device=0
```



#### GALAXY, SPACE, SPACE+WavLM and SPACE+WavLM$_{align}$

**Environment Setup**

```
pip install -r requirement.txt

```

**Training & Evaluation**

Processing of text data:

```
sh data_process.sh
sh move_data.sh
```

The baseline contains the main following resources:

- `space-3`: It contains GALAXY, SPACE.
- `space_concat`:It contains SPACE+WavLM.
- `space_word`: It contains  SPACE+WavLM$_{align}$.

For SPACE and GALAXY, you need to download the corresponding models and place them in `. /space_baseline/space_word/space/model/`

```
(Select the target model)cd space-3 or space_concat or space_word

(Optional) sh train_galaxy.sh
sh train_space.sh
```

# Citation

```
@article{si2023spokenwoz,
  title={SpokenWOZ: A Large-Scale Speech-Text Dataset for Spoken Task-Oriented Dialogue in Multiple Domains},
  author={Si, Shuzheng and Ma, Wentao and Wu, Yuchuan and Dai, Yinpei and Gao, Haoyu and Lin, Ting-En and Li, Hangyu and Yan, Rui and Huang, Fei and Li, Yongbin},
  journal={arXiv preprint arXiv:2305.13040},
  year={2023}
}
```

