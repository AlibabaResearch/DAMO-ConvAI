
# DialogueCSE

Code for the paper *DialogueCSE: Dialogue-based Contrastive Learning of Sentence Embeddings*

---
## Training & Evaluation

Steps to train DialogueCSE:
1. Put dialogues in `data/session.txt`. The format should be `{session_id}\t{role}\t{text}\n`.
2. Run `python data/data_generator.py` to generate the training data.
3. Run `sh run_train.sh` to train DialogueCSE.
4. Run `sh eval/batch_test_cmd.sh {ecd|jddc}` to evaluate the model.


The argument `bert_init_dir` in `run_train.sh` refers to a pre-trained BERT model, the parameters of which could be either the original version or that with continued pre-training on the training data of DialogueCSE.
The latter produces the best performance.


To conduct continued pre-training, please refer to the standard [BERT codebase](https://github.com/google-research/bert/blob/master/run_pretraining.py).

---
## Datasets

Download evaluation data from [Google Drive](https://drive.google.com/file/d/1gPCJ1H0A60CG88G_EHEbACmPz6cIei-_/view?usp=sharing)
and move them to `dialogue-cse/dataset`

1. JDDC: JDDC is an open-source dataset released by JD AI [1].
2. ECD: ECD is released in [2] . We have been granted for releasing our evaluation data which is derived from the original ECD dataset.
3. MDC: The license of MDC does not support secondary distribution and we are in communication with relevant parties.


[1] https://jddc.jd.com/2019/jddc

[2] Modeling Multi-turn Conversation with Deep Utterance Aggregation.