import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer

with open('dev.json') as f_in:
    with open('input.jsonl', 'w') as f_out:
        for line in f_in.readlines():
            sample = json.loads(line)
            sample['positive'] = ''
            sample['negative'] = ''
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open('input.jsonl') as f:
    eval_dataset = [json.loads(line) for line in f.readlines()]

all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json') as f:
        all_passages += json.load(f)

cache_path = './DAMO_ConvAI/nlp_convai_retrieval_pretrain'
trainer = DocumentGroundedDialogRetrievalTrainer(
    model=cache_path,
    train_dataset=None,
    eval_dataset=eval_dataset,
    all_passages=all_passages
)

trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir,
                                 'finetuned_model.bin'))
