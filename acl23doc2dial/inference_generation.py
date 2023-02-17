import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from train_generation import evaluate

with open('all_passages/id_to_passage.json') as f:
    id_to_passage = json.load(f)

eval_dataset = []
with open('rerank_output.jsonl') as f:
    for line in f.readlines():
        sample = json.loads(line)
        eval_dataset.append({
            'query': sample['input'],
            'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                 ensure_ascii=False),
            'response': '<response> @'
        })

cache_path = './DAMO_ConvAI/nlp_convai_generation_pretrain'
trainer = DocumentGroundedDialogGenerateTrainer(
    model=cache_path,
    train_dataset=None,
    eval_dataset=eval_dataset,
)
evaluate(trainer, checkpoint_path=os.path.join(trainer.model.model_dir,
                                               'finetuned_model.bin'))
with open(f'{cache_path}/evaluate_result.json') as f:
    predictions = json.load(f)['outputs']

with open('outputStandardFileBaseline.json', 'w') as f:
    for query, prediction in zip(eval_dataset, predictions):
        f.write(json.dumps({
            'query': query['query'],
            'response': prediction.replace('<response>','').strip()
        }, ensure_ascii=False) + '\n')
