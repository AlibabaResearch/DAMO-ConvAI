import os
import re
import string
from collections import Counter

import json
import sacrebleu
import torch
import tqdm
from rouge import Rouge
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import transformers

transformers.logging.set_verbosity_error()

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.logger import get_logger

logger = get_logger()


def collate(batch):
    query = [item['query'] for item in batch]
    context = [json.loads(item['rerank']) for item in batch]
    label = [item['response'] for item in batch]
    return query, context, label


def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


def measure_result(result_dict):
    meters = dict()

    hypothesis_list = [
        x.replace('<extra_id_0>', '') for x in result_dict['outputs']
    ]
    hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]
    reference_list = [
        x.replace('<response>', '') for x in result_dict['targets']
    ]
    instance_num = len(reference_list)

    # F1
    f1, em = matching_evaluate(reference_list, hypothesis_list)
    meters['f1'] = f1

    # SacreBleu
    bleu_score = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num
    meters['bleu'] = bleu_score

    # Rouge-L
    rouge_func = Rouge()
    rouge_score = [
        x['rouge-l']['f']
        for x in rouge_func.get_scores(hypothesis_list, reference_list)
    ]
    rouge_score = (sum(rouge_score) / instance_num) * 100
    meters['rouge'] = rouge_score

    return meters


def train(trainer,
          total_epoches=10,
          batch_size=16,
          accumulation_steps=1,
          learning_rate=1e-4,
          warmup_ratio=0.1,
          weight_decay=0.1,
          eps=1e-06,
          loss_log_freq=40,
          clip_grad_norm=1.0):
    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    train_loader = DataLoader(
        dataset=trainer.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate)

    optimizer = prepare_optimizer(trainer.model.model, learning_rate,
                                  weight_decay, eps)
    steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = prepare_scheduler(optimizer, total_epoches,
                                  steps_per_epoch, warmup_ratio)
    best_score = 0.0
    for epoch in range(total_epoches):
        trainer.model.model.train()
        losses = []
        for index, payload in enumerate(tqdm.tqdm(train_loader)):
            query, context, label = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:128])
                for x in query
            ]
            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]
            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)
            label_ids = tokenizer.batch_encode_plus(
                list(label), padding=True, return_tensors='pt').input_ids.to(device)
            loss = model(input_ids=input_ids, labels=label_ids)[0]

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if (index + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            if (index + 1) % loss_log_freq == 0:
                logger.info(
                    f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)}'
                )
                losses = []
        if losses:
            logger.info(
                f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
            )

        meters = evaluate(trainer, batch_size=batch_size)
        total_score = sum([x for x in meters.values()])
        if total_score >= best_score:
            best_score = total_score
            model_path = os.path.join(trainer.model.model_dir,
                                      'finetuned_model.bin')
            state_dict = trainer.model.model.state_dict()
            torch.save(state_dict, model_path)
            logger.info(
                'epoch %d obtain max score: %.4f, saving model to %s' %
                (epoch, total_score, model_path))


def evaluate(trainer, batch_size=16, checkpoint_path=None):
    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=batch_size,
        collate_fn=collate)
    trainer.model.model.eval()
    with torch.no_grad():
        results = {'outputs': [], 'targets': []}
        for index, payload in enumerate(tqdm.tqdm(valid_loader)):
            query, context, label = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:128])
                for x in query
            ]
            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]
            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)

            outputs = model.generate(input_ids, num_beams=3, max_length=128, early_stopping=True,
                                     no_repeat_ngram_size=3)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label = trainer.preprocessor.generation_tokenizer.batch_decode(
                trainer.preprocessor.generation_tokenizer.batch_encode_plus(
                    label, add_special_tokens=False).input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            results['outputs'] += predictions
            results['targets'] += label
        meters = measure_result(results)
        result_path = os.path.join(trainer.model.model_dir,
                                   'evaluate_result.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(meters)
    return meters


if __name__ == '__main__':
    fr_train_dataset = MsDataset.load(
        'DAMO_ConvAI/FrDoc2BotGeneration',
        download_mode=DownloadMode.FORCE_REDOWNLOAD)
    vi_train_dataset = MsDataset.load(
        'DAMO_ConvAI/ViDoc2BotGeneration',
        download_mode=DownloadMode.FORCE_REDOWNLOAD)

    train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

    with open('all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)

    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_generation_pretrain', cache_dir='./')
    trainer = DocumentGroundedDialogGenerateTrainer(
        model=cache_path,
        train_dataset=train_dataset,
        eval_dataset=train_dataset[:100],
    )

    train(trainer, batch_size=16, accumulation_steps=1, total_epoches=10, learning_rate=1e-4)
    evaluate(trainer, checkpoint_path=os.path.join(trainer.model.model_dir,
                                                   'finetuned_model.bin'))
