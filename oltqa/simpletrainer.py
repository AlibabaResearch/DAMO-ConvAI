from transformers import Seq2SeqTrainer, is_torch_tpu_available, EvalPrediction
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import nltk
import datasets
import re
import os
import numpy as np
import torch
import random
from pathlib import Path
import time

# nltk.download('punkt') 
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
import warnings
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

from transformers.trainer_utils import PredictionOutput,EvalLoopOutput
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)
def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")
def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0

class QuestionAnsweringTrainer(Seq2SeqTrainer):
    def __init__(self, *args, tokenizer,eval_examples=None,answer_column_name='answers',dataset_name='squad',  **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.answer_column_name = answer_column_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.scores_per_sample = {}
        if dataset_name in ['squad1_1', "quoref", 'qaconv','ropes', 'drop', 'natural_questions_with_dpr_para', "newsqa","adversarialqa_dbert_dev","adversarialqa_dbidaf_dev","adversarialqa_droberta_dev","record_extractive"]:
            self.post_process_function = self._post_process_squad
        elif dataset_name in ['boolq', 'multirc', 'boolq_np','pubmedqa_pqal_short_ans','strategyqa']:
            self.post_process_function = self._post_process_bool_squad#self._post_process_boolq#
        elif dataset_name== 'narrativeqa_dev':
            self.post_process_function = self._post_process_narrative_qa
        elif dataset_name in ['race_string', 'openbookqa','mctest_corrected_the_separator', 'social_iqa','arc_hard','arc_easy',"dream"]:
            self.post_process_function = self._post_process_race
        elif dataset_name in ["race_c","head_qa_en_test","measuring_massive_multitask_language_understanding","reclor","quail","mcscript2","mcscript","cosmosqa","processbank_test","prost_multiple_choice_with_no_context","onestopqa_intermediate","onestopqa_elementry","onestopqa_advanced","winogrande_xl","physical_iqa","qasc","commonsenseqa"]:
            self.post_process_function = self._post_process_race
        elif dataset_name in ["squad2"]:
            self.post_process_function = self._post_process_squad_v2
        elif dataset_name in ['dream']:
            self.post_process_function = self._post_process_dream
        elif dataset_name in ['tweetqa']:
            self.post_process_function = self._post_process_bleu
        else:
            print(f"{dataset_name} has not QuestionAnsweringTrainer.post_process_function!")
            raise NotImplementedError

    def _post_process_bool_squad(
        self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ,version_2_with_negative=False):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            try:
                feature_index = feature_per_example[example_index]
            except:
                print(feature_per_example)
                print(example_index)
                print(len(feature_per_example))
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        if version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": {"text":[ex['answer']],"answer_start":[0]}} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def save_score(self,localrank):
        pass
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        #assert False
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            single_labels = torch.repeat_interleave(labels,repeats=4,dim=0)
            inputs["single_labels"]=single_labels
            inputs["train"]=True
        else:
            labels = None

        outputs,sim_loss = model(**inputs)

#
     #   print(model.encoder.)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss = loss+sim_loss

        return (loss, outputs) if return_outputs else loss    

    def _post_process_squad(
        self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ,version_2_with_negative=False):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            try:
                feature_index = feature_per_example[example_index]
            except:
                print(feature_per_example)
                print(example_index)
                print(len(feature_per_example))
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        if version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def _post_process_squad_v2(
        self, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    @classmethod
    def postprocess_text(cls,preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def _post_process_boolq(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # preds = [" ".join(pred) for pred in decoded_preds]
        preds = decoded_preds
        outputs = []

        references = []
        for pred, ref in zip(preds, examples):
            ans = ref['answer']
      #      print("ans:",ans,"pred:",pred.lower())

            references.append(1 if ans=="yes" else 0)
            if pred.lower() in ['yes', 'no']:
                outputs.append(1 if pred.lower() == 'yes' else 0)
            else:
                # generate a wrong prediction if pred is not true/false
                outputs.append(0 if ans=='yes' else 1)


        assert(len(references)==len(outputs))

        formatted_predictions = []
        return EvalPrediction(predictions=outputs, label_ids=references)
    def _post_process_bleu(self, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # preds = [" ".join(pred) for pred in decoded_preds]
        preds = [item.split() for item in decoded_preds]
        references = [[item.split() for item in exp["answer"].split("//")] for exp in examples]
        assert(len(preds)==len(references))
        return EvalPrediction(predictions=preds, label_ids=references)


    def _post_process_narrative_qa(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # preds = [" ".join(pred) for pred in decoded_preds]
        preds = decoded_preds
        references = [exp['answer'] for exp in examples]
        formatted_predictions = []
        preds, references = self.postprocess_text(preds,references)
        assert(len(preds)==len(references))
        return EvalPrediction(predictions=preds, label_ids=references)

    def _post_process_drop(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # preds = [" ".join(pred) for pred in decoded_preds]
        preds = decoded_preds
        references = [exp['answers'][0]['text'] for exp in examples]
        formatted_predictions = []
        preds, references = self.postprocess_text(preds, references)
        assert (len(preds) == len(references))
        return EvalPrediction(predictions=preds, label_ids=references)

    def _post_process_race(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [{"answer": ex['answer'], 'options':ex['options']} for ex in examples]
        assert(len(references)==len(decoded_preds))
        gold_ids , pred_ids = [],[]
        for prediction, reference in zip(decoded_preds, references):
            #             reference = json.loads(reference)
            gold = int(ord(reference['answer'].strip()) - ord('A'))
            options = reference['options']
            prediction = prediction.replace("\n", "").strip()
            options = [opt.strip() for opt in options if len(opt) > 0]
            #             print('options',options,type(options))
            #             print('prediction',prediction)
            #             print('answer',gold)
            scores = [score_string_similarity(opt, prediction) for opt in options]
            max_idx = np.argmax(scores)
            gold_ids.append(gold)
            pred_ids.append(max_idx)
            # selected_ans = chr(ord('A') + max_idx)
        # print(len(references),len(decoded_preds))
        return EvalPrediction(predictions=pred_ids,label_ids = gold_ids)

    def _post_process_openbookqa(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [{"answer": ex['answerKey'], 'options': ex['choices']['text']} for ex in examples]
        assert len(references) == len(decoded_preds)
        gold_ids, pred_ids = [], []
        for prediction, reference in zip(decoded_preds, references):
            # reference = json.loads(reference)
            gold = int(ord(reference['answer'].strip()) - ord('A'))
            options = reference['options']
            prediction = prediction.replace("\n", "").strip()
            options = [opt.strip() for opt in options if len(opt) > 0]
            scores = [score_string_similarity(opt, prediction) for opt in options]
            max_idx = np.argmax(scores)
            gold_ids.append(gold)
            pred_ids.append(max_idx)
        return EvalPrediction(predictions=pred_ids,label_ids = gold_ids)

    def _post_process_dream(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [{"answer": ex['answer'], 'options': ex['options']} for ex in examples]
        assert(len(references)==len(decoded_preds))
        gold_ids, pred_ids = [],[]
        for prediction, reference in zip(decoded_preds, references):
            gold = reference['options'].index(reference['answer'])
            options = reference['options']
            prediction = prediction.replace("\n", "").strip()
            options = [opt.strip() for opt in options if len(opt) > 0]
            scores = [score_string_similarity(opt, prediction) for opt in options]
            max_idx = np.argmax(scores)
            gold_ids.append(gold)
            pred_ids.append(max_idx)
        return EvalPrediction(predictions=pred_ids, label_ids = gold_ids)

    def evaluate(self, eval_dataset=None, eval_examples=None, tokenizer=None,ignore_keys=None, metric_key_prefix: str = "eval",
                 max_length: Optional[int] = None,num_beams: Optional[int] = None):
        self._memory_tracker.start()
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output,tokenizer)
            metrics = self.compute_metrics(eval_preds)
            if self.dataset_name=='narrativeqa':
                metrics = {key: value.mid.fmeasure * 100 for key, value in metrics.items()}
                metrics = {k: round(v, 4) for k, v in metrics.items()}
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            print(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        save_only_best = True
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        if not save_only_best:
            self.save_model(output_dir)
            if self.deepspeed:
                # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
                # config `stage3_gather_fp16_weights_on_model_save` is True
                self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            if smp.dp_rank() == 0:
                # Consolidate the state dict on all processed of dp_rank 0
                opt_state_dict = self.optimizer.state_dict()
                # Save it and the scheduler on the main process
                if self.args.should_save and not save_only_best:
                    torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
                    if self.do_grad_scaling:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed and not save_only_best:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            self.do_grad_scaling=True
            # if self.do_grad_scaling:
            #     torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
                best_dir = os.path.join(run_dir,'best-checkpoint')
                # if not os.path.exists(best_dir):
                #     os.mkdir(best_dir,exist_ok=True)
                if self.args.should_save:
                    self.save_model(best_dir)
                    self.state.save_to_json(os.path.join(best_dir, TRAINER_STATE_NAME))


        # Save the Trainer state
        if self.args.should_save and not save_only_best:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save and not save_only_best:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)




    # def predict(self, predict_dataset=None, predict_examples=None, tokenizer=None,ignore_keys=None, metric_key_prefix: str = "predict",
    #              max_length: Optional[int] = None,num_beams: Optional[int] = None):
    #     self._memory_tracker.start()
    #     self._max_length = max_length if max_length is not None else self.args.generation_max_length
    #     self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
    #     predict_dataset = self.predict_dataset if predict_dataset is None else predict_dataset
    #     predict_dataloader = self.get_eval_dataloader(predict_dataset)
    #     predict_examples = predict_examples
    #
    #     # Temporarily disable metric computation, we will do it in the loop here.
    #     compute_metrics = self.compute_metrics
    #     self.compute_metrics = None
    #     eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    #     try:
    #         output = eval_loop(
    #             predict_dataloader,
    #             description="Predict",
    #             # No point gathering the predictions if there are no metrics, otherwise we defer to
    #             # self.args.prediction_loss_only
    #             prediction_loss_only=True if compute_metrics is None else None,
    #             ignore_keys=ignore_keys,
    #         )
    #     finally:
    #         self.compute_metrics = compute_metrics
    #
    #     if self.post_process_function is not None and self.compute_metrics is not None:
    #         predict_preds = self.post_process_function(predict_examples, predict_dataset, output,tokenizer)
    #         metrics = self.compute_metrics(predict_preds)
    #         if self.dataset_name=='narrativeqa':
    #             metrics = {key: value.mid.fmeasure * 100 for key, value in metrics.items()}
    #             metrics = {k: round(v, 4) for k, v in metrics.items()}
    #         # Prefix all keys with metric_key_prefix + '_'
    #         for key in list(metrics.keys()):
    #             if not key.startswith(f"{metric_key_prefix}_"):
    #                 metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    #
    #         self.log(metrics)
    #     else:
    #         metrics = {}
    #
    #     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
    #     self._memory_tracker.stop_and_update_metrics(metrics)
    #     return metrics
