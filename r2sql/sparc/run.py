"""Contains a main function for training and/or evaluating a model."""

import os
import sys

import numpy as np
import random

from parse_args import interpret_args

import data_util
from data_util import atis_data
from model.schema_interaction_model import SchemaInteractionATISModel
from logger import Logger
from model.model import ATISModel
from model_util import Metrics, evaluate_utterance_sample, evaluate_interaction_sample, \
    train_epoch_with_utterances, train_epoch_with_interactions, evaluate_using_predicted_queries

import torch

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]

def train(model, data, params):
    """ Trains a model.

    Inputs:
        model (ATISModel): The model to train.
        data (ATISData): The data that is used to train.
        params (namespace): Training parameters.
    """
    # Get the training batches.
    log = Logger(os.path.join(params.logdir, params.logfile), "w")
    num_train_original = atis_data.num_utterances(data.train_data)
    log.put("Original number of training utterances:\t"
            + str(num_train_original))

    eval_fn = evaluate_utterance_sample
    trainbatch_fn = data.get_utterance_batches
    trainsample_fn = data.get_random_utterances
    validsample_fn = data.get_all_utterances
    batch_size = params.batch_size
    if params.interaction_level:
        batch_size = 1
        eval_fn = evaluate_interaction_sample
        trainbatch_fn = data.get_interaction_batches
        trainsample_fn = data.get_random_interactions
        validsample_fn = data.get_all_interactions

    maximum_output_length = params.train_maximum_sql_length
    train_batches = trainbatch_fn(batch_size,
                                  max_output_length=maximum_output_length,
                                  randomize=not params.deterministic)

    if params.num_train >= 0:
        train_batches = train_batches[:params.num_train]

    training_sample = trainsample_fn(params.train_evaluation_size,
                                     max_output_length=maximum_output_length)
    valid_examples = validsample_fn(data.valid_data,
                                    max_output_length=maximum_output_length)

    num_train_examples = sum([len(batch) for batch in train_batches])
    num_steps_per_epoch = len(train_batches)

    log.put(
        "Actual number of used training examples:\t" +
        str(num_train_examples))
    log.put("(Shortened by output limit of " +
            str(maximum_output_length) +
            ")")
    log.put("Number of steps per epoch:\t" + str(num_steps_per_epoch))
    log.put("Batch size:\t" + str(batch_size))

    print(
        "Kept " +
        str(num_train_examples) +
        "/" +
        str(num_train_original) +
        " examples")
    print(
        "Batch size of " +
        str(batch_size) +
        " gives " +
        str(num_steps_per_epoch) +
        " steps per epoch")

    # Keeping track of things during training.
    epochs = 0
    patience = params.initial_patience
    learning_rate_coefficient = 1.
    previous_epoch_loss = float('inf')
    maximum_validation_accuracy = 0.
    maximum_string_accuracy = 0.

    countdown = int(patience)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.trainer, mode='min', )

    keep_training = True
    while keep_training:
        log.put("Epoch:\t" + str(epochs))
        model.set_dropout(params.dropout_amount)

        if not params.scheduler:
            model.set_learning_rate(learning_rate_coefficient * params.initial_learning_rate)

        # Run a training step.
        if params.interaction_level:
            epoch_loss = train_epoch_with_interactions(
                train_batches,
                params,
                model,
                randomize=not params.deterministic)
        else:
            epoch_loss = train_epoch_with_utterances(
                train_batches,
                model,
                randomize=not params.deterministic)

        log.put("train epoch loss:\t" + str(epoch_loss))

        model.set_dropout(0.)

        # Run an evaluation step on a sample of the training data.
        train_eval_results = eval_fn(training_sample,
                                     model,
                                     params.train_maximum_sql_length,
                                     name=os.path.join(params.logdir, "train-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=TRAIN_EVAL_METRICS)[0]

        for name, value in train_eval_results.items():
            log.put(
                "train final gold-passing " +
                name.name +
                ":\t" +
                "%.2f" %
                value)

        # Run an evaluation step on the validation set.
        valid_eval_results = eval_fn(valid_examples,
                                     model,
                                     params.eval_maximum_sql_length,
                                     name=os.path.join(params.logdir, "valid-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=VALID_EVAL_METRICS)[0]
        for name, value in valid_eval_results.items():
            log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)
            if name.name == "STRING_ACCURACY":
                print("vaild STRING_ACCURACY", value)
        valid_loss = valid_eval_results[Metrics.LOSS]
        valid_token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
        string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

        if params.scheduler:
            scheduler.step(valid_loss)

        if valid_loss > previous_epoch_loss:
            learning_rate_coefficient *= params.learning_rate_ratio
            log.put(
                "learning rate coefficient:\t" +
                str(learning_rate_coefficient))

        previous_epoch_loss = valid_loss
        saved = False
        
        if not saved and string_accuracy > maximum_string_accuracy:
            maximum_string_accuracy = string_accuracy
            patience = patience * params.patience_ratio
            countdown = int(patience)
            last_save_file = os.path.join(params.logdir, "save_" + str(epochs))
            model.save(last_save_file)

            log.put(
                "maximum string accuracy:\t" +
                str(maximum_string_accuracy))
            log.put("patience:\t" + str(patience))
            log.put("save file:\t" + str(last_save_file))

        if countdown <= 0:
            keep_training = False

        countdown -= 1
        log.put("countdown:\t" + str(countdown))
        log.put("")

        epochs += 1

    log.put("Finished training!")
    log.close()

    return last_save_file


def evaluate(model, data, params, last_save_file, split):
    """Evaluates a pretrained model on a dataset.

    Inputs:
        model (ATISModel): Model class.
        data (ATISData): All of the data.
        params (namespace): Parameters for the model.
        last_save_file (str): Location where the model save file is.
    """
    if last_save_file:
        model.load(last_save_file)
    else:
        if not params.save_file:
            raise ValueError(
                "Must provide a save file name if not training first.")
        model.load(params.save_file)

    filename = split

    if filename == 'dev':
        split = data.dev_data
    elif filename == 'train':
        split = data.train_data
    elif filename == 'test':
        split = data.test_data
    elif filename == 'valid':
        split = data.valid_data
    else:
        raise ValueError("Split not recognized: " + str(params.evaluate_split))

    if params.use_predicted_queries:
        filename += "_use_predicted_queries"
    else:
        filename += "_use_gold_queries"
    
    if filename == 'train':
        full_name = os.path.join(params.logdir, filename) + params.results_note
    else:
        full_name = os.path.join("results", params.save_file.split('/')[-1]) + params.results_note
    
    if params.interaction_level or params.use_predicted_queries:
        examples = data.get_all_interactions(split)
        if params.interaction_level:
            valid_eval_results, _ = evaluate_interaction_sample(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout,
                use_predicted_queries=params.use_predicted_queries,
                max_generation_length=params.eval_maximum_sql_length,
                write_results=True,
                use_gpu=True,
                compute_metrics=params.compute_metrics)
        else:
            evaluate_using_predicted_queries(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout)
    else:
        examples = data.get_all_utterances(split)
        evaluate_utterance_sample(
            examples,
            model,
            name=full_name,
            gold_forcing=False,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            max_generation_length=params.eval_maximum_sql_length,
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            write_results=True)
    

    #for name, value in valid_eval_results.items():
    #    print("valid gold-passing " + name.name + ":\t" + "%.2f" % value)


def init_env(params):
    """
    seed manually to make runs reproducible
    """
    seed = params.seed
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    """Main function that trains and/or evaluates a model."""
    params = interpret_args()
    
    # Prepare the dataset into the proper form.
    init_env(params)
    data = atis_data.ATISDataset(params)

    # Construct the model object.
    if params.interaction_level:
        model_type = SchemaInteractionATISModel
    else:
        print('not implemented')
        exit()

    model = model_type(
        params,
        data.input_vocabulary,
        data.output_vocabulary,
        data.output_vocabulary_schema,
        data.anonymizer if params.anonymize and params.anonymization_scoring else None)

    model = model.cuda()
    model.build_optim()
    """
    print('=====================Model Parameters=====================')
    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.is_cuda, param.size())
        assert param.is_cuda


    print('=====================Parameters in Optimizer==============')
    for param_group in model.trainer.param_groups:
        print(param_group.keys())
        for param in param_group['params']:
            print(param.size())

    if params.fine_tune_bert:
        print('=====================Parameters in BERT Optimizer==============')
        for param_group in model.bert_trainer.param_groups:
            print(param_group.keys())
            for param in param_group['params']:
                print(param.size())
    """
    sys.stdout.flush()

    last_save_file = ""   

    if params.train:
        last_save_file = train(model, data, params)
    if params.evaluate and 'valid' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='valid')
    if params.evaluate and 'dev' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='dev')
    if params.evaluate and 'test' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='test')

if __name__ == "__main__":
    main()
