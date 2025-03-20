import sys
import os
import torch
import numpy as np

from transformers import T5Tokenizer

from opts import init_opts_for_inference
from tools.logger import init_logger
from uda.models import load_model
from evaluate.evaluator import Evaluator
from uda import utils


def inference(tokenizer_path, model_path, logger, eval_prefix, infer_generated_text_dir, inference_args):
    device = torch.device(inference_args.device)

    tokenizer, config, model = load_model(tokenizer_path=tokenizer_path,
                                          model_name=inference_args.model_name, model_path=model_path,
                                          args=inference_args)
    model.to(device)
    special_tokens = []
    if inference_args.special_token_path is not None and os.path.exists(inference_args.special_token_path):
        special_tokens = utils.read_text_file(inference_args.special_token_path)

    evaluator = Evaluator.init_evaluator(args=inference_args, model_name=inference_args.model_name, tokenizer=tokenizer,
                                         special_tokens=special_tokens,
                                         generated_text_dir=infer_generated_text_dir)

    evaluator.evaluate(model=model, device=device, args =inference_args, prefix=eval_prefix)



def main():
    args = init_opts_for_inference()
    logger = init_logger(__name__)
    #
    infer_generated_text_dir = args.infer_generated_text_dir
    if not os.path.exists(infer_generated_text_dir):
        os.mkdir(infer_generated_text_dir)

    #
    if os.path.isdir(args.checkpoint_path):
        checkpoint_names = os.listdir(args.checkpoint_path)
        checkpoints = [os.path.join(args.checkpoint_path, checkpoint_name) for checkpoint_name in checkpoint_names]
    else:
        checkpoints = [args.checkpoint_path]

    logger.info("There are {} checkpoints wait for evaluating".format(len(checkpoints)))
    for i, checkpoint in enumerate(checkpoints):
        logger.info("Evaluating for {} / {}-th checkpoint".format(i+1, len(checkpoints)))
        checkpoint_name = os.path.basename(checkpoint)
        prefix = "_".join(checkpoint_name.split('_')[:4])
        if args.file_save_prefix is not None and len(args.file_save_prefix) > 0:
            prefix = args.file_save_prefix + "_" + prefix

        inference(args.tokenizer_path, model_path=checkpoint, logger=logger, eval_prefix=prefix,
                  infer_generated_text_dir=infer_generated_text_dir,
                  inference_args=args)


if __name__ == '__main__':
    main()
