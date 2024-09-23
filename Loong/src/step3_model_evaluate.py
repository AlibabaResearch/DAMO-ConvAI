import os
from utils.args import parse_arguments
from utils.prompt import get_evaluate_prompts
from utils.generate import generate
from utils.util import create_path, continue_gen, logger
from utils.config import load


if __name__ == '__main__':
    args = parse_arguments()

    eval_config = load(open(f"{args.model_config_dir}/{args.eval_model}"))
    evaluate_prompts = get_evaluate_prompts(args, tag="generate_response")
    tag = "eval_response"

    if not os.path.exists(args.evaluate_output_path):
        create_path(args.evaluate_output_path)
        generate(evaluate_prompts, eval_config, args.evaluate_output_path, args.process_num_eval, tag=tag)
    else:
        if args.continue_gen:
            continue_evaluate_prompts = continue_gen(args.evaluate_output_path, evaluate_prompts, tag=tag)
            generate(continue_evaluate_prompts, eval_config, args.evaluate_output_path, args.process_num_eval,
                     tag=tag)
        else:
            logger.debug(f"Path exist: {args.evaluate_output_path}")
