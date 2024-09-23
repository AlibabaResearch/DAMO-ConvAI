import json
import random
import os
from utils.args import parse_arguments
from utils.prompt import get_generate_prompts
from utils.util import count_lines, logger


if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    logger.debug(f"args: {args}")
    ## step1
    if not os.path.exists(args.output_process_path) or (args.debug_num > 0 and count_lines(args.output_process_path) != args.debug_num) or (args.debug_num < 0 and count_lines(args.output_process_path) != count_lines(args.input_path)):
        generate_prompts = get_generate_prompts(args)

        with open(args.output_process_path, 'w') as f:
            for p in generate_prompts:
                f.write(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + "\n")
    else:
        logger.debug(f"Path exist: {args.output_process_path}")
