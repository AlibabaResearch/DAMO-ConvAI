from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from tools.logger import init_logger
from uda.models.modeling_uda import UdaForConditionalGeneration


def load_model(tokenizer_path,
               model_name,
               model_path, args):
    if model_name == 't5':
        tokenizer, config, model = load_t5(tokenizer_path=tokenizer_path, model_path=model_path, args=args)
    elif model_name == 'uda':
        tokenizer, config, model = load_uda(tokenizer_path=tokenizer_path, model_path=model_path, args=args)
    else:
        raise NotImplemented
    return tokenizer, config, model


def load_uda(tokenizer_path, model_path, args):
    # print("tokenizer_path", tokenizer_path)

    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)

    config = T5Config.from_pretrained(pretrained_model_name_or_path=tokenizer_path)

    config = modified_plm_default_config(args=args, config=config)

    model = UdaForConditionalGeneration.from_pretrained(model_path, config=config)

    return tokenizer, config, model


def load_t5(tokenizer_path, model_path, args):
    # print("tokenizer_path", tokenizer_path)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)

    config = T5Config.from_pretrained(pretrained_model_name_or_path=tokenizer_path)

    config = modified_plm_default_config(args=args, config=config)

    model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)

    return tokenizer, config, model


def modified_plm_default_config(args, config):
    logger = init_logger(__name__)
    if getattr(args, 'modified_default_plm_config', False):
        args_dict = {k: v for k, v in vars(args).items()}
        modified_args = {k.replace('plms_', ''): v for k, v in args_dict.items() if k.startswith('plms_')}
        for k, v in modified_args.items():
            original_value = getattr(config, k, None)
            if original_value != v:
                setattr(config, k, v)
                logger.info("--- Modified the model config {} from {} to {}".format(k, original_value, v))

    return config


