import logging


def init_logger(file_name=None):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    logger = logging.getLogger(name=file_name)
    return logger