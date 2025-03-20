
from opts import init_opts, raw_dataset_process_opts, load_config_from_file


def main():

    parser = init_opts()
    raw_dataset_process_opts(parser)
    args = parser.parse_args()
    args = load_config_from_file(args)


