import os
import random
import pickle
import torch
from multiprocessing import Pool
from transformers import AutoTokenizer

from data.misc import count_lines


def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        tmp_dataset_reader = open("dataset-tmp-" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2**20)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()


class RawDataset(object):
    def __init__(self, args, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seed = args.seed

    def build_and_save(self, workers_num):
        """
                Build dataset from the given corpus.
                Start workers_num processes and each process deals with a part of data.
                """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class UnifiedGraphDataset(RawDataset):
    def __init__(self, args, tokenizer: AutoTokenizer):
        super(UnifiedGraphDataset, self).__init__(args, tokenizer)

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r") as f:
            while pos < start:
                f.readline()
                pos += 1

            while True:
                line = f.readline()
                pos += 1

                if pos >= end:
                    break

        dataset_writer.close()

    def build_instance(self, example):
        """

        :param example: dict()
        :return:
        """
        pass