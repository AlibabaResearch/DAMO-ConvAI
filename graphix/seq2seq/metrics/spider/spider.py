"""Spider metrics."""

from typing import Optional, Union
from seq2seq.metrics.spider.spider_test_suite import compute_test_suite_metric
from seq2seq.metrics.spider.spider_exact_match import compute_exact_match_metric
import datasets


_DESCRIPTION = """
Spider metrics.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
@misc{zhong2020semantic,
  title={Semantic Evaluation for Text-to-SQL with Distilled Test Suites}, 
  author={Ruiqi Zhong and Tao Yu and Dan Klein},
  year={2020},
  eprint={2010.02840},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
"""

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Spider(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        if self.config_name not in [
            "exact_match",
            "test_suite",
            "both",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["exact_match", "test_suite", "both"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "query": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                        "db_path": datasets.Value("string"),
                        "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                        "db_column_names": datasets.features.Sequence(
                            {
                                "table_id": datasets.Value("int32"),
                                "column_name": datasets.Value("string"),
                            }
                        ),
                        "db_foreign_keys": datasets.features.Sequence(
                            {
                                "column_id": datasets.Value("int32"),
                                "other_column_id": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            reference_urls=[_URL],
        )

    def _compute(self, predictions, references):
        if self.config_name == "exact_match" or self.config_name == "both":
            exact_match = compute_exact_match_metric(predictions, references)
        else:
            exact_match = dict()

        if self.config_name == "test_suite" or self.config_name == "both":
            test_suite = compute_test_suite_metric(predictions, references, db_dir=self.test_suite_db_dir)
        else:
            test_suite = dict()

        return {**exact_match, **test_suite}
