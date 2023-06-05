import random

from tasks.base import BaseProbInference


class AGNewsProbInferenceForMC(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["World", "Sports", "Business", "Technology"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "result": ("ag_news", None, "test"),
            "sample": ("ag_news", None, "train"),
        }

    def do_load(self):
        with self._rng_context:
            full_result_data = self.do_load_part("result")
            self.raw_data_result = random.sample(full_result_data, 2000)
            self.raw_data_sample = self.do_load_part("sample")

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["text"].strip(), "choices": self.CHOICES, "answer_idx": e["label"]})
        return data

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return "\n\n"
        else:
            raise ValueError(f"AGNews: Not supported prompt_version: {self.prompt_version}")

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("sp"):
            with_query = f"Article: {query}\nCategory:"
            with_query_and_choice = f"{with_query} {choice}"
        else:
            raise ValueError(f"AGNews: Not supported prompt_version: {self.prompt_version}")
        return with_query, with_query_and_choice
