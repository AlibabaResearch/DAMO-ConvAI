from tasks.base import BaseProbInference


class SST2ProbInferenceForMC(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["negative", "positive"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "result": ("glue", "sst2", "validation"),
            "sample": ("glue", "sst2", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            # print(e, flush=True)
            data.append({"query": e["sentence"].strip(), "choices": self.CHOICES, "answer_idx": e["label"]})
        return data

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return "\n\n"
        else:
            raise ValueError(f"SST2: Not supported prompt_version: {self.prompt_version}")

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("sp"):
            with_query = f"Review: {query}\nSentiment:"
            with_query_and_choice = f"{with_query} {choice}"
        else:
            raise ValueError(f"SST2: Not supported prompt_version: {self.prompt_version}")
        return with_query, with_query_and_choice
