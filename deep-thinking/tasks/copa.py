from tasks.base import BaseProbInference


class COPAProbInferenceForMC(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "result": ("super_glue", "copa", "validation"),
            "sample": ("super_glue", "copa", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = (e["premise"], e["question"])
            choices = [e["choice1"], e["choice2"]]
            data.append({"query": query, "choices": choices, "answer_idx": e["label"]})
        return data

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return "\n\n"
        else:
            raise ValueError(f"COPA: Not supported prompt_version: {self.prompt_version}")

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("sp"):
            premise, question = query
            if premise.endswith("."):
                premise = premise[:-1]  # looks like a sentence

            intermediate = "because" if question == "cause" else "therefore"
            with_query = f"{premise} {intermediate}"
            with_query_and_choice = f"{with_query} {choice}"
        else:
            raise ValueError(f"COPA: Not supported prompt_version: {self.prompt_version}")
        return with_query, with_query_and_choice
