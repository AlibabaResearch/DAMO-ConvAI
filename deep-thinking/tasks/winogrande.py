from tasks.base import BaseProbInference


class WinoGrandeProbInferenceForMC(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "result": ("winogrande", "winogrande_xs", "validation"),
            "sample": ("winogrande", "winogrande_xs", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = e["sentence"]
            choices = [e["option1"], e["option2"]]
            label = int(e["answer"]) - 1
            data.append({"query": query, "choices": choices, "answer_idx": label})
        return data

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return "\n\n"
        elif self.prompt_version.startswith("oneline"):
            return "\n\n"
        else:
            raise ValueError(f"WinoGrande: Not supported prompt_version: {self.prompt_version}")

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("sp"):
            before_under, after_under = [e.strip() for e in query.split("_")]
            with_query = before_under
            with_query_and_choice = f"{with_query} {choice} {after_under}"
        elif self.prompt_version.startswith("oneline"):
            before_under, after_under = [e.strip() for e in query.split("_")]
            with_query = ""
            with_query_and_choice = f"{before_under} {choice} {after_under}"
        else:
            raise ValueError(f"WinoGrande: Not supported prompt_version: {self.prompt_version}")

        return with_query, with_query_and_choice
