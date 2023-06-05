from tasks.base import BaseProbInference


class QASCProbInferenceForMC(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "concat"

    def dataset_signature(self):
        return {
            "result": ("qasc", None, "validation"),
            "sample": ("qasc", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["question"], "choices": e["choices"]["text"], "answer_idx": ord(e["answerKey"]) - ord("A")})
        return data

    def handcrafted_exemplars(self):
        prefix = ""
        ex_list = [
            ["What type of water formation is formed by clouds?", "beads"],
            ["What can prevent food spoilage?", "dehydrating food"],
            ["The process by which genes are passed is", "reproduction"],
            ["The stomach does what in the body?", "breaks food into nutrients"],
            ["What can cause rocks to break down?", "Water"],
        ]
        return self.build_exemplar_from_examples(prefix, ex_list)

    def exemplar_seperator(self):
        if self.prompt_version.startswith("concat"):
            return "\n\n"
        else:
            raise ValueError(f"QASC: Not supported prompt_version: {self.prompt_version}")

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("concat"):
            with_query = query
            with_query_and_choice = f"{with_query} {choice}"
        else:
            raise ValueError(f"QASC: Not supported prompt_version: {self.prompt_version}")

        return with_query, with_query_and_choice
