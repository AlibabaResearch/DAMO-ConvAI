from tasks.base import BaseProbInference


class OBQAProbInferenceForMC(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "concat"

    def dataset_signature(self):
        return {
            "result": ("openbookqa", None, "validation"),
            "sample": ("openbookqa", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["question_stem"], "choices": e["choices"]["text"], "answer_idx": ord(e["answerKey"]) - ord("A")})
        return data

    def handcrafted_exemplars(self):
        prefix = ""
        ex_list = [
            ["The sun is responsible for", "plants sprouting, blooming and wilting"],
            ["When standing miles away from Mount Rushmore?", "the mountains seem smaller than in photographs"],
            ["When food is reduced in the stomach?", "nutrients are being deconstructed"],
            ["Stars are", "great balls of gas burning billions of miles away"],
        ]
        return self.build_exemplar_from_examples(prefix, ex_list)

    def exemplar_seperator(self):
        if self.prompt_version.startswith("concat"):
            return "\n\n"
        else:
            raise ValueError(f"OBQA: Not supported prompt_version: {self.prompt_version}")

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("concat"):
            with_query = query
            with_query_and_choice = f"{with_query} {choice}."
        else:
            raise ValueError(f"OBQA: Not supported prompt_version: {self.prompt_version}")

        return with_query, with_query_and_choice
