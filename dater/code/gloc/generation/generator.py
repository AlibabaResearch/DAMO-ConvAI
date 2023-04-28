from typing import Dict, List, Union, Tuple
import openai
import time
import numpy as np

from gloc.generation.prompt import PromptBuilder

class Generator(object):
    """
    CodeX generation wrapper
    """
    def __init__(self,args, keys=None) -> None:
        self.args = args
        self.keys = keys
        self.current_key_id = 0
        self._few_shot_prompt_cache = dict()

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            n_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        key = file_path + '_shot' + str(n_shots)
        if key in self._few_shot_prompt_cache.keys():
            return self._few_shot_prompt_cache[key]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:n_shots]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt = '\n'.join(few_shot_prompt_list)
        
        self._few_shot_prompt_cache[key] = few_shot_prompt
        return few_shot_prompt


    def build_generate_prompt(
            self,
            data_item: Dict,
            num_rows: int,
            select_type: str
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            num_rows=num_rows,
            select_type=select_type
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple],
            verbose: bool = False
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        result_idx_to_eid = []
        for p in prompts:
            result_idx_to_eid.extend([p[0]] * self.args.sampling_n)
        prompts = [p[1] for p in prompts]

        result = self._call_codex_api(
            engine=self.args.engine,
            prompt=prompts,
            max_tokens=self.args.max_generation_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.sampling_n,
            stop=self.args.stop_tokens
        )

        if verbose:
            print('\n', '*' * 20, 'Codex API Call', '*' * 20)
            for prompt in prompts:
                print(prompt)
                print('\n')
            print('- - - - - - - - - - ->>')

        # parse api results
        response_dict = dict()
        for idx, g in enumerate(result['choices']):
            try:
                text = g['text']
                logprob = sum(g['logprobs']['token_logprobs'])
                eid = result_idx_to_eid[idx]
                eid_pairs = response_dict.get(eid, None)
                if eid_pairs is None:
                    eid_pairs = []
                    response_dict[eid] = eid_pairs
                eid_pairs.append((text, logprob,np.mean(g['logprobs']['token_logprobs'])))

                if verbose:
                    print(text)

            except ValueError as e:
                if verbose:
                    print('----------- Error Msg--------')
                    print(e)
                    print(text)
                    print('-----------------------------')
                pass

        return response_dict

    def _call_codex_api(
            self,
            engine: str,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str]
    ):
        start_time = time.time()
        result = None
        while result is None:
            try:
                key = self.keys[self.current_key_id]
                self.current_key_id = (self.current_key_id + 1) % len(self.keys)
                print(f"Using openai api key: {key}")
                result = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    api_key=key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    logprobs=1
                )
                print('Openai api inference time:', time.time() - start_time)
                return result
            except Exception as e:
                print(e, 'Retry.')
                time.sleep(20)
