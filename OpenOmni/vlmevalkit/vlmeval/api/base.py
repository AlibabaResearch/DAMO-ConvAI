import time
import random as rd
from abc import abstractmethod
import os.path as osp
import copy as cp
from ..smp import get_logger, parse_file


class BaseAPI:

    allowed_types = ['text', 'image']
    INTERLEAVE = True
    INSTALL_REQ = False

    def __init__(self,
                 retry=10,
                 wait=3,
                 system_prompt=None,
                 verbose=True,
                 fail_msg='Failed to obtain answer via API.',
                 **kwargs):
        """Base Class for all APIs.

        Args:
            retry (int, optional): The retry times for `generate_inner`. Defaults to 10.
            wait (int, optional): The wait time after each failed retry of `generate_inner`. Defaults to 3.
            system_prompt (str, optional): Defaults to None.
            verbose (bool, optional): Defaults to True.
            fail_msg (str, optional): The message to return when failed to obtain answer.
                Defaults to 'Failed to obtain answer via API.'.
            **kwargs: Other kwargs for `generate_inner`.
        """

        self.wait = wait
        self.retry = retry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fail_msg = fail_msg
        self.logger = get_logger('ChatAPI')

        if len(kwargs):
            self.logger.info(f'BaseAPI received the following kwargs: {kwargs}')
            self.logger.info('Will try to use them as kwargs for `generate`. ')
        self.default_kwargs = kwargs

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        """The inner function to generate the answer.

        Returns:
            tuple(int, str, str): ret_code, response, log
        """
        self.logger.warning('For APIBase, generate_inner is an abstract method. ')
        assert 0, 'generate_inner not defined'
        ret_code, answer, log = None, None, None
        # if ret_code is 0, means succeed
        return ret_code, answer, log

    def working(self):
        """If the API model is working, return True, else return False.

        Returns:
            bool: If the API model is working, return True, else return False.
        """
        retry = 3
        while retry > 0:
            ret = self.generate('hello')
            if ret is not None and ret != '' and self.fail_msg not in ret:
                return True
            retry -= 1
        return False

    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.

        Args:
            msgs: Raw input messages.

        Returns:
            str: The message type.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text', item['value']
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    def generate(self, message, **kwargs1):
        """The main function to generate the answer. Will call `generate_inner` with the preprocessed input messages.

        Args:
            message: raw input messages.

        Returns:
            str: The generated answer of the Failed Message if failed to obtain answer.
        """
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(message, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except:
                            self.logger.warning(f'Failed to parse {log} as an http response. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}:')
                    self.logger.error(err)
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    def message_to_promptimg(self, message):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        import warnings
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        elif num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<image>' for x in message])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image
