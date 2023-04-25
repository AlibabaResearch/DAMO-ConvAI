from apis.api import API
import json
import os
import datetime


class ModifyReminder(API):
    description = "Modify a reminder API that takes three parameters - 'token'，'content' and 'time'. " \
                  "The 'token' parameter refers to the user's token " \
                  "and the 'content' parameter refers to the description of the reminder " \
                  "and the 'time' parameter specifies the time at which the reminder " \
                  "should be triggered."
    input_parameters = {
        'token': {'type': 'str', 'description': "User's token."},
        'content': {'type': 'str', 'description': 'The content of the conference.'},
        'time': {'type': 'str', 'description': 'The time for conference. Format: %Y-%m-%d %H:%M:%S'}
    }
    output_parameters = {
        'status': {'type': 'str', 'description': 'success or failed'}
    }

    database_name = 'Reminder'

    def __init__(self, init_database=None, token_checker=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}
        self.token_checker = token_checker

    def check_api_call_correctness(self, response, groundtruth) -> bool:
        """
        Checks if the response from the API call is correct.

        Parameters:
        - response (dict): the response from the API call.
        - groundtruth (dict): the groundtruth response.

        Returns:
        - is_correct (bool): whether the response is correct.
        """
        assert response['api_name'] == groundtruth['api_name'], "The API name is not correct."
        response_content, groundtruth_content = response['input']['content'].lower().split(" "), groundtruth['input']['content'].lower().split(" ")
        content_satisfied = False
        if len(set(response_content).intersection(set(groundtruth_content))) / len(set(response_content).union(
                set(groundtruth_content))) > 0.5:
            content_satisfied = True

        if content_satisfied and response['input']['time'] == groundtruth['input']['time'] and response['output'] == \
                groundtruth['output'] and response['exception'] == groundtruth['exception']:
            return True
        else:
            return False

    def call(self, token: str, content: str, time: str) -> dict:
        """
        Calls the API with the given parameters.

        Parameters:
        - token (str) : user's token
        - content (str): the content of the remind.
        - time (datetime): the time of remind.

        Returns:
        - response (str): the statu from the API call.
        """
        input_parameters = {
            'token': token,
            'content': content,
            'time': time
        }
        try:
            status = self.modify_remind(token, content, time)
        except Exception as e:
            exception = str(e)
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': None,
                    'exception': exception}
        else:
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': status,
                    'exception': None}

    def modify_remind(self, token: str, content: str, time: str) -> str:
        """
        Modify a remind.

        Parameters:
        - token (str) : user's token
        - content (str): the content of the remind.
        - time (datetime): the time of remind.
        Returns:
        - response (str): the statu from the API call.
        """

        # Check the format of the input parameters.
        datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        if content.strip() == "":
            raise Exception('Content should not be null')
        modified = False

        username = self.token_checker.check_token(token)
        for key in self.database:
            if self.database[key]['username'] == username:
                if self.database[key]['content'] == content or self.database[key]['time'] == time:
                    self.database[key]['content'] = content
                    self.database[key]['time'] = time
                    modified = True
                    break
        if not modified:
            raise Exception(f'You have no reminder about {content} or at time : {time}')
        return "success"

