from apis.api import API
import json
import os
import datetime


class ModifyMeeting(API):
    description = "This API allows users to modify a reservation for a meeting" \
                  "Function：" \
                  "Delete user's reservation for a meeting." \
                  "Exception Handling：" \
                  "1. If the modification is successful, return a success message." \
                  "2. If the modification fails, return a corresponding error message."
    input_parameters = {
        'token': {'type': 'str', 'description': "User's token."},
        'meeting_topic': {'type': 'str', 'description': 'The title of the meeting, no more than 50 characters.'},
        'start_time': {'type': 'str',
                       'description': 'The start time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S'},
        'end_time': {'type': 'str',
                     'description': 'The end time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S'},
        'location': {'type': 'str',
                     'description': 'The location where the meeting to be held, no more than 100 characters.'},
        'attendees': {'type': 'list(str)',
                      'description': 'The attendees of the meeting, including names, positions and other information.'}
    }
    output_parameters = {
        'status': {'type': 'str', 'description': 'success or failed'}
    }

    database_name = 'Meeting'

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
        response_content, groundtruth_content = response['input']['meeting_topic'].split(" "), groundtruth['input'][
            'meeting_topic'].split(" ")
        content_satisfied = False
        if len(set(response_content).intersection(set(groundtruth_content))) / len(set(response_content).union(
                set(groundtruth_content))) > 0.5:
            content_satisfied = True

        response['input'].pop('meeting_topic')
        groundtruth['input'].pop('meeting_topic')

        if content_satisfied and response['input'] == groundtruth['input'] and response['output'] == \
                groundtruth['output'] and response['exception'] == groundtruth['exception']:
            return True
        else:
            return False

    def call(self, token: str, meeting_topic: str, start_time: str, end_time: str, location: str,
             attendees: list) -> dict:
        input_parameters = {
            'token': token,
            'meeting_topic': meeting_topic,
            'start_time': start_time,
            'end_time': end_time,
            'location': location,
            'attendees': attendees
        }
        try:
            status = self.modify_meeting(token, meeting_topic, start_time, end_time, location, attendees)
        except Exception as e:
            exception = str(e)
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': None,
                    'exception': exception}
        else:
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': status,
                    'exception': None}

    def modify_meeting(self, token: str, meeting_topic: str, start_time: str, end_time: str, location: str,
                       attendees: list) -> str:
        # Check the format of the input parameters.
        if start_time:
            datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if end_time:
            datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')


        if meeting_topic.strip() == "" and not start_time:
            raise Exception('Meeting Topic and start_time should not be null both')

        modified = False
        username = self.token_checker.check_token(token)
        for key in self.database:
            if self.database[key]['username'] == username:
                if self.database[key]['meeting_topic'] == meeting_topic or self.database[key]['start_time'] == start_time:
                    if meeting_topic:
                        self.database[key]['meeting_topic'] = meeting_topic
                    if start_time:
                        self.database[key]['start_time'] = start_time
                        self.database[key]['end_time'] = end_time
                    modified = True
                    break
        if not modified:
            if meeting_topic:
                raise Exception(f'You have no meeting about {meeting_topic}')
            if start_time:
                raise Exception(f'You have no meeting at time : {start_time}')
        return "success"
