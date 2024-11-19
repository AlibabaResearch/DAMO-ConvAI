import requests, json
import openai

api_key_file = 'path/to/your/key.json'
openai.api_key = load_api_key(api_key_file)

def load_api_key(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data.get('api_key')

def request_openai_standard(model_name, messages, temperature=0.8, functions=None):
    if functions is None:
        data = {
            'model': model_name,
            'temperature': temperature,
            'messages': messages
        } 
    else:
        data = {
            'model': model_name,
            'temperature': temperature,
            'messages': messages,
            'functions': functions
        }
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", gpt_url, headers=headers, json=data)


    if response.status_code == 200:
        response = json.loads(response.text)
        return response
    else:
        print('Failed. Status code:', response.status_code)
        print('Response:', response.text)
