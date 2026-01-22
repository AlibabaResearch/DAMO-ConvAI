
def robust_API_response(
        model_engine,
        system_prompt,
        user_prompt,
        flag_web_search=False,
        temperature=0.2,
        require_json=True
):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    return_response=None

    
    return return_response


