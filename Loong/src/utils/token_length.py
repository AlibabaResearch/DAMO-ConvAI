import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))

if __name__ == "__main__":
    res = token_length("{docs}")
    print(res)