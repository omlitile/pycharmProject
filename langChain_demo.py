import requests
import json

url = "https://api.openai-hk.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer hk-j2hhar1000054133a617b7cd99cc774867d05b97385bd058"
}

data = {
    "max_tokens": 1200,
    "model": "gpt-4.1-mini",
    "temperature": 0.8,
    "top_p": 1,
    "presence_penalty": 1,
    "messages": [
        {
            "role": "system",
            "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
        },
        {
            "role": "user",
            "content": "你是chatGPT多少？"
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
result = response.content.decode("utf-8")

print(result)