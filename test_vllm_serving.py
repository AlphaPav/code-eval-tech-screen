#!/usr/bin/env python3
"""
test vLLM model serving
"""

import requests

port=8080
response = requests.post(
    f"http://localhost:{port}/v1/completions",
    json={
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "prompt": "def hello():",
        "max_tokens": 50,
        "temperature": 0.1
    },
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    print("API test successful!")
    print(f"Generated text: {result['choices'][0]['text']}")

else:
    print(f"API test failed: {response.status_code}")

