from openai import OpenAI

base_url = "http://g3.ai.qylis.com:8000/v1"

# Initialize client
client = OpenAI(
    api_key="EMPTY",
    base_url=base_url
)

# Send request
response1 = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Give me a paragraph on India."},
        ]
    }],
    stream=True,
)

print('Response in chunks.')
print('\n\n')
for sse_chunk in response1:
    content = sse_chunk.choices[0].delta.content
    print(content, end='', flush=True)
print('\n\n')
