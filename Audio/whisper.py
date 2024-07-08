import requests
import time


API_URL = "https://api-inference.huggingface.co/models/distil-whisper/distil-large-v3"
headers = {"Authorization": "Bearer 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"}



CHUNK_SIZE = 1024 * 1024  # 1MB

def wait_for_model():
    while True:
        response = requests.get(API_URL)
        status = response.json().get("status")

        if status == "ready":
            print("Model is ready for inference.")
            break

        time.sleep(2) 

def audio2text(data):
    text = ""  
    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Send each chunk separately
    for i in range(total_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = data[start:end]

        response = requests.post(API_URL, headers=headers, data=chunk)
        result = response.json().get("text")
        text += result  

    return text  