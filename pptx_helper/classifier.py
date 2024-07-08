import requests

API_URL = "https://api-inference.huggingface.co/models/Ahmed235/roberta_classification"
headers = {"Authorization": "Bearer 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"}

def classify(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
