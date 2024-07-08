import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from cvHelper.utils import predict_photo, calculate_score

from pptx_helper.helper import extract_text_from_pptx, clean_text, limit_text_length, extract_text_from_pdf
from pptx_helper.classifier import classify
from pptx_helper.summary import summary

from Audio.helper import analyze_text_with_audio
from Audio.whisper import audio2text
from Audio.handle_audio import check_and_convert_to_wav

import uvicorn
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/ping")
async def ping():
    return "Hello there"

@app.post("/video")
async def process_video(file: UploadFile = File(...)):
    # Save the uploaded file locally
    video_path = "temp.mp4"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process one frame per second
        second = frame_count // frame_rate
        if frame_count % frame_rate == 0:
            body_language_class, _ = predict_photo(frame)
            results.append({
                "time": second,
                "frame_number": frame_count,
                "body_language_class": body_language_class,
            })

        frame_count += 1

    cap.release()
    os.remove(video_path)

    score = calculate_score(results)

    return {
        'score': score,
        "results": results
    }

@app.post("/predict_photo")
async def process_photo(file: UploadFile = File(...)):
    # Save the uploaded file locally
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Load the image
    image = cv2.imread(image_path)

    # Make prediction using the image
    body_language_class, _ = predict_photo(image)

    # Delete the temporary image file
    os.remove(image_path)

    return {
        "body_language_class": body_language_class
    }

@app.post("/predict_pptx")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded audio file
    doc_data = await file.read()

    doc_path = "temp" + file.filename
    with open(doc_path, "wb") as f:
        f.write(doc_data)

    extracted_text = extract_text_from_pptx(doc_path)
    cleaned_text = clean_text(extracted_text)
    limited_text = limit_text_length(cleaned_text)

    data_summary = {
        "inputs": cleaned_text,
    }

    data_classify = {
        "inputs": limited_text,
    }

    text_summary = summary(data_summary)
    text_classify = classify(data_classify)

    best_score = -1
    best_label = ""

    for item in text_classify[0]:
        label = item['label']
        score = item['score']

        if score > best_score:
            best_score = score
            best_label = label

    summurization = text_summary[0]["summary_text"]

    os.remove(doc_path)

    return {
        "label": best_label,
        "proba": best_score,
        "summurization": summurization,
    }

@app.post("/predict_pdf")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded audio file
    doc_data = await file.read()

    doc_path = "temp" + file.filename
    with open(doc_path, "wb") as f:
        f.write(doc_data)

    extracted_text = extract_text_from_pdf(doc_path)
    cleaned_text = clean_text(extracted_text)
    limited_text = limit_text_length(cleaned_text)

    data_summary = {
        "inputs": cleaned_text,
    }

    data_classify = {
        "inputs": limited_text,
    }

    text_summary = summary(data_summary)
    text_classify = classify(data_classify)

    best_score = -1
    best_label = ""

    for item in text_classify[0]:
        label = item['label']
        score = item['score']

        if score > best_score:
            best_score = score
            best_label = label

    summurization = text_summary[0]["summary_text"]

    os.remove(doc_path)

    return {
        "label": best_label,
        "proba": best_score,
        "summurization": summurization,
    }

@app.post("/predict_Audio_Analysis")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded audio file
    audio_data = await file.read()

    # Save the audio file
    audio_path = "temp_" + file.filename
    with open(audio_path, "wb") as f:
        f.write(audio_data)

    audio_patht = check_and_convert_to_wav(audio_path)

    generated_text = audio2text(audio_data)

    analysis_results = analyze_text_with_audio(generated_text, audio_patht)

    os.remove(audio_path)

    return {
        "generated_text": generated_text,
        "text_length": analysis_results["text_length"],
        "num_sentences": analysis_results["num_sentences"],
        "most_common_words_all": analysis_results["most_common_words_all"],
        "most_common_words_no_stop": analysis_results["most_common_words_no_stop"],
        "longest_sentence": analysis_results["longest_sentence"],
        "longest_sentence_word_count": analysis_results["longest_sentence_word_count"],
        "repeated_word_sentences": analysis_results["repeated_word_sentences"],
        "wpm": analysis_results["wpm"],
        "duration_minutes": analysis_results["duration_minutes"],
        "loudness_percentage": analysis_results["loudness_percentage"],
        "volume_advice": analysis_results["volume_advice"],
        "fillers": analysis_results["fillers"],
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
