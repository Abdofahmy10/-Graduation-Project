# Graduation Project: Comprehensive Analysis System ⚡🚀

## Project Overview
This project integrates multiple analytical techniques to provide a holistic evaluation system. The main components include Audio Analysis, Body Language Analysis, and Material Quality Assessment. This README provides a summary of each component and their sub-modules, as well as an outline for the presentation.

## Table of Contents
1. [Audio Analysis](#audio-analysis)
    - [Speech to Text using Whisper](#speech-to-text-using-whisper)
    - [Text Analysis](#text-analysis)
2. [Body Language Analysis](#body-language-analysis)
3. [Material Quality](#material-quality)
    - [Summary for the Presentation](#Summary-for-the-Presentation)
    - [Topic Detection and Evaluation](#Topic-Detection-and-Evaluation)     

## Audio Analysis

### Speech to Text using Whisper
The Audio Analysis component begins with converting speech to text using the Whisper model. Whisper is a state-of-the-art speech recognition system that provides accurate transcription of audio input.

### Text Analysis
Once the audio is transcribed to text, the Text Analysis sub-module processes the text to extract various features and metrics, returning a comprehensive analysis:
- **Text Length**: The total number of characters in the text.
- **Number of Sentences**: The total count of sentences.
- **Most Common Words (All)**: Frequently occurring words including common stop words.
- **Most Common Words (No Stop Words)**: Frequently occurring words excluding common stop words.
- **Longest Sentence**: The longest sentence in the text.
- **Longest Sentence Word Count**: The word count of the longest sentence.
- **Repeated Word Sentences**: Sentences with repeated words.
- **Words Per Minute (WPM)**: The speed of speech in words per minute.
- **Duration (Minutes)**: The total duration of the audio input in minutes.
- **Loudness Percentage**: The percentage of loudness throughout the audio.
- **Volume Advice**: Suggestions for optimal volume levels.
- **Fillers**: The count and types of filler words used.

### Body Language Analysis
The Body Language Analysis component uses computer vision and deep learning techniques to analyze non-verbal cues. It examines gestures, posture, facial expressions, and movements to provide a comprehensive assessment of body language. The insights generated help in understanding the effectiveness of communication and areas that need enhancement.


### Material Quality

## Summary for the Presentation
The Audio Analysis component focuses on evaluating the audio input for various parameters such as tone, pitch, speech clarity, and emotion detection. This module utilizes advanced machine learning algorithms to analyze speech patterns and provide insightful feedback.

## Topic Detection and Evaluation
Topic Detection: This sub-module identifies the primary topics discussed in the audio input. By leveraging natural language processing (NLP) techniques, it can accurately determine the subject matter and categorize it into predefined topics.
Evaluation: Post detection, the system evaluates the relevance and depth of the discussion on each topic. This helps in understanding how well the speaker covered the subject matter and provides feedback on areas for improvement.




