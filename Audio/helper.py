import librosa
from collections import Counter
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

def analyze_text_with_audio(text, audio_file):
    doc = nlp(text)
    text_length = len(doc.text)
    num_sentences = len(list(doc.sents))

    all_words = [token.text for token in doc]
    most_common_words_all = Counter(all_words).most_common(3)

    non_stop_words = [token.text for token in doc if token.is_alpha and not token.is_stop]
    most_common_words_no_stop = Counter(non_stop_words).most_common(3)

    longest_sentence = ""
    longest_sentence_word_count = 0
    repeated_word_sentences = 0

    for sentence in doc.sents:
        sentence_length = len([token.text for token in sentence if token.is_alpha])
        sentence_words = [token.text for token in sentence if token.is_alpha]
        word_counts = Counter(sentence_words)
        if any(count > 1 for count in word_counts.values()):
            repeated_word_sentences += 1
        if sentence_length > longest_sentence_word_count:
            longest_sentence = sentence.text
            longest_sentence_word_count = sentence_length

    y, sr = librosa.load(audio_file, sr=None)
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    duration_minutes = round(duration_seconds / 60)
    wpm = round(text_length / duration_minutes)

    def analyze_and_advise_loudness(y, min_rms=0.02, max_rms=0.15, recommended_rms=0.05):
        def rms_loudness(y):
            return np.sqrt(np.mean(np.square(y)))

        def loudness_to_percentage(rms, min_rms, max_rms):
            normalized_rms = (rms - min_rms) / (max_rms - min_rms)
            normalized_rms = np.clip(normalized_rms, 0, 1)
            return normalized_rms * 100

        rms = rms_loudness(y)
        loudness_percentage = loudness_to_percentage(rms, min_rms, max_rms)
        
        if rms > recommended_rms:
            advice = "The volume is good for the audience."
        else:
            advice = "The volume may be too low for the audience."
        
        return loudness_percentage, advice

    loudness_percentage, volume_advice = analyze_and_advise_loudness(y)

    def check_fillers(doc, filler_words = {"um", "uh", "like", "you know", "I mean", "well", "so", "actually", "basically", "literally",
                                            "I guess", "kind of", "sort of", "you know what I mean", "right?", "if you will", "at the end of the day",
                                            "to be honest", "to be fair", "in my opinion", "in my experience", "if that makes sense", "if you will", 
                                            "let me put it this way", "what I'm trying to say is", "what I'm getting at is", "the thing is", 
                                            "the thing is is that", "and stuff", "and everything", "or whatever", "or whatever you call it", 
                                            "or something", 
                                           "or something like that", "or something of the sort", "or whatnot", "or what have you"}):
        fillers = [token.text for token in doc if token.text.lower() in filler_words]
        return fillers

    fillers = check_fillers(doc)

    return {
        "text_length": text_length,
        "num_sentences": num_sentences,
        "most_common_words_all": most_common_words_all,
        "most_common_words_no_stop": most_common_words_no_stop,
        "longest_sentence": longest_sentence,
        "longest_sentence_word_count": longest_sentence_word_count,
        "repeated_word_sentences": repeated_word_sentences,
        "wpm": wpm,
        "duration_minutes": duration_minutes,
        "loudness_percentage": loudness_percentage,
        "volume_advice": volume_advice,
        "fillers": fillers,
    }