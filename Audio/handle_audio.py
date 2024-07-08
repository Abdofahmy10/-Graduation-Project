import os
import tempfile
import soundfile as sf

def check_and_convert_to_wav(file_path):
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".wav":
        # File is already in WAV format
        return file_path

    # Convert file to WAV
    temp_dir = tempfile.mkdtemp()
    temp_wav_path = os.path.join(temp_dir, "temp.wav")

    # Load audio data
    audio, sample_rate = sf.read(file_path)

    # Save as WAV file
    sf.write(temp_wav_path, audio, sample_rate, format="WAV")

    return temp_wav_path
