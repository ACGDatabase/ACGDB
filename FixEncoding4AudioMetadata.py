import os
import chardet
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.oggopus import OggOpus
from mutagen.wave import WAVE
from mutagen.easyid3 import EasyID3

def detect_and_correct_encoding(text):
    encodings_to_try = [
        'shift_jis', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'iso-8859-2', 
        'iso-8859-15', 'cp1251', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 
        'cp1257', 'cp1258', 'big5', 'gb2312', 'gbk', 'euc-kr', 'euc-jp'
    ]
    for enc in encodings_to_try:
        try:
            # Attempt to decode the text using the current encoding
            decoded_text = text.encode('latin1').decode(enc)
            # Detect if the newly decoded text is valid UTF-8
            if detect_encoding(decoded_text.encode()) == 'utf-8':
                return decoded_text
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return text  # If no valid decoding found, return the original text

def detect_encoding(text):
    result = chardet.detect(text)
    return result['encoding']

def convert_encoding_if_needed(file_path, audio):
    if 'title' in audio:
        title = audio['title'][0]
        corrected_title = detect_and_correct_encoding(title)
        if corrected_title != title:
            try:
                audio['title'] = corrected_title
                audio.save()
                print(f"Converted title encoding for {file_path} to UTF-8")
            except Exception as e:
                print(f"Failed to convert title encoding for {file_path}: {e}")

    if 'artist' in audio:
        artist = audio['artist'][0]
        corrected_artist = detect_and_correct_encoding(artist)
        if corrected_artist != artist:
            try:
                audio['artist'] = corrected_artist
                audio.save()
                print(f"Converted artist encoding for {file_path} to UTF-8")
            except Exception as e:
                print(f"Failed to convert artist encoding for {file_path}: {e}")

def process_file(file_path):
    try:
        if file_path.endswith('.mp3'):
            audio = MP3(file_path, ID3=EasyID3)
        elif file_path.endswith('.flac'):
            audio = FLAC(file_path)
        elif file_path.endswith('.wav'):
            audio = WAVE(file_path)
        elif file_path.endswith('.opus'):
            audio = OggOpus(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return

        convert_encoding_if_needed(file_path, audio)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp3', '.flac', '.wav', '.opus')):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == "__main__":
    directory = input("Enter the directory path containing audio files: ")
    process_directory(directory)
