import os
import argparse
import logging
import speech_recognition as sr
from googletrans import Translator
from langdetect import detect
import numpy as np
from hmmlearn import hmm
import re
import json

# List of supported audio file formats (including MP3)
SUPPORTED_FORMATS = ('.m4a', '.mp3', '.webm', '.mp4', '.mpga', '.wav', '.mpeg')

# Define a mapping of Hinglish words to proper Hindi
HINGLISH_TO_HINDI = {}

def load_hinglish_mapping(file_path):
    """Load the Hinglish to Hindi mapping from the provided dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            translation = data.get('translation', {})
            hinglish = translation.get('hi_ng', '')
            hindi = translation.get('en', '')
            HINGLISH_TO_HINDI[hinglish.lower()] = hindi

def is_valid_file(file_path):
    """Checks if a file exists and is a regular file."""
    return os.path.isfile(file_path)

def is_supported_format(file_path):
    """Checks if a file is of a supported audio format."""
    return os.path.splitext(file_path)[1].lower() in SUPPORTED_FORMATS

def create_transcript_file(file_path, transcript, confidence_scores=None):
    """Creates a transcript file with optional confidence scores."""
    base_filename, _ = os.path.splitext(os.path.basename(file_path))
    output_file_path = f"{base_filename}-transcript.txt"

    if os.path.exists(output_file_path):
        logging.info(f"Output file '{output_file_path}' already exists. Skipping.")
        return

    try:
        # Write transcript and confidence scores (if available) to file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"{base_filename}:\n")
            f.write(transcript)
            if confidence_scores:
                for word, confidence in zip(transcript.split(), confidence_scores):
                    f.write(f"\n  - {word} ({confidence:.2f})")

    except Exception as e:
        logging.error(f"Error while creating transcript file: {e}")

def preprocess_text(text):
    """Preprocesses Hinglish text to proper Hindi."""
    for hinglish, hindi in HINGLISH_TO_HINDI.items():
        text = re.sub(r'\b' + hinglish + r'\b', hindi, text, flags=re.IGNORECASE)
    return text

def detect_and_transcribe(file_path):
    """Detects language, preprocesses text, and transcribes audio to Hindi."""
    try:
        recognizer = sr.Recognizer()

        # Use default approach (speech_recognition) for all formats
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(audio_data, language="hi-IN")
        preprocessed_transcript = preprocess_text(transcript)
        detected_language = detect(preprocessed_transcript)
        logging.info(f"Detected language: {detected_language}")
        return detected_language, preprocessed_transcript

    except sr.UnknownValueError:
        logging.warning("Speech recognition could not understand audio.")
        return None, None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech-to-Text service: {e}")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None

def translate_text(text, source_lang):
    """Translates text to English."""
    translator = Translator()
    try:
        translated_text = translator.translate(text, src=source_lang, dest='en').text
        return translated_text
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return None

def translate_audio_to_text(file_path):
    """Transcribes audio to text and translates it to English if necessary."""
    detected_language, transcript = detect_and_transcribe(file_path)
    if transcript is None:
        return None

    # Translate transcript to English if needed
    if detected_language != 'en':  # Only translate if not already in English
        translated_text = translate_text(transcript, detected_language)
        if translated_text:
            return translated_text
    return transcript

def process_file(file_path, translate_to_english=False):
    """Processes a single audio file."""
    if not is_valid_file(file_path) or not is_supported_format(file_path):
        logging.error(f"Input file '{file_path}' does not exist, is not a file, or has an unsupported format.")
        return

    transcript = translate_audio_to_text(file_path)
    if transcript is None:
        return

    create_transcript_file(file_path, transcript)

def process_directory(directory_path, recursive, translate_to_english=False):
    """Processes audio files in a directory."""
    if not os.path.isdir(directory_path):
        logging.error(f"Directory '{directory_path}' does not exist or is not a directory.")
        return

    for root, _, files in os.walk(directory_path if recursive else [directory_path]):
        for file in files:
            file_path = os.path.join(root, file)
            if is_supported_format(file_path):
                process_file(file_path, translate_to_english)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe and translate audio files.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", dest="input_file_path", help="Path to the input audio file.")
    input_group.add_argument("-d", "--directory", dest="input_directory_path", help="Path to the directory containing audio files.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Enable recursion for processing audio files in subdirectories.")
    parser.add_argument("-t", "--translate", action="store_true", help="Translate transcripts to English after recognition.")
    parser.add_argument("-n", "--noise_reduction", action="store_true", help="Apply basic noise reduction (spectral subtraction).")
    return parser.parse_args()

def main():
    """
    Main function that processes command line arguments and calls process_file or process_directory.
    """
    args = parse_args()

    load_hinglish_mapping("F:\\Darpg hackathon\\hinglish.txt")

    if args.input_file_path:
        file_path = args.input_file_path
        process_file(file_path, args.translate)  # Call process_file for a single file

    elif args.input_directory_path:
        process_directory(args.input_directory_path, args.recursive, args.translate)  # Call process_directory for a directory

    logging.info("Processing completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
