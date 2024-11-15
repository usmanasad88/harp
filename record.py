import pyaudio
import wave
import os
from pynput import keyboard
from openai import OpenAI
import whisper
from gtts import gTTS
from io import BytesIO
import re
from playsound import playsound


# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
CHUNK_DURATION = 10  # Duration of each chunk in seconds
CHUNK_FRAMES = int(RATE / CHUNK * CHUNK_DURATION)
AUDIO_FOLDER = "audio"

stop_recording = False
client = OpenAI()
#model = whisper.load_model("small")
model=whisper.load_model("turbo")
model


def on_press(key):
    global stop_recording
    stop_recording = True
    return False  # Stop listener

def record_audio():
    global stop_recording

    # Create the audio folder if it doesn't exist
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)

    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")

    chunk_index = 0

    #while not stop_recording:
    frames = []
    for _ in range(CHUNK_FRAMES):
        data = stream.read(CHUNK)
        frames.append(data)

    chunk_filename = os.path.join(AUDIO_FOLDER, f"audio_chunk_{chunk_index}.wav")
    with wave.open(chunk_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Saved {chunk_filename}")

    chunk_index += 1

    print("Finished recording")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
# Function to transcribe audio chunks using OpenAI Whisper
def transcribe_audio_chunks(audio):
    result = model.transcribe(audio, verbose=False, word_timestamps=False,)
    text=result["text"]
    language=result["language"]
    return text, language

def text_to_speech_openai(text, output_filename):
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )
    response.stream_to_file(output_filename)
    return 

#record_audio()
#print("Recording completed.")
#audio="audio/audio_chunk_0.wav"
#text, lang= transcribe_audio_chunks(audio)
#no_one_talking = not re.search(r'\b\w+\b', text)
#print("No one is talking", no_one_talking)
#print(text)
#speech_file_path = "/home/rml/harp/speech.mp3"
#text_to_speech_openai(text,speech_file_path)
#playsound(speech_file_path)

