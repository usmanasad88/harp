import speech_recognition as sr
from pynput import keyboard
import time

r = sr.Recognizer()
#r.pause_threshold = 0.5
#r.energy_threshold=10

stop_recording = False

def on_press(key):
    global stop_recording
    stop_recording = True
    return False  # Stop listener

def save_audio_chunk(audio_chunk, index):
    chunk_filename = f"audio_chunk_{index}.wav"
    with open(chunk_filename, "wb") as f:
        f.write(audio_chunk.get_wav_data())
    return chunk_filename

def input_voice():
    global stop_recording
    audio_chunks = []

    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with sr.Microphone() as source:
        index = 0
        while True:
            print("")
            audio_chunk = r.listen(source, phrase_time_limit=5)
            print("--")
            #chunk_filename = save_audio_chunk(audio_chunk, index)
            #index += 1
            #audio_chunks.append(chunk_filename)
            try:
                text = r.recognize_google(audio_chunk)
                print(text)
            except sr.UnknownValueError:
                print("-")
                continue
            except sr.RequestError as e:
                print(f"Could not request result from Google Speech Recognition service: {e}")
            

    listener.stop()

    print("Finished recording, now processing...")
    # Process the audio chunks
    try:
        combined_audio = sr.AudioData(b''.join([open(chunk, "rb").read() for chunk in audio_chunks]), source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        text = r.recognize_google(combined_audio)
        print("You said: ", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio")
        return "Sorry, could not understand the audio"
    except sr.RequestError as e:
        print(f"Could not request result from Google Speech Recognition service: {e}")
        return f"Could not request result from Google Speech Recognition service: {e}"

inputtext = input_voice()
print(f"Final recognized text: {inputtext}") 