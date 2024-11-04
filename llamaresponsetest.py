from langchain_ollama import OllamaLLM
import json

import speech_recognition as sr
import webrtcvad
import collections
import contextlib
import wave


class AudioTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.vad = webrtcvad.Vad(1)  # Set aggressiveness level (0-3)
        self.frames = collections.deque(maxlen=10)
        self.sentence = ""

    def transcribe_audio(self):
        with sr.Microphone() as source:
            print("Listening...")
            while True:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    raw_data = audio.get_raw_data()
                    raw_data, sample_rate = self.read_wave('output_resampled.wav')

                    frame_duration = 30  # in milliseconds
                    frame_size = int(16000 * (frame_duration / 1000.0))
                    frames = [raw_data[i:i + frame_size] for i in range(0, len(raw_data), frame_size)]
                    for frame in frames:
                        if len(frame) == frame_size:
                            self.frames.append(frame)                            
                    if self.is_speech_ended():
                        try:
                            text = self.recognizer.recognize_google(raw_data)
                            print(f"Recognized: {text}")
                            self.sentence += " " + text
                            if self.is_sentence_complete():
                                user_input = self.sentence.strip()
                                print(f"User Input: {user_input}")
                                self.sentence = ""
                                return user_input
                        except sr.UnknownValueError:
                            print("Could not understand audio")
                        except sr.RequestError as e:
                            print(f"Could not request results; {e}")
                except sr.WaitTimeoutError:
                    print("Listening timed out while waiting for phrase to start")
    
    def read_wave(path):    
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            print(f"Sample rate: {sample_rate}")  # Debugging line
            assert sample_rate in (8000, 16000, 32000, 48000), "Unsupported sample rate"
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def is_speech_ended(self):
        # Check for silence in the last few frames
        for frame in self.frames:
            if self.vad.is_speech(frame, 16000):
                return False
        return True

    def is_sentence_complete(self):
        # Simple heuristic to check if a sentence is complete
        return self.sentence.endswith(('.', '!', '?'))
    

class ConversationalAgent:
    def __init__(self, model_name="llama3.1"):
        self.llm = OllamaLLM(model=model_name)
        self.conversation_history = []

    def add_to_history(self, user_text, model_response):
        self.conversation_history.append({"user": user_text, "model": model_response})

    def get_conversation_history(self):
        return "\n".join([f"User: {entry['user']}\nModel: {entry['model']}" for entry in self.conversation_history])

    def get_response(self, user_text, output_file='current_output.txt'):
        # Include the conversation history in the input
        conversation_context = self.get_conversation_history()
        full_input = f"{conversation_context}\nUser: {user_text}\nModel:"

        # Get the response from the model
        response = self.llm.invoke(full_input)

        # Add the new interaction to the history
        self.add_to_history(user_text, response)

        # Save the response to a text file
        with open(output_file, 'w') as file:
            file.write(response)

        return response

    def save_conversation_history(self, file_path='conversation_history.json'):
            with open(file_path, 'w') as file:
                json.dump(self.conversation_history, file)

    def load_conversation_history(self, file_path='conversation_history.json'):
            with open(file_path, 'r') as file:
                self.conversation_history = json.load(file)

agent = ConversationalAgent()
transcriber = AudioTranscriber()

agent.load_conversation_history()

# Continuously transcribe audio and get responses
while True:
    user_input = transcriber.transcribe_audio()
    if user_input:
        response = agent.get_response(user_input)
        print(f"Model: {response}")

        # Save the updated conversation history
        agent.save_conversation_history()

user_input = "That was good. Always keep your output restricted in this manner. Now move a bit to the right"
response = agent.get_response(user_input)
print(f"Model: {response}")

# Print the entire conversation history
print("\nConversation History:")
print(agent.get_conversation_history())

agent.save_conversation_history()
