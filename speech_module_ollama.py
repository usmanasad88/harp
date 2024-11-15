import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
from PIL import Image

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import translators as ts

from langchain_ollama import OllamaLLM

load_dotenv()
r = sr.Recognizer()
r.pause_threshold=3


def input_voice():
    audio_chunks = []
    stop_recording = False

    def stop_recording_callback():
        nonlocal stop_recording
        stop_recording = True

    st.button("Stop Recording", key="stop_button", on_click=stop_recording_callback)

    with sr.Microphone() as source:
        with st.spinner("Listening..."):
            print("Listening...")
            while not stop_recording:
                audio_chunk = r.listen(source)
                # Save each chunk to a file
                chunk_filename = f"audio_chunk_{len(audio_chunks)}.wav"
                with open(chunk_filename, "wb") as f:
                    f.write(audio_chunk.get_wav_data())
                audio_chunks.append(chunk_filename)
                time.sleep(0.5)  # Add a small delay to avoid creating too many chunks

    # Process the audio chunks
    with st.spinner("Processing..."):
        try:
            # Combine all audio chunks into one AudioData instance
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



def text_to_speech(text):
    mp3_fp = BytesIO()
    tts = gTTS(text)
    tts.write_to_fp(mp3_fp)
    return mp3_fp


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



def remove_star(response):
    remove = response.replace('*', '')
    return remove


# ----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    st.set_page_config("HARP")
    st.title("--")
    agent = ConversationalAgent()      
    
    text = st.text_input("Enter your text here:")
    inputtext=text

    # Buttons for sending and recording
    send_pressed = st.button("Send", key="send_button")

    if send_pressed:
                
        with st.spinner("Generating..."):
            agent.load_conversation_history()
            response = agent.get_response(inputtext)
            agent.save_conversation_history()
            
            # Remove * in output from LLM model
            filtered_response = remove_star(response)
            
            # Convert text to speech
            sound = text_to_speech(filtered_response)
            sound.seek(0)
            
            # Display response and play audio
            st.text_area(label="Response:", value=filtered_response, height=350)
            st.audio(sound, autoplay=True)
    
if __name__ == "__main__":
    main()
