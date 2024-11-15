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
from openai import OpenAI
import threading
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from playsound import playsound
from pynput import keyboard
from record import record_audio, transcribe_audio_chunks
from langchain_ollama import OllamaLLM

load_dotenv()
client = OpenAI()

# Global variables
stop_recording = False
transcribed_text = ""
is_recording = False

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

def text_to_speech_openai(text, output_filename):
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )
    response.stream_to_file(output_filename)
    return 


# Function to listen for keyboard events to stop recording
def stop_record_listener():
    global stop_recording
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


# Stop listener function when user presses a key
def on_press(key):
    global stop_recording
    stop_recording = True
    return False  # Stop the listener once a key is pressed

# ----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    st.set_page_config("HARP")
    st.title("--")
    # Start a conversational agent which can load and save conversation history
    # and get response to an input text
    agent = ConversationalAgent()
    
    # A textbox, in case voice is not working / or for test
    text = st.text_input("Enter your text here:")
    inputtext=text

    # Buttons for sending
    send_pressed = st.button("Send", key="send_button")

    # Button for Talk
    talk=st.button("Talk",key="talk_button")
    
    while talk:
        st.text_area(label="Response:", value="Talk Pressed", height=350)
        record_audio()
        print("Recording completed.")
        audio="audio/audio_chunk_0.wav"
        ttext, lang= transcribe_audio_chunks(audio)
        no_one_talking = not re.search(r'\b\w+\b', text)
        print("No one is talking", no_one_talking)
        print(ttext)
        talk=False

        with st.spinner("Generating..."):
            agent.load_conversation_history()
            response = agent.get_response(ttext)
            agent.save_conversation_history()
            
            # Remove * in output from LLM model
            filtered_response = remove_star(response)
            
            # Convert text to speech
            speech_file_path = "/home/mani/harp/speech.mp3"
            #text_to_speech_openai(filtered_response,speech_file_path)

            # Display response and play audio
            st.text_area(label="Response:", value=filtered_response, height=350)
            #st.audio(sound, autoplay=True)
            playsound(speech_file_path)

    
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)

    if picture:
        st.image(picture)

    if send_pressed:   
        
        with st.spinner("Generating..."):
            agent.load_conversation_history()
            response = agent.get_response(inputtext)
            agent.save_conversation_history()
            
            # Remove * in output from LLM model
            filtered_response = remove_star(response)
            
            # Convert text to speech
            speech_file_path = "/home/rml/harp/speech.mp3"
            #text_to_speech_openai(filtered_response,speech_file_path)

            # Display response and play audio
            st.text_area(label="Response:", value=filtered_response, height=350)
            #st.audio(sound, autoplay=True)
            playsound(speech_file_path)

    
if __name__ == "__main__":
    main()
