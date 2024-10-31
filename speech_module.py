import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
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
from pygame import mixer
import translators as ts

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def get_pdf_text(pdf_docs):
    text = ""
    # Open and read PDF text
    with open(pdf_docs, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are HARP, Humanoid Assistant Robot. "I am a Humanoid Assistant Robot. I assist 
    people with their queries". Keep your answer clear and concise and use sentences, not bullet points. Respond in a 
    friendly and natural manner. If the answer is not in the provided context, just say "Sorry, I couldn't find any 
    specific data on that. Can you specify which professor or topic you are asking about?"

    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
    response = response["output_text"]
    return response


def input_voice(selection):
    if selection == "اردو":
        lang_selected = "ur-PK"
    elif selection == "English":
        lang_selected = "en-PK"

    r = sr.Recognizer()
    with st.spinner("Listening..."):
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source, 10, 5)
    with st.spinner("Processing..."):
        try:
            text = r.recognize_google(audio, language=lang_selected)
            print("You said: ", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request result from Google Speech Recognition service: {e}")


def text_to_speech(text, selection):
    if selection == "اردو":
        lang_selected = "ur"
    elif selection == "English":
        lang_selected = "en"
    mp3_fp = BytesIO()
    tts = gTTS(text, lang=lang_selected)
    tts.write_to_fp(mp3_fp)
    return mp3_fp


def get_gemini_response(user_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(user_text)
    result = response.text

    return result


def get_gemini_image_response(input_text, image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)

    result = response.text
    return result


def remove_star(response):
    remove = response.replace('*', '')
    return remove


# ----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    st.set_page_config("HARP")
    st.title("HARP BOT/ہارپ بوٹ")

    with st.sidebar:
        clicked = option_menu(
            menu_title="Main Menu/مین مینو",
            options=["ChatBot/چیٹ بوٹ", "Image ChatBot/تصویر چیٹ بوٹ"],
            icons=["chat", "image"],
            menu_icon="robot",
            default_index=0,
        )

    if clicked == "ChatBot/چیٹ بوٹ":
        language_list = ["English", "اردو"]
        selection = st.selectbox("Select Language/زبان منتخب کریں", language_list)
        if selection == "اردو":
            button1_text = "مجھ سے میکاٹرونکس ڈیپارٹمنٹ/ہارپ کے بارے میں پوچھیں"
            button2_text = "مجھ سے کچھ پوچھیں"
        elif selection == "English":
            button1_text = "Ask me about Mechatronics Dept/HARP"
            button2_text = "Ask me something"
        col1, col2 = st.columns(2)
        with col1:
            button1 = st.button(button1_text)
        with col2:
            button2 = st.button(button2_text)

        if button1:
            text = input_voice(selection)
            with st.spinner("Generating..."):
                response = user_input(text)
                # Remove * in output from LLM model
                filtered_response = remove_star(response)
                sound = text_to_speech(filtered_response, selection)
                sound.seek(0)
                st.text_area(label="Response:", value=filtered_response, height=350)
                st.audio(sound, autoplay=True)
                st.download_button(
                    label="Download Speech",
                    data=sound,
                    file_name="speech.mp3",
                    mime="audio/mp3"
                )

        if button2:
            text = input_voice(selection)
            with st.spinner("Generating..."):
                response = get_gemini_response(text)
                # Remove * in output from LLM model
                filtered_response = remove_star(response)
                sound = text_to_speech(filtered_response, selection)
                sound.seek(0)
                st.text_area(label="Response:", value=filtered_response, height=350)
                st.audio(sound, autoplay=True)
                st.download_button(
                    label="Download Speech",
                    data=sound,
                    file_name="speech.mp3",
                    mime="audio/mp3"
                )

        pdf_docs = "NER-Context-v2.pdf"

        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    elif clicked == "Image ChatBot/تصویر چیٹ بوٹ":
        language_list = ["English", "اردو"]
        selection = st.selectbox("Select Language/زبان منتخب کریں", language_list)
        if selection == "اردو":
            button3_text = "مجھ سے تصویر کے بارے میں پوچھیں"
        elif selection == "English":
            button3_text = "Ask me about the picture"

        # Upload image file
        uploaded_file = st.file_uploader("Upload image/تصویر اپ لوڈ کریں", type=["jpg", "jpeg", "png"])
        image = ""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image/تصویر اپ لوڈ کریں", use_column_width=True)

        submit = st.button(button3_text)
        if submit:
            text = input_voice(selection)
            with st.spinner("Generating..."):
                response = get_gemini_image_response(text, image)
                # Remove * in output from LLM model
                filtered_response = remove_star(response)
                sound = text_to_speech(filtered_response, selection)
                sound.seek(0)
                st.text_area(label="Response:", value=filtered_response, height=350)
                st.audio(sound, autoplay=True)
                st.download_button(
                    label="Download Speech",
                    data=sound,
                    file_name="speech.mp3",
                    mime="audio/mp3"
                )


if __name__ == "__main__":
    main()
