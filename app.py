import streamlit as st
import pandas as pd
import os
import pyttsx3
import requests
from PIL import Image
import speech_recognition as sr
import json
from transformers import pipeline

# === Configuration ===
DATA_FOLDER = "Data"
MASTERDATA_FOLDER = "MasterData"
HF_API_URL = "https://api-inference.huggingface.co/models/google/derm-foundation"
CHATBOT_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = "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" # Use st.secrets or environment variable for production

# --- Set Streamlit Theme and Custom CSS ---
st.set_page_config(
    page_title="Dhanwantari",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use Markdown for a dynamic and modern look
st.markdown("""
<style>
    /* Use Poppins from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Overall Page Styling */
    .stApp {
        background-color: #FFFFFF; /* Light sky blue background */
        color: #000000;
    }

    /* Streamlit's Main Containers */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Customizing the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 2px solid #e0e6ed;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Header (Main Title) */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Subheadings */
    h2, h3 {
        color: #007bff;
        font-weight: 600;
    }
    
    /* General Text */
    p {
        color: #555;
        font-size: 1.1rem;
    }
    
    /* Buttons & Multiselect */
    .stButton>button, .stMultiSelect {
        background-color: #007bff;
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    }
    
    .stButton>button:hover, .stMultiSelect:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 123, 255, 0.4);
    }

    /* Cards/Sections */
    .st-emotion-cache-1c5vsmv, .st-emotion-cache-1629p8f {
        background-color: #fff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }

    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 12px;
        font-size: 1rem;
        border-left: 5px solid;
    }
    .stAlert.success {
        border-color: #28a745; /* Green */
    }
    .stAlert.info {
        border-color: #17a2b8; /* Cyan */
    }
    .stAlert.warning {
        border-color: #ffc107; /* Yellow */
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    [data-testid="stChatMessage"].user {

        color: #333;
    }

    [data-testid="stChatMessage"].assistant {
        background-color: #f5f5f5; /* Light grey */
        color: #333;
    }
    
    .st-chat-input .st-emotion-cache-1p6f58p {
        background-color: #fff;
        border-radius: 30px;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Local Translation Models Setup ---
@st.cache_resource
def get_translator(target_lang):
    if target_lang == "en":
        return None
    if target_lang == "hi":
        model_name = "Helsinki-NLP/opus-mt-en-hi"
        task = "translation_en_to_hi"
    elif target_lang == "mr":
        model_name = "Helsinki-NLP/opus-mt-en-mr"
        task = "translation_en_to_mr"
    elif target_lang == "ori":
        model_name = "ai4bharat/indictrans-fairseq"
        task = "translation_en_to_ori"
    else:
        st.error(f"No translation model available for language code '{target_lang}'.")
        return None
    try:
        translator = pipeline(task, model=model_name)
        return translator
    except Exception as e:
        st.error(f"Failed to load translation model for {target_lang}: {e}")
        return None

translators = {lang: get_translator(lang) for lang in ["hi", "mr", "ori"]}

# === Load CSV Datasets ===
try:
    training_data = pd.read_csv(os.path.join(DATA_FOLDER, "training.csv"))
    symptom_desc = pd.read_csv(os.path.join(MASTERDATA_FOLDER, "symptom_Description.csv"))
    symptom_precaution = pd.read_csv(os.path.join(MASTERDATA_FOLDER, "symptom_precaution.csv"))
    all_symptoms = training_data.columns[1:].tolist()

except FileNotFoundError as e:
    st.error(f"Error: A required data file was not found. Please ensure your project structure is correct.")
    st.stop()

# === Dynamic Translation Function using Hugging Face ===
@st.cache_data(show_spinner=False)
def translate_text(text, target_language):
    if target_language == 'en' or not text:
        return text
    
    translator = translators.get(target_language)
    if translator:
        try:
            translated_result = translator(text)
            return translated_result[0]['translation_text']
        except Exception as e:
            st.error(f"Translation failed: {e}")
            return text
    else:
        return text

# --- Helper Function for Language Code Mapping ---
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Odia": "ori"
}

# === Chatbot Functionality ===
def get_chatbot_response(question, disease_name, lang_code):
    greetings = ["hi", "hello", "hey", "namaste", "namaskar"]
    if any(greeting in question.lower() for greeting in greetings):
        return translate_text("Hello, how can I help you today?", lang_code)
    
    if disease_name:
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"The user has been diagnosed with {disease_name}. They have a question: {question}\n\nPlease provide a helpful and brief response."
        payload = {
            "inputs": prompt,
            "options": {"wait_for_model": True}
        }
        try:
            response = requests.post(CHATBOT_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() 
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                translated_response = translate_text(result[0]["generated_text"].strip(), lang_code)
                return translated_response
            else:
                return translate_text("I'm sorry, I cannot provide an answer for that.", lang_code)
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the chatbot service: {e}")
            return translate_text("I'm sorry, I cannot provide an answer for that.", lang_code)
    
    return translate_text("I'm sorry, I cannot provide an answer for that.", lang_code)


# === Streamlit UI ===
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "predicted_disease" not in st.session_state:
    st.session_state["predicted_disease"] = None

st.title(translate_text("🩺 Dhanwantari: Healthcare Chatbot", "en"))
st.write(translate_text("Your multilingual AI assistant for health assessments.", "en"))

# --- Main Page Content ---

# Create columns for the main content
col1, col2 = st.columns([1, 1.5])

# Column 1: Symptom Checker
with col1:
    st.subheader("Symptom-Based Analysis")
    name = st.text_input(translate_text("Enter your name", "en"))
    
    symptom_options_translated = [translate_text(sym.replace('_', ' '), "en") for sym in all_symptoms]
    user_symptoms_translated = st.multiselect(translate_text("Select your symptoms", "en"), symptom_options_translated)
    
    days = st.number_input(translate_text("How many days have you had symptoms?", "en"), min_value=1, max_value=30, value=3)

    if st.button(translate_text("🔎 Submit Symptoms", "en")):
        if user_symptoms_translated:
            translated_to_original = {translate_text(sym.replace('_', ' '), "en"): sym for sym in all_symptoms}
            all_user_symptoms_original = [translated_to_original[s] for s in user_symptoms_translated]

            matching_diseases = training_data.copy()
            matching_diseases["match_count"] = matching_diseases.iloc[:, 1:].apply(
                lambda row: sum(row[s] for s in all_user_symptoms_original if s in row.index), axis=1)
            predicted_disease = matching_diseases.sort_values(by="match_count", ascending=False).iloc[0]["prognosis"]
            st.session_state["predicted_disease"] = predicted_disease

            st.success(translate_text(f"Based on your symptoms, you might have: **{predicted_disease}**", "en"))
            
            description = symptom_desc[symptom_desc["prognosis"] == predicted_disease]["Description"].values
            if len(description) > 0:
                st.info(translate_text(description[0], "en"))

            precautions_data = symptom_precaution[symptom_precaution["Disease"] == predicted_disease].iloc[:, 1:].values.flatten()
            if len(precautions_data) > 0:
                st.write(f"### {translate_text('🛡️ Precautions', "en")}")
                for i, precaution in enumerate(precautions_data):
                    st.write(f"{i + 1}. {translate_text(precaution, "en")}")
        else:
            st.warning(translate_text("Please select at least one symptom.", "en"))

# Column 2: Chatbot & Image Uploader
with col2:
    st.subheader("AI Chatbot & Image Analysis")
    
    # Chatbot Section
    st.markdown("---")
    st.markdown(f"**💬 Ask a Question**")
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    
    user_question = st.chat_input("Type your question here...")
    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.spinner("Thinking..."):
            response = get_chatbot_response(user_question, st.session_state["predicted_disease"], "en")
            st.session_state.chat_history.append(("assistant", response))
            with st.chat_message("assistant"):
                st.markdown(response)

    st.markdown("---")
    st.markdown(f"**📸 Upload Skin Image for Disease Detection**")
    uploaded_file = st.file_uploader("Upload an image of the skin disease", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Skin Image", use_column_width=True)
        with st.spinner("🔄 Analyzing image using Hugging Face model..."):
            image_bytes = uploaded_file.read()
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/octet-stream"
            }
            try:
                response = requests.post(HF_API_URL, headers=headers, data=image_bytes)
                response.raise_for_status()
                predictions = response.json()
                if isinstance(predictions, list) and len(predictions) > 0:
                    top_prediction = predictions[0]
                    predicted_label = top_prediction["label"]
                    confidence = round(top_prediction["score"] * 100, 2)
                    st.success(f"🤖 AI Prediction: **{predicted_label}** ({confidence}% confidence)")
                    
                    precautions = symptom_precaution[symptom_precaution["Disease"].str.lower() == predicted_label.lower()].iloc[:, 1:].values.flatten()
                    if len(precautions) > 0:
                        st.markdown(f"### 🛡️ Suggested Precautions for this Condition")
                        for i, precaution in enumerate(precautions):
                            st.write(f"{i + 1}. {precaution}")
                    else:
                        st.info("No specific precautions found for this disease.")
                else:
                    st.error("⚠️ Could not detect any condition from the image.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to connect to Hugging Face API. Status Code: {e.response.status_code if e.response else 'N/A'}")
                st.write("API Response:")
                st.write(e.response.text if e.response else "No response body.")
