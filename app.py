import streamlit as st
import pandas as pd
import os
import pyttsx3
import requests
from PIL import Image
import speech_recognition as sr

# === Configuration ===
DATA_FOLDER = "Data"
MASTERDATA_FOLDER = "MasterData"
HF_API_URL = "https://api-inference.huggingface.co/models/Jayanth2002/dinov2-base-finetuned-SkinDisease"
HF_API_KEY = "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"

# === Load CSV Datasets ===
try:
    dataset = pd.read_csv(os.path.join(DATA_FOLDER, "dataset.csv"))
    training_data = pd.read_csv(os.path.join(DATA_FOLDER, "training.csv"))
    testing_data = pd.read_csv(os.path.join(DATA_FOLDER, "testing.csv"))

    symptom_desc = pd.read_csv(os.path.join(MASTERDATA_FOLDER, "symptom_Description.csv"))
    symptom_precaution = pd.read_csv(os.path.join(MASTERDATA_FOLDER, "symptom_precaution.csv"))
    symptom_severity = pd.read_csv(os.path.join(MASTERDATA_FOLDER, "Symptom_severity.csv"))

    all_symptoms = training_data.columns[1:].tolist()

except FileNotFoundError as e:
    st.error(f"Error: A required data file was not found. Please ensure your project structure is correct.")
    st.stop()

# === Translation Dictionaries ===
# UI Text Translations
translations = {
    "en": {
        "title": "🩺 Healthcare Chatbot", "sidebar_header": "Patient Info", "name_label": "Enter your name",
        "symptoms_label": "Select your symptoms", "days_label": "How many days have you had symptoms?",
        "additional_symptoms_header": "🔍 Additional Symptoms", "additional_symptom_question": "Do you also have {}?",
        "submit_button": "🔎 Submit Symptoms", "voice_button": "🎙️ Start Voice Recognition",
        "symptoms_title": "✔️ Your Selected Symptoms:", "diagnosis_title": "🩺 Possible Diagnosis from Symptoms",
        "description_title": "📚 Disease Description", "precautions_title": "🛡️ Precautions",
        "severity_title": "⚠️ Symptom Severity Levels", "image_upload_header": "📸 Upload Skin Image for Disease Detection",
        "upload_label": "Upload an image of the skin disease",
        "spinner_text": "🔄 Analyzing image using Hugging Face model...",
        "ai_prediction": "🤖 AI Prediction:", "ai_precaution_header": "🛡️ Suggested Precautions for this Condition",
        "no_precaution": "No specific precautions found for this disease.",
        "api_error": "❌ Failed to connect to Hugging Face API. Please check your key or quota.",
        "ai_no_detect": "⚠️ Could not detect any condition from the image.",
        "warning_select_symptom": "Please select at least one symptom.", "listening": "Listening...",
        "you_said": "You said:", "no_understand": "Could not understand audio.",
        "request_error": "Could not request results;", "processing_voice": "Processing your voice input:",
        "detected_symptoms": "✔️ Detected Symptoms from Voice Input:",
        "no_symptoms_found": "No symptoms found in your voice input.", "greeting": "Hello, how can I help you today?",
        "google_lang": "en-US"
    },
    "hi": {
        "title": "🩺 स्वास्थ्य देखभाल चैटबॉट", "sidebar_header": "रोगी जानकारी", "name_label": "अपना नाम दर्ज करें",
        "symptoms_label": "अपने लक्षण चुनें", "days_label": "आपको कितने दिनों से लक्षण हैं?",
        "additional_symptoms_header": "🔍 अतिरिक्त लक्षण", "additional_symptom_question": "क्या आपको {} भी है?",
        "submit_button": "🔎 लक्षण सबमिट करें", "voice_button": "🎙️ ध्वनि पहचान शुरू करें",
        "symptoms_title": "✔️ आपके चयनित लक्षण:", "diagnosis_title": "🩺 लक्षणों से संभावित निदान",
        "description_title": "📚 रोग का विवरण", "precautions_title": "🛡️ सावधानियां",
        "severity_title": "⚠️ लक्षणों का गंभीरता स्तर", "image_upload_header": "📸 त्वचा रोग का पता लगाने के लिए छवि अपलोड करें",
        "upload_label": "त्वचा रोग की एक छवि अपलोड करें",
        "spinner_text": "🔄 हगिंग फेस मॉडल का उपयोग करके छवि का विश्लेषण किया जा रहा है...",
        "ai_prediction": "🤖 AI भविष्यवाणी:", "ai_precaution_header": "🛡️ इस स्थिति के लिए सुझाई गई सावधानियां",
        "no_precaution": "इस रोग के लिए कोई विशेष सावधानी नहीं मिली।",
        "api_error": "❌ हगिंग फेस API से कनेक्ट होने में विफल। कृपया अपनी कुंजी या कोटा जांचें।",
        "ai_no_detect": "⚠️ छवि से कोई भी स्थिति का पता नहीं चला।",
        "warning_select_symptom": "कृपया कम से कम एक लक्षण चुनें।", "listening": "सुन रहा हूँ...",
        "you_said": "आपने कहा:", "no_understand": "ऑडियो समझ नहीं आया।",
        "request_error": "अनुरोध के परिणाम नहीं मिल सके;", "processing_voice": "आपके ध्वनि इनपुट का प्रसंस्करण किया जा रहा है:",
        "detected_symptoms": "✔️ ध्वनि इनपुट से पता चले लक्षण:",
        "no_symptoms_found": "आपके ध्वनि इनपुट में कोई लक्षण नहीं मिला।", "greeting": "नमस्ते, मैं आज आपकी कैसे मदद कर सकता हूँ?",
        "google_lang": "hi-IN"
    },
    "mr": {
        "title": "🩺 आरोग्य सेवा चॅटबॉट", "sidebar_header": "रुग्णाची माहिती", "name_label": "तुमचे नाव टाका",
        "symptoms_label": "तुमची लक्षणे निवडा", "days_label": "तुम्हाला किती दिवसांपासून लक्षणे आहेत?",
        "additional_symptoms_header": "🔍 अतिरिक्त लक्षणे", "additional_symptom_question": "तुम्हाला {} सुद्धा आहे का?",
        "submit_button": "🔎 लक्षणे सबमिट करा", "voice_button": "🎙️ व्हॉइस रेकग्निशन सुरू करा",
        "symptoms_title": "✔️ तुमची निवडलेली लक्षणे:", "diagnosis_title": "🩺 लक्षणांवर आधारित संभाव्य निदान",
        "description_title": "📚 रोगाचे वर्णन", "precautions_title": "🛡️ खबरदारी",
        "severity_title": "⚠️ लक्षणांची तीव्रता पातळी", "image_upload_header": "📸 त्वचेच्या रोगाच्या शोधासाठी प्रतिमा अपलोड करा",
        "upload_label": "त्वचेच्या रोगाची प्रतिमा अपलोड करा",
        "spinner_text": "🔄 हगिंग फेस मॉडेल वापरून प्रतिमेचे विश्लेषण करत आहे...",
        "ai_prediction": "🤖 AI अंदाज:", "ai_precaution_header": "🛡️ या स्थितीसाठी सुचवलेली खबरदारी",
        "no_precaution": "या रोगासाठी कोणतीही विशिष्ट खबरदारी सापडली नाही।",
        "api_error": "❌ हगिंग फेस API शी कनेक्ट होण्यात अयशस्वी. कृपया तुमची की किंवा कोटा तपासा.",
        "ai_no_detect": "⚠️ प्रतिमेतून कोणतीही स्थिती ओळखता आली नाही।",
        "warning_select_symptom": "कृपया कमीतकमी एक लक्षण निवडा।", "listening": "ऐकत आहे...",
        "you_said": "तुम्ही म्हणालात:", "no_understand": "ऑडिओ समजू शकला नाही।",
        "request_error": "विनंती परिणाम मिळू शकला नाही;", "processing_voice": "तुमची व्हॉइस इनपुट प्रक्रिया करत आहे:",
        "detected_symptoms": "✔️ व्हॉइस इनपुटने आढळलेली लक्षणे:",
        "no_symptoms_found": "तुमच्या व्हॉइस इनपुटमध्ये कोणतीही लक्षणे आढळली नाहीत।", "greeting": "नमस्कार, मी आज तुम्हाला कशी मदत करू शकेन?",
        "google_lang": "mr-IN"
    },
    "or": {
        "title": "🩺 ସ୍ୱାସ୍ଥ୍ୟସେବା ଚାଟବଟ୍", "sidebar_header": "ରୋଗୀ ସୂଚନା", "name_label": "ଆପଣଙ୍କ ନାମ ପ୍ରବେଶ କରନ୍ତୁ",
        "symptoms_label": "ଆପଣଙ୍କ ଲକ୍ଷଣଗୁଡ଼ିକୁ ବାଛନ୍ତୁ", "days_label": "ଆପଣଙ୍କୁ କେତେ ଦିନ ହେବ ଲକ୍ଷଣ ଅଛି?",
        "additional_symptoms_header": "🔍 ଅତିରିକ୍ତ ଲକ୍ଷଣଗୁଡ଼ିକ", "additional_symptom_question": "ଆପଣଙ୍କୁ {} ମଧ୍ୟ ଅଛି କି?",
        "submit_button": "🔎 ଲକ୍ଷଣ ଦାଖଲ କରନ୍ତୁ", "voice_button": "🎙️ ଭଏସ୍ ଚିହ୍ନଟ ଆରମ୍ଭ କରନ୍ତୁ",
        "symptoms_title": "✔️ ଆପଣଙ୍କ ଦ୍ୱାରା ଚୟନିତ ଲକ୍ଷଣଗୁଡ଼ିକ:", "diagnosis_title": "🩺 ଲକ୍ଷଣରୁ ସମ୍ଭାବ୍ୟ ନିର୍ଣ୍ଣୟ",
        "description_title": "📚 ରୋଗର ବର୍ଣ୍ଣନା", "precautions_title": "🛡️ ସାବଧାନତା",
        "severity_title": "⚠️ ଲକ୍ଷଣର ଗୁରୁତ୍ୱ ସ୍ତର", "image_upload_header": "📸 ଚର୍ମ ରୋଗ ଚିହ୍ନଟ ପାଇଁ ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ",
        "upload_label": "ଚର୍ମ ରୋଗର ଏକ ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ",
        "spinner_text": "🔄 ହଗିଂ ଫେସ୍ ମଡେଲ୍ ବ୍ୟବହାର କରି ଛବି ବିଶ୍ଳେଷଣ କରାଯାଉଛି...",
        "ai_prediction": "🤖 AI ପୂର୍ବାନୁମାନ:", "ai_precaution_header": "🛡️ ଏହି ସ୍ଥିତି ପାଇଁ ସୁଗଠିତ ସାବଧାନତା",
        "no_precaution": "ଏହି ରୋଗ ପାଇଁ କୌଣସି ନିର୍ଦ୍ଦିଷ୍ଟ ସାବଧାନତା ମିଳିଲା ନାହିଁ।",
        "api_error": "❌ ହଗିଂ ଫେସ୍ API ସହିତ ସଂଯୋଗ ବିଫଳ ହେଲା। ଦୟାକରି ଆପଣଙ୍କ କି କିମ୍ବା କୋଟା ଯାଞ୍ଚ କରନ୍ତୁ।",
        "ai_no_detect": "⚠️ ଛବିରୁ କୌଣସି ଅବସ୍ଥା ଚିହ୍ନଟ ହୋଇପାରିଲା ନାହିଁ।",
        "warning_select_symptom": "ଦୟାକରି ଅତି କମରେ ଗୋଟିଏ ଲକ୍ଷଣ ବାଛନ୍ତୁ।", "listening": "ଶୁଣୁଛି...",
        "you_said": "ଆପଣ କହିଲେ:", "no_understand": "ଅଡିଓ ବୁଝିହେଲା ନାହିଁ।",
        "request_error": "ଅନୁରୋଧ ଫଳାଫଳ ମିଳିଲା ନାହିଁ;", "processing_voice": "ଆପଣଙ୍କ ଭଏସ୍ ଇନପୁଟ୍ ପ୍ରକ୍ରିୟାକରଣ କରାଯାଉଛି:",
        "detected_symptoms": "✔️ ଭଏସ୍ ଇନପୁଟ୍ରୁ ଚିହ୍ନଟ ଲକ୍ଷଣଗୁଡ଼ିକ:",
        "no_symptoms_found": "ଆପଣଙ୍କ ଭଏସ୍ ଇନପୁଟ୍ରେ କୌଣସି ଲକ୍ଷଣ ମିଳିଲା ନାହିଁ।", "greeting": "ନମସ୍କାର, ମୁଁ ଆଜି ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
        "google_lang": "or-IN"
    }
}

# Symptom Name Translations
symptom_translations = {
    "itching": {"en": "Itching", "hi": "खुजली", "mr": "खाज", "or": "କାଛୁ"},
    "skin_rash": {"en": "Skin Rash", "hi": "त्वचा पर चकत्ते", "mr": "त्वचेवर पुरळ", "or": "ଚର୍ମ ଘା'"},
    "nodal_skin_eruptions": {"en": "Nodal Skin Eruptions", "hi": "नोडल त्वचा विस्फोट", "mr": "नोडल त्वचेचा उद्रेक", "or": "ନୋଡାଲ୍ ଚର୍ମ ଫାଟ"},
    "continuous_sneezing": {"en": "Continuous Sneezing", "hi": "लगातार छींक आना", "mr": "सतत शिंका येणे", "or": "ଲଗାତାର ଛିଙ୍କ"},
    "shivering": {"en": "Shivering", "hi": "कंपकंपी", "mr": "थंडी वाजणे", "or": "ଥରିବା"},
    "chills": {"en": "Chills", "hi": "ठंड लगना", "mr": "थंडी", "or": "ଥଣ୍ଡା ଲାଗିବା"},
    "joint_pain": {"en": "Joint Pain", "hi": "जोड़ों का दर्द", "mr": "सांधेदुखी", "or": "ଗଣ୍ଠି ଯନ୍ତ୍ରଣା"},
    "stomach_pain": {"en": "Stomach Pain", "hi": "पेट दर्द", "mr": "पोटदुखी", "or": "ପେଟ ଯନ୍ତ୍ରଣା"},
    "acidity": {"en": "Acidity", "hi": "एसिडिटी", "mr": "आंबटपणा", "or": "ଏସିଡିଟି"},
    "ulcers_on_tongue": {"en": "Ulcers on Tongue", "hi": "जीभ पर छाले", "mr": "जिभेवर फोड", "or": "ଜିଭରେ ଘା"},
    "muscle_wasting": {"en": "Muscle Wasting", "hi": "मांसपेशियों का क्षय", "mr": "स्नायूंचा ऱ्हास", "or": "ମାଂସପେଶୀ ନଷ୍ଟ"},
    "vomiting": {"en": "Vomiting", "hi": "उल्टी", "mr": "उलट्या होणे", "or": "ବାନ୍ତି"},
    "burning_micturition": {"en": "Burning Micturition", "hi": "पेशाब में जलन", "mr": "लघवी करताना जळजळ", "or": "ପରିସ୍ରା କରିବାରେ ଜଳାପୋଡା"},
    "fatigue": {"en": "Fatigue", "hi": "थकान", "mr": "थकवा", "or": "କ୍ଳାନ୍ତି"},
    "weight_gain": {"en": "Weight Gain", "hi": "वजन बढ़ना", "mr": "वजन वाढणे", "or": "ଓଜନ ବୃଦ୍ଧି"},
    "anxiety": {"en": "Anxiety", "hi": "चिंता", "mr": "चिंता", "or": "ଚିନ୍ତା"},
    "cold_hands_and_feets": {"en": "Cold Hands and Feet", "hi": "हाथ और पैर ठंडे", "mr": "हात आणि पाय थंड", "or": "ହାତ ଏବଂ ପାଦ ଥଣ୍ଡା"},
    "mood_swings": {"en": "Mood Swings", "hi": "मूड में बदलाव", "mr": "मूड बदलणे", "or": "ମନର ପରିବର୍ତ୍ତନ"},
    "weight_loss": {"en": "Weight Loss", "hi": "वजन घटना", "mr": "वजन कमी होणे", "or": "ଓଜନ ହ୍ରାସ"},
    "restlessness": {"en": "Restlessness", "hi": "बेचैनी", "mr": "अस्वस्थता", "or": "ଅଶାନ୍ତି"},
    "lethargy": {"en": "Lethargy", "hi": "आलस्य", "mr": "आलस", "or": "ଆଳସ୍ୟ"},
    "patches_in_throat": {"en": "Patches in Throat", "hi": "गले में धब्बे", "mr": "घशात डाग", "or": "ଗଳାରେ ଦାଗ"},
    "high_fever": {"en": "High Fever", "hi": "तेज बुखार", "mr": "तीव्र ताप", "or": "ଅଧିକ ଜ୍ୱର"},
    "sunken_eyes": {"en": "Sunken Eyes", "hi": "धंसी हुई आंखें", "mr": "डोळे खोल जाणे", "or": "ପଶି ଯାଇଥିବା ଆଖି"},
    "dehydration": {"en": "Dehydration", "hi": "निर्जलीकरण", "mr": "निर्जलीकरण", "or": "ଜଳହୀନତା"},
    "indigestion": {"en": "Indigestion", "hi": "अपच", "mr": "अपचन", "or": "ଅଜୀର୍ଣ୍ଣ"},
    "headache": {"en": "Headache", "hi": "सरदर्द", "mr": "डोकेदुखी", "or": "ମୁଣ୍ଡବିନ୍ଧା"},
    "yellowish_skin": {"en": "Yellowish Skin", "hi": "पीली त्वचा", "mr": "पिवळी त्वचा", "or": "ହଳଦିଆ ଚର୍ମ"},
    "dark_urine": {"en": "Dark Urine", "hi": "गहरा पेशाब", "mr": "गडद लघवी", "or": "ଗାଢ ପରିସ୍ରା"},
    "nausea": {"en": "Nausea", "hi": "जी मिचलाना", "mr": "मळमळ", "or": "ଅଇ"},
    "loss_of_appetite": {"en": "Loss of Appetite", "hi": "भूख न लगना", "mr": "भूक न लागणे", "or": "ଭୋକ ନ ଲାଗିବା"},
    "pain_behind_the_eyes": {"en": "Pain Behind the Eyes", "hi": "आंखों के पीछे दर्द", "mr": "डोळ्यांच्या मागे वेदना", "or": "ଆଖି ପଛରେ ଯନ୍ତ୍ରଣା"},
    "back_pain": {"en": "Back Pain", "hi": "पीठ दर्द", "mr": "पाठदुखी", "or": "ପିଠି ଯନ୍ତ୍ରଣା"},
    "constipation": {"en": "Constipation", "hi": "कब्ज", "mr": "बद्धकोष्ठता", "or": "କୋଷ୍ଠକାଠିନ୍ୟ"},
    "abdominal_pain": {"en": "Abdominal Pain", "hi": "पेट दर्द", "mr": "पोटदुखी", "or": "ପେଟ ଯନ୍ତ୍ରଣା"},
    "diarrhoea": {"en": "Diarrhoea", "hi": "दस्त", "mr": "जुलाब", "or": "ଝାଡା"},
    "mild_fever": {"en": "Mild Fever", "hi": "हल्का बुखार", "mr": "हलका ताप", "or": "ହାଲୁକା ଜ୍ୱର"},
    "yellow_urine": {"en": "Yellow Urine", "hi": "पीला पेशाब", "mr": "पिवळी लघवी", "or": "ହଳଦିଆ ପରିସ୍ରା"},
    "yellowing_of_eyes": {"en": "Yellowing of Eyes", "hi": "आंखों का पीला होना", "mr": "डोळ्यांचा पिवळेपणा", "or": "ଆଖି ହଳଦିଆ ହେବା"},
    "acute_liver_failure": {"en": "Acute Liver Failure", "hi": "तीव्र यकृत विफलता", "mr": "तीव्र यकृत निकामी", "or": "ତୀବ୍ର ଯକୃତ ବିଫଳତା"},
    "fluid_overload": {"en": "Fluid Overload", "hi": "फ्लूइड ओवरलोड", "mr": "द्रव वाढणे", "or": "ତରଳ ପଦାର୍ଥର ବୃଦ୍ଧି"},
    "swelling_of_stomach": {"en": "Swelling of Stomach", "hi": "पेट में सूजन", "mr": "पोटात सूज", "or": "ପେଟ ଫୁଲିବା"},
    "swelled_lymph_nodes": {"en": "Swollen Lymph Nodes", "hi": "सूजे हुए लिम्फ नोड्स", "mr": "सुजलेले लिम्फ नोड्स", "or": "ଫୁଲିଥିବା ଲିମ୍ଫ ନୋଡ୍"},
    "malaise": {"en": "Malaise", "hi": "अस्वस्थता", "mr": "अस्वस्थता", "or": "ଅସୁସ୍ଥତା"},
    "blurred_and_distorted_vision": {"en": "Blurred and Distorted Vision", "hi": "धुंधली और विकृत दृष्टि", "mr": "अंधुक आणि विकृत दृष्टी", "or": "ଅସ୍ପଷ୍ଟ ଏବଂ ବିକୃତ ଦୃଷ୍ଟି"},
    "phlegm": {"en": "Phlegm", "hi": "कफ", "mr": "कफ", "or": "କଫ"},
    "throat_irritation": {"en": "Throat Irritation", "hi": "गले में जलन", "mr": "घसा खवखवणे", "or": "ଗଳା ଜଳାପୋଡା"},
    "redness_of_eyes": {"en": "Redness of Eyes", "hi": "आंखों का लाल होना", "mr": "डोळ्यांची लालसरपणा", "or": "ଆଖି ଲାଲ ହେବା"},
    "sinus_pressure": {"en": "Sinus Pressure", "hi": "साइनस दबाव", "mr": "सायनस दाब", "or": "ସାଇନସ୍ ଚାପ"},
    "runny_nose": {"en": "Runny Nose", "hi": "बहती नाक", "mr": "नाक वाहणे", "or": "ନାକ ବହିବା"},
    "congestion": {"en": "Congestion", "hi": "भीड़", "mr": "गर्दी", "or": "ଭିଡ଼"},
    "chest_pain": {"en": "Chest Pain", "hi": "सीने में दर्द", "mr": "छातीत दुखणे", "or": "ଛାତି ଯନ୍ତ୍ରଣା"},
    "fast_heart_rate": {"en": "Fast Heart Rate", "hi": "तेज हृदय गति", "mr": "जलद हृदयाची गती", "or": "ଦ୍ରୁତ ହୃଦୟ ଗତି"},
    "dizziness": {"en": "Dizziness", "hi": "चक्कर आना", "mr": "चक्कर येणे", "or": "ମୁଣ୍ଡ ବୁଲାଇବା"},
    "loss_of_balance": {"en": "Loss of Balance", "hi": "संतुलन खोना", "mr": "संतुलन गमावणे", "or": "ସନ୍ତୁଳନ ହରାଇବା"},
    "lack_of_concentration": {"en": "Lack of Concentration", "hi": "एकाग्रता की कमी", "mr": "एकाग्रतेचा अभाव", "or": "ଏକାଗ୍ରତାର ଅଭାବ"},
    "altered_sensorium": {"en": "Altered Sensorium", "hi": "परिवर्तित संवेदनशीलता", "mr": "बदललेली संवेदनशीलता", "or": "ପରିବର୍ତ୍ତିତ ସେନ୍ସରିୟମ୍"},
    "family_history": {"en": "Family History", "hi": "पारिवारिक इतिहास", "mr": "कौटुंबिक इतिहास", "or": "ପାରିବାରିକ ଇତିହାସ"},
    "mucoid_sputum": {"en": "Mucoid Sputum", "hi": "म्यूकॉइड थूक", "mr": "म्यूकॉइड थुंका", "or": "ମ୍ୟୁକଏଡ୍ ଖଙ୍କାର"},
    "rusty_sputum": {"en": "Rusty Sputum", "hi": "जंग लगा थूक", "mr": "गंजलेला थुंका", "or": "ଜଙ୍ଗ ଖଙ୍କାର"},
    "lack_of_odor": {"en": "Lack of Odor", "hi": "गंध की कमी", "mr": "वासाचा अभाव", "or": "ଗନ୍ଧର ଅଭାବ"},
    "irritation_in_anus": {"en": "Irritation in Anus", "hi": "गुदा में जलन", "mr": "गुदद्वारात जळजळ", "or": "ଗୁଦ୍ୱାରରେ ଜଳାପୋଡା"},
    "passage_of_gases": {"en": "Passage of Gases", "hi": "गैसों का निकलना", "mr": "वायू बाहेर पडणे", "or": "ଗ୍ୟାସ୍ ବାହାରିବା"},
    "internal_itching": {"en": "Internal Itching", "hi": "आंतरिक खुजली", "mr": "अंतर्गत खाज", "or": "ଆଭ୍ୟନ୍ତରୀଣ କାଛୁ"},
    "toxic_look": {"en": "Toxic Look", "hi": "विषाक्त रूप", "mr": "विषारी दिसणे", "or": "ବିଷାକ୍ତ ଦେଖାଯିବା"},
    "unsteadiness": {"en": "Unsteadiness", "hi": "अस्थिरता", "mr": "अस्थिरता", "or": "ଅସ୍ଥିରତା"},
    "swelling_of_legs": {"en": "Swelling of Legs", "hi": "पैरों में सूजन", "mr": "पायांना सूज", "or": "ଗୋଡ ଫୁଲିବା"},
    "swollen_ankles": {"en": "Swollen Ankles", "hi": "सूजे हुए टखने", "mr": "सुजलेले घोटे", "or": "ଫୁଲିଥିବା ଗୋଇଠି"},
    "brittle_nails": {"en": "Brittle Nails", "hi": "कमजोर नाखून", "mr": "ठिसूळ नखे", "or": "ଭଙ୍ଗୁର ନଖ"},
    "puffy_face_and_eyes": {"en": "Puffy Face and Eyes", "hi": "फूला हुआ चेहरा और आंखें", "mr": "सुजलेले चेहरा आणि डोळे", "or": "ଫୁଲିଥିବା ମୁହଁ ଏବଂ ଆଖି"},
    "enlarged_thyroid": {"en": "Enlarged Thyroid", "hi": "बढ़ा हुआ थायराइड", "mr": "वाढलेली थायरॉइड ग्रंथी", "or": "ବର୍ଦ୍ଧିତ ଥାଇରଏଡ୍"},
    "slurred_speech": {"en": "Slurred Speech", "hi": "अस्पष्ट भाषण", "mr": "अस्पष्ट भाषण", "or": "ସ୍ଲୁର୍ଡ୍ ବକ୍ତୃତା"},
    "red_sore_around_nose": {"en": "Red Sore Around Nose", "hi": "नाक के चारों ओर लाल घाव", "mr": "नाकाजवळ लाल फोड", "or": "ନାକ ଚାରିପଟେ ଲାଲ ଘା'"},
    "ulcers_on_lips": {"en": "Ulcers on Lips", "hi": "होंठों पर छाले", "mr": "ओठांवर फोड", "or": "ଓଠରେ ଘା"},
    "blistering": {"en": "Blistering", "hi": "छाले पड़ना", "mr": "फोडे येणे", "or": "ଫୋଟକା ହେବା"},
    "spotting_urination": {"en": "Spotting Urination", "hi": "पेशाब में धब्बे", "mr": "लघवीत ठिपके", "or": "ପରିସ୍ରାରେ ଦାଗ"}
}

# === Text-to-Speech ===
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)
    engine.say(text)
    engine.runAndWait()

# === Voice Recognition ===
def voice_input(lang_dict):
    r = sr.Recognizer()
    try:
        st.info(lang_dict["listening"])
        speak(lang_dict["listening"])
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=5)
        text = r.recognize_google(audio, language=lang_dict.get("google_lang", "en-US"))
        st.success(f"{lang_dict['you_said']} {text}")
        return text
    except sr.UnknownValueError:
        st.error(lang_dict["no_understand"])
        return ""
    except sr.RequestError as e:
        st.error(f"{lang_dict['request_error']} {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return ""

# === Streamlit UI ===
st.set_page_config(page_title="Healthcare Chatbot", layout="wide")

# Add the manual language selection dropdown
selected_lang = st.sidebar.selectbox(
    "Select Language",
    list(translations.keys()),
    format_func=lambda x: {"en": "English", "hi": "हिंदी", "mr": "Marathi", "or": "Odia"}[x]
)
lang_dict = translations[selected_lang]

# Create the list of translated symptom options for the multiselect
symptom_options = [symptom_translations.get(sym, {}).get(selected_lang, sym.replace('_', ' ')) for sym in all_symptoms]

st.title(lang_dict["title"])
st.sidebar.header(lang_dict["sidebar_header"])

# Sidebar inputs
name = st.sidebar.text_input(lang_dict["name_label"])
user_symptoms_translated = st.sidebar.multiselect(lang_dict["symptoms_label"], symptom_options)
days = st.sidebar.number_input(lang_dict["days_label"], min_value=1, max_value=30, value=3)

# Voice input for symptoms
st.write("### 🎙️ Voice Recognition")
if st.button(lang_dict["voice_button"]):
    recognized_text = voice_input(lang_dict)
    if recognized_text:
        st.write(f"### {lang_dict['detected_symptoms']}")
        # Match voice input with original symptom names (more robust)
        matching_symptoms = [sym for sym in all_symptoms if sym.replace('_', ' ').lower() in recognized_text.lower()]
        
        if matching_symptoms:
            display_symptoms = [symptom_translations.get(sym, {}).get(selected_lang, sym.replace('_', ' ')) for sym in matching_symptoms]
            st.info(", ".join(display_symptoms))
        else:
            st.warning(lang_dict["no_symptoms_found"])

# Additional symptoms
additional_symptoms_translated = []
if user_symptoms_translated:
    st.write(f"### {lang_dict['additional_symptoms_header']}")
    # Get the original symptom names from the translated ones
    selected_english_symptoms = [sym for sym in all_symptoms if symptom_translations.get(sym, {}).get(selected_lang, sym.replace('_', ' ')) in user_symptoms_translated]

    for symptom in all_symptoms:
        if symptom not in selected_english_symptoms:
            translated_symptom_name = symptom_translations.get(symptom, {}).get(selected_lang, symptom.replace('_', ' '))
            translated_question = lang_dict['additional_symptom_question'].format(translated_symptom_name)
            
            # --- THE FIX IS HERE ---
            # We use the original symptom name as a unique key for each checkbox
            response = st.checkbox(translated_question, key=f"checkbox_{symptom}")
            
            if response:
                additional_symptoms_translated.append(translated_symptom_name)

# ===== Submit Symptoms Button =====
if st.button(lang_dict["submit_button"]):
    if user_symptoms_translated:
        all_user_symptoms_translated = list(set(user_symptoms_translated + additional_symptoms_translated))
        
        # Convert all selected symptoms back to original English names
        all_user_symptoms_original = [
            sym for sym in all_symptoms if symptom_translations.get(sym, {}).get(selected_lang, sym.replace('_', ' ')) in all_user_symptoms_translated
        ]
        
        st.write(f"### {lang_dict['symptoms_title']}")
        st.write(", ".join(all_user_symptoms_translated))

        # Predict Disease using original symptom names
        matching_diseases = training_data.copy()
        matching_diseases["match_count"] = matching_diseases.iloc[:, 1:].apply(
            lambda row: sum(row[all_user_symptoms_original]), axis=1)
        predicted_disease = matching_diseases.sort_values(by="match_count", ascending=False).iloc[0]["prognosis"]

        st.write(f"### {lang_dict['diagnosis_title']}")
        st.success(f"Based on your symptoms, you might have: **{predicted_disease}**")
        speak(f"Based on your symptoms, you might have {predicted_disease}")

        # Disease Description
        description = symptom_desc[symptom_desc["prognosis"] == predicted_disease]["Description"].values
        if len(description) > 0:
            st.write(f"### {lang_dict['description_title']}")
            st.info(description[0])

        # Precautions
        precautions = symptom_precaution[symptom_precaution["Disease"] == predicted_disease].iloc[:, 1:].values.flatten()
        if len(precautions) > 0:
            st.write(f"### {lang_dict['precautions_title']}")
            for i, precaution in enumerate(precautions):
                st.write(f"{i + 1}. {precaution}")

        # Symptom Severity
        st.write(f"### {lang_dict['severity_title']}")
        severity_data = []
        for symptom in all_user_symptoms_original:
            severity = symptom_severity[symptom_severity["Symptom"] == symptom]["weight"].values
            if len(severity) > 0:
                severity_data.append((symptom.replace('_', ' '), severity[0]))
        severity_df = pd.DataFrame(severity_data, columns=["Symptom", "Severity Level"])
        st.table(severity_df)
    else:
        st.warning(lang_dict["warning_select_symptom"])

# ====== Skin Disease Image Upload (Hugging Face Model) ======
st.write(f"## {lang_dict['image_upload_header']}")
uploaded_file = st.file_uploader(lang_dict["upload_label"], type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Skin Image", use_column_width=True)
    # Send image to Hugging Face API
    with st.spinner(lang_dict["spinner_text"]):
        image_bytes = uploaded_file.read()
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(HF_API_URL, headers=headers, data=image_bytes)
        if response.status_code == 200:
            predictions = response.json()
            if isinstance(predictions, list) and len(predictions) > 0:
                top_prediction = predictions[0]
                predicted_label = top_prediction["label"]
                confidence = round(top_prediction["score"] * 100, 2)
                st.success(f"{lang_dict['ai_prediction']} **{predicted_label}** ({confidence}% confidence)")
                speak(f"The AI model predicts {predicted_label} with {confidence} percent confidence")
                # Show precautions
                st.write(f"### {lang_dict['ai_precaution_header']}")
                precautions = symptom_precaution[symptom_precaution["Disease"].str.lower() == predicted_label.lower()].iloc[:, 1:].values.flatten()
                if len(precautions) > 0:
                    for i, precaution in enumerate(precautions):
                        st.write(f"{i + 1}. {precaution}")
                else:
                    st.info(lang_dict["no_precaution"])
            else:
                st.error(lang_dict["ai_no_detect"])
        else:
            st.error(lang_dict["api_error"])
