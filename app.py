import os, io 
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from utils import (
    download_from_drive, download_folder_from_drive,
    load_class_index, load_model, predict_species,
    mw_search_titles, fetch_wikipedia_text,
    build_vector_store_from_text, make_rag_chain
)

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Bird ID + Wikipedia RAG", page_icon="🦜", layout="wide")
load_dotenv()

# Permanent configuration (no sidebar needed) ### CHANGED
MODEL_URL = "https://drive.google.com/uc?id=1DKRfw7Cdpi_LEVIPJfkdffgUjC4Boyf8"
CLASS_INDEX_URL = "https://drive.google.com/uc?id=1DEpmAZQDiWh_SRXB6wXTy1mfdmPns1ni"
IMAGES_DRIVE_URL = "https://drive.google.com/drive/folders/16f3KnxqFv2GTW70os7xBxIWT4chiR5sg?usp=sharing"

LOCAL_MODEL_PATH = "cache/model.pth"
LOCAL_CLASS_INDEX_PATH = "class_index.json"
SAMPLES_DIR = "sample_bird_images"
os.makedirs("cache", exist_ok=True)

# Get API key only from environment ### CHANGED
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment. Please set it before running.")
    st.stop()

# ----------------------------
# Download model + class index
# ----------------------------
if not os.path.exists(LOCAL_MODEL_PATH):
    st.write("📥 Downloading model weights...")
    if not download_from_drive(MODEL_URL, LOCAL_MODEL_PATH):
        st.error("❌ Could not download model file.")

if not os.path.exists(LOCAL_CLASS_INDEX_PATH):
    st.write("📥 Downloading class index...")
    if not download_from_drive(CLASS_INDEX_URL, LOCAL_CLASS_INDEX_PATH):
        st.error("❌ Could not download class index.")

# ----------------------------
# Download sample images (first run only)
# ----------------------------
if not os.path.exists(SAMPLES_DIR) or not os.listdir(SAMPLES_DIR):
    st.write("📥 Downloading sample images...")
    success = download_folder_from_drive(IMAGES_DRIVE_URL, SAMPLES_DIR)
    if not success:
        st.error("❌ Could not download sample images.")

# ----------------------------
# UI
# ----------------------------
st.title("🦜 AI Assistant for Bird Species: Image Classification + Wikipedia RAG Q&A")
st.warning("⚠️ Works on 200 bird species (CUB-200-2011). Other images may be unreliable.")

if os.path.exists(LOCAL_CLASS_INDEX_PATH):
    try:
        raw_class_index = load_class_index(LOCAL_CLASS_INDEX_PATH)
        cleaned_classes = []
        for v in raw_class_index.values():
            if "." in v: v = v.split(".", 1)[1]
            v = v.replace("_", " ")
            cleaned_classes.append(v)
        with st.expander("📖 View available bird classes"):
            st.write(cleaned_classes)
    except Exception as e:
        st.error(f"Could not load class index for display: {e}")

# Upload or select sample
col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("1) Upload or choose a sample image")
    file = st.file_uploader("Upload a bird image", type=["jpg","jpeg","png","webp"]) 
    image, sample_file = None, None

    if os.path.exists(SAMPLES_DIR) and os.listdir(SAMPLES_DIR):
        sample_images = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
        choice = st.selectbox("Or choose a sample image:", ["-- None --"] + sample_images)
        if choice != "-- None --":
            sample_file = os.path.join(SAMPLES_DIR, choice)
            image = Image.open(sample_file).convert("RGB")
            st.image(sample_file, caption=f"Sample: {choice}", use_column_width=True)

    if file is not None:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

# Prediction
with col_right:
    st.subheader("2) Model Prediction")
    class_index, model = None, None
    if os.path.exists(LOCAL_CLASS_INDEX_PATH):
        # Load class index mapping
        class_index = load_class_index(LOCAL_CLASS_INDEX_PATH)
        
        # Load model
        if os.path.exists(LOCAL_MODEL_PATH):
            model = load_model(LOCAL_MODEL_PATH, class_index)
            st.success("✅ Model loaded.")

    predicted_label, confidence = None, None
    if image is not None and model and class_index:
        predicted_label, confidence, _ = predict_species(model, class_index, image)

        # Convert ID → name using class_index
        if predicted_label in class_index:
            predicted_label = class_index[predicted_label]

        # Clean formatting
        if "." in predicted_label: 
            predicted_label = predicted_label.split(".", 1)[1]
        predicted_label = predicted_label.replace("_", " ")

        st.metric("Predicted species", predicted_label)
        st.write(f"Confidence: **{confidence:.2%}**")

# ----------------------------
# RAG Q&A
# ----------------------------
st.subheader("3) Ask questions about the species (Wikipedia-based)")
species_query = st.text_input("Wikipedia search query", value=(predicted_label or ""))

if st.button("🔎 Search Wikipedia"):
    candidates = mw_search_titles(species_query.strip(), limit=5)
    if not candidates:
        st.error("No Wikipedia results found.")
    else:
        st.session_state["wiki_candidates"] = candidates

if "wiki_candidates" in st.session_state:
    choice = st.selectbox("Choose the correct Wikipedia page:", st.session_state["wiki_candidates"])
    if choice:
        text, resolved = fetch_wikipedia_text(choice)
        if text:
            st.session_state["wiki_text"] = text
            st.session_state["resolved_title"] = resolved
            st.session_state["vector_store"] = build_vector_store_from_text(text)
            st.success(f"Knowledge base ready for: {resolved}")

question = st.text_input("Your question about the species")
if st.button("🧠 Ask with RAG"):
    if "vector_store" not in st.session_state:
        st.warning("Search & build the knowledge base first.")
    elif not question.strip():
        st.info("Type a question first.")
    else:
        retriever = st.session_state["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k": 4})
        chain = make_rag_chain(retriever, API_KEY)   ### CHANGED
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke(question)
                st.markdown("**Answer:**\n\n" + answer)
                st.caption(f"Grounded on Wikipedia page: {st.session_state['resolved_title']}")
            except Exception as e:
                st.error(f"LLM error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: grey;">
        🦜 Built with <b>Streamlit</b>, <b>PyTorch</b>, and <b>LangChain</b><br>
        Dataset: <a href="https://www.kaggle.com/datasets/wenewone/cub2002011" target="_blank">CUB-200-2011</a> |
        Wikipedia API | Groq LLM
        <br><br>
        <b>Created by Om Prakash Gadhwal</b><br>
        📧 <a href="mailto:omprakashg2004@gmail.com">omprakashg2004@gmail.com</a> |
        📱 <a href="tel:+916375834047">+91 63758 34047</a><br>
        🌐 <a href="https://github.com/omg04j" target="_blank">GitHub</a> •
        <a href="https://www.linkedin.com/in/om-prakash-gadhwal-19422628b/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
