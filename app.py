"""
Streamlit App: Bird Classifier + Wikipedia RAG Q&A

✅ Updated: Model + Class Index fetched from Google Drive links using gdown
✅ Your best_model.pth Google Drive link integrated
"""

import os
import io
import json
from typing import Dict, Tuple, Optional

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from dotenv import load_dotenv
import wikipediaapi
import gdown

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Bird ID + Wikipedia RAG", page_icon="🦜", layout="wide")
load_dotenv()

# Sidebar: model links and API key
st.sidebar.header("⚙️ Configuration")

# ✅ Your fixed Google Drive model link
DEFAULT_MODEL_URL = "https://drive.google.com/uc?id=1DKRfw7Cdpi_LEVIPJfkdffgUjC4Boyf8"
# You will still need to upload your class_index.json to Drive and set link here
DEFAULT_CLASS_INDEX_URL = os.getenv("CLASS_INDEX_URL", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

model_url = st.sidebar.text_input("Google Drive URL for model (.pth)", value=DEFAULT_MODEL_URL)
class_index_url = st.sidebar.text_input("Google Drive URL for class index (.json)", value=DEFAULT_CLASS_INDEX_URL)
api_key = st.sidebar.text_input("GROQ_API_KEY (hidden)", value=GROQ_API_KEY, type="password")

LOCAL_MODEL_PATH = "cache/model.pth"
LOCAL_CLASS_INDEX_PATH = "cache/class_index.json"

os.makedirs("cache", exist_ok=True)


# ----------------------------
# Utilities
# ----------------------------
def download_from_drive(url: str, output_path: str) -> bool:
    """Download file from Google Drive link using gdown"""
    try:
        if os.path.exists(output_path):
            return True
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        return os.path.exists(output_path)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False


@st.cache_resource(show_spinner=False)
def load_class_index(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


def build_model(num_classes: int) -> nn.Module:
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2048),
        nn.BatchNorm1d(2048),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str, class_index: Dict[int, str]) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_index))
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model."):]] = v
            elif k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        state = new_state
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


# Normalization & Denormalization
@st.cache_resource(show_spinner=False)
def get_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean


def predict_species(model: nn.Module, class_index: Dict[int, str], image: Image.Image) -> Tuple[str, float, int]:
    device = next(model.parameters()).device
    tfm = get_transform()
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    idx = pred_idx.item()
    label = class_index.get(idx, f"class_{idx}")
    return label, conf.item(), idx


# ----------------------------
# Wikipedia + RAG
# ----------------------------
@st.cache_resource(show_spinner=False)
def fetch_wikipedia_text(title: str) -> str:
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="Bird-RAG-App/0.1")
    page = wiki.page(title)
    if page.exists():
        return page.text
    alt = title.replace("_", " ").title()
    page2 = wiki.page(alt)
    return page2.text if page2.exists() else ""


@st.cache_resource(show_spinner=True)
def build_vector_store_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def make_llm(api_key: str) -> ChatGroq:
    return ChatGroq(api_key=api_key, model="llama3-8b-8192", temperature=0.2, max_tokens=512)


PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Answer ONLY from the provided Wikipedia context.\n"
        "If the context is insufficient, say 'I don't know.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    ),
    input_variables=["context", "question"],
)


# ----------------------------
# UI
# ----------------------------
st.title("🦜 Bird Species Classifier + Wikipedia RAG Q&A")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1) Upload an image")
    file = st.file_uploader("Choose a bird image", type=["jpg", "jpeg", "png", "webp"]) 
    if file is not None:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

with col_right:
    st.subheader("2) Model prediction")
    class_index: Optional[Dict[int, str]] = None
    model: Optional[nn.Module] = None

    class_index_ok = False
    model_ok = False

    if class_index_url.strip():
        if download_from_drive(class_index_url.strip(), LOCAL_CLASS_INDEX_PATH):
            try:
                class_index = load_class_index(LOCAL_CLASS_INDEX_PATH)
                class_index_ok = True
                st.success("Class index loaded.")
            except Exception as e:
                st.error(f"Failed to load class index: {e}")
    else:
        st.info("Paste a Google Drive link for class index JSON.")

    if class_index_ok and model_url.strip():
        if download_from_drive(model_url.strip(), LOCAL_MODEL_PATH):
            try:
                model = load_model(LOCAL_MODEL_PATH, class_index)
                model_ok = True
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    predicted_label = None
    confidence = None
    if file is not None and model_ok and class_index:
        try:
            predicted_label, confidence, idx = predict_species(model, class_index, image)
            st.metric("Predicted species", predicted_label)
            st.write(f"Confidence: **{confidence:.2%}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")

# ----------------------------
# 3) Wikipedia RAG Section
# ----------------------------
st.subheader("3) Ask questions about the predicted species (Wikipedia-grounded)")

species_title = st.text_input(
    "Wikipedia page title (auto-filled from prediction; edit if needed)",
    value=(predicted_label or ""),
)

col_a, col_b = st.columns([1, 1])
with col_a:
    build_ctx = st.button("📚 Build Knowledge Base from Wikipedia")

if build_ctx and species_title.strip():
    with st.spinner("Fetching Wikipedia and building vector store..."):
        wiki_text = fetch_wikipedia_text(species_title.strip())
        if not wiki_text:
            st.error("Could not find a Wikipedia page for that title.")
        else:
            st.session_state["wiki_text"] = wiki_text
            st.session_state["vector_store"] = build_vector_store_from_text(wiki_text)
            st.success("Knowledge base ready.")

question = st.text_input("Your question about the species", placeholder="e.g., What are its key characteristics?")
ask = st.button("🧠 Ask with RAG")

if ask:
    if not api_key:
        st.error("Please provide GROQ_API_KEY in the sidebar or environment.")
    elif "vector_store" not in st.session_state:
        st.warning("Build the knowledge base first (click the button above).")
    elif not question.strip():
        st.info("Type a question first.")
    else:
        vector_store = st.session_state["vector_store"]
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        llm = make_llm(api_key)
        chain = parallel | PROMPT | llm | StrOutputParser()

        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke(question)
                st.markdown("**Answer:**\n\n" + answer)
            except Exception as e:
                st.error(f"LLM error: {e}")

if "vector_store" in st.session_state:
    with st.expander("🔎 Preview retrieved chunks (debug)"):
        retriever = st.session_state["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k": 4})
        if question:
            try:
                docs = retriever.get_relevant_documents(question)
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}:**\n\n{d.page_content[:800]}...")
            except Exception as e:
                st.write(f"(retrieval error: {e})")

st.markdown("""
---
**Tips**
- ✅ Model is automatically downloaded from Google Drive (link hardcoded, you can change in sidebar).
- Uses your ResNeXt50 classification head.
- Normalization/denormalization match your training pipeline.
""")
