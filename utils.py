import os, json, requests
from typing import Dict, Tuple, Optional, List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import gdown
import wikipediaapi

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ----------------------------
# Google Drive downloads
# ----------------------------
def download_from_drive(url: str, output_path: str) -> bool:
    """Download a file from Google Drive."""
    try:
        if os.path.exists(output_path):
            return True
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        return os.path.exists(output_path)
    except Exception:
        return False


def download_folder_from_drive(folder_url: str, output_dir: str) -> bool:
    """Download a folder (sample images) from Google Drive."""
    try:
        if os.path.exists(output_dir) and os.listdir(output_dir):
            return True
        gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
        return os.path.exists(output_dir) and os.listdir(output_dir)
    except Exception:
        return False


# ----------------------------
# Model Utilities
# ----------------------------
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
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.to(device)
    return model


def get_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


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
# Wikipedia Utilities
# ----------------------------
WIKI = wikipediaapi.Wikipedia(language="en", user_agent="Bird-RAG-App/0.1")

def mw_search_titles(query: str, limit: int = 5) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
    headers = {"User-Agent": "Bird-RAG-App/0.1"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        return [hit.get("title", "") for hit in data.get("query", {}).get("search", [])]
    except Exception:
        return []


def fetch_wikipedia_text(title: str) -> Tuple[str, Optional[str]]:
    page = WIKI.page(title)
    if page.exists():
        return page.text, title
    return "", None


# ----------------------------
# RAG Utilities
# ----------------------------
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


def make_rag_chain(retriever, api_key: str):
    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    })
    llm = make_llm(api_key)
    return parallel | PROMPT | llm | StrOutputParser()
