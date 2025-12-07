import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Ancient Manuscript Summarizer", layout="wide")

# ---------------------------
# CUSTOM MANUSCRIPT CSS
# ---------------------------
st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background-color: #f4e7c5;
        background-image: url("https://i.pinimg.com/736x/7b/8f/b8/7b8fb81af2cd095a71f8f91542185c53.jpg");
        background-size: cover;
        background-attachment: fixed;
    }

    /* Main container styling */
    .block-container {
        background: rgba(244, 231, 197, 0.95);
        border-radius: 20px;
        padding: 2.5rem;
        margin-top:100px;
        max-width: 1100px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.4);
        border: 5px solid #8b5a2b;
    }

    /* Title */
    h1 {
        font-family: "Papyrus", "Cursive";
        color: #5b2c02;
        text-align: center;
        letter-spacing: 2px;
        font-size: 42px;
    }

    /* Sub text */
    p, label, span {
        font-family: "Georgia", serif !important;
        font-size: 18px !important;
        color: #3b1e04 !important;
    }

    /* Text area */
    textarea {
        background-color: #fff2cc !important;
        border: 3px solid #8b5a2b !important;
        border-radius: 12px !important;
        font-family: "Georgia", serif !important;
        font-size: 17px !important;
        color: #3b1e04 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #8b5a2b, #d2a679);
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 12px;
        border: none;
        font-family: "Papyrus", "Cursive";
        letter-spacing: 1px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(to right, #d2a679, #8b5a2b);
        transform: scale(1.03);
    }

    /* Summary Manuscript Output Box */
    .manuscript-box {
        background: url("./image.png");
        background-size: cover;
        padding: 35px;
        border-radius: 18px;
        border: 5px solid #8b5a2b;
        font-family: "Georgia", serif;
        font-size: 20px;
        color: #3b1e04;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.5);
        line-height: 1.7;
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0px); }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ---------------------------
# TEXT CHUNKING
# ---------------------------
def chunk_text(text, max_words=600):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i : i + max_words])

def summarize_text(article):
    chunk_summaries = []

    for chunk in chunk_text(article):
        input_text = "summarize: " + chunk
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(chunk_summary)

    final_input = "summarize: " + " ".join(chunk_summaries)
    final_ids = model.generate(
        tokenizer.encode(final_input, return_tensors="pt").to(device),
        max_length=200,
        min_length=50,
        num_beams=4,
        early_stopping=True
    )

    final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    return final_summary

# ---------------------------
# UI LAYOUT
# ---------------------------
st.title("📜 Text Summarizer")
st.write("Paste your text below and receive an elegant summary.")

article = st.text_area("📜 Enter Your Text:", height=280)

if article:
    st.info(f"🖋️ **Article Length:** {len(article)} characters")

if st.button("🔮 Reveal Summary"):
    if len(article.strip()) == 0:
        st.warning("Please enter some sacred text to summarize.")
    else:
        with st.spinner("📜 Inscribing..."):
            summary = summarize_text(article)

        st.success("✨ Your summary is Ready!")
        st.markdown("## 🪶 Here is your Summary:")

        st.markdown(
            f"""
            <div class="manuscript-box">
            {summary}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(f"📏 **Summary Length:** {len(summary)} characters")
