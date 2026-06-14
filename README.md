# 📜 AI Text Summarizer (Transformer-Based with Streamlit UI)

This project is an **AI-powered Text Summarizer** built using the **T5 Transformer model** and a **Streamlit web interface**.  
It automatically converts long articles into short, meaningful summaries while preserving the core information.  
The project also supports **long-text summarization using chunking** and features a **manuscript-style UI design**.

---

## 🚀 Features

- ✅ Abstractive text summarization using **T5-base**
- ✅ Handles **long documents using chunking**
- ✅ **Hierarchical summarization** for large inputs
- ✅ Interactive **Streamlit web interface**
- ✅ Custom **ancient manuscript-style UI**
- ✅ Displays:
  - Article length
  - Summary length
- ✅ CPU & GPU compatible
- ✅ Ready for **real-world deployment**

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Model:** T5 (Text-to-Text Transfer Transformer)  
- **Deep Learning Framework:** PyTorch  
- **NLP Library:** HuggingFace Transformers  
- **Evaluation Metric:** ROUGE  
- **Styling:** Custom CSS with manuscript background  

---
## 📂 Project Structure
```bash
Text-Summarizer/
│
├── app.py # Streamlit main application
├── image.png # Manuscript background image
├── requirements.txt # Required Python packages
└── README.md # Project documentation
```
---

## ⚙️ Installation & Setup

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/sahithimeneni06/textsummarizer.git
cd text-summarizer
```
### ✅ Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv summarizer_env
summarizer_env\Scripts\activate   # For Windows
```
### ✅ Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run the Application
```bash
streamlit run app.py
```
---
