
# 🧠 RAG Chatbot with LangChain, HuggingFace, FAISS & Streamlit

This is a Retrieval-Augmented Generation (RAG) based chatbot capable of answering document-based queries using state-of-the-art language models and vector search techniques.

It supports multi-file document upload (PDF, DOCX, TXT), document chunking, semantic embedding, vector storage, and high-quality answer generation using **LLaMA-3 via Groq API**.

---

## 🔧 Features

- 📄 Upload multiple files (PDF, DOCX, TXT)
- 🔍 Chunk and embed documents using HuggingFace embeddings
- 📚 Store and retrieve chunks using **FAISS vector store**
- 🧠 Query using **LangChain** with **LLaMA-3 (via Groq API)**
- 📊 Evaluate using **RAGAS** metrics
- 🎛️ Clean and interactive **Streamlit UI**

---

## 📁 Project Structure

```
RAG_CHATBOT/
│
├── app.py                # Streamlit app entry point
├── app2.py               # (Optional) alternate version or test version
├── chat.py               # Core logic to handle LLM chat
├── modules/              # Custom utilities (e.g., loading, chunking, embedding)
│   └── utils.py
├── requirements.txt      # All dependencies
├── workflow.txt          # Project design or notes
├── .gitignore
└── README.md             # You're here!
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your environment variables (e.g., in a `.env` file)
```env
GROQ_API_KEY=your_api_key_here
```

### 5. Launch the chatbot
```bash
streamlit run app.py
```

---

## 🔐 Tech Stack

| Area               | Tools Used                                 |
|--------------------|---------------------------------------------|
| Embedding          | HuggingFace Sentence Transformers          |
| Vector Store       | FAISS (Facebook AI Similarity Search)      |
| LLM                | LLaMA-3 via Groq API                        |
| Frameworks         | LangChain, Streamlit                       |
| Evaluation         | RAGAS (Retrieval-Augmented Generation Eval)|

---

## 📊 Sample Use Cases

- 📚 Chat with long research documents
- 🧾 Legal document Q&A
- 🏫 Educational assistant for textbooks
- 🧠 Knowledge retrieval from reports

---

## 👨‍💻 Author

**Harnaik Singh Sahni**  
[LinkedIn](https://www.linkedin.com/in/harnaik-singh) • [GitHub](https://github.com/harnaiksahni)

---

## 📝 License

This project is for educational and demonstration purposes. For commercial or large-scale use, please ensure proper handling of data, APIs, and LLM licensing.

---

## 🌟 Contribute

Feel free to fork, star ⭐, and open issues or pull requests!
