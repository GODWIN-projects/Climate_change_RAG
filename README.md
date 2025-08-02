# 🧠 Local RAG Chatbot with LangChain, Ollama, and ChromaDB

A local, privacy-friendly Retrieval-Augmented Generation (RAG) chatbot powered by [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [ChromaDB](https://www.trychroma.com/). This app lets you ingest PDF, TXT, or Markdown files into a local vector store and ask questions using a local language model.

---

## ✨ Features

- 💬 Natural language querying over your documents  
- 📄 Document ingestion from PDF, TXT, or Markdown  
- ⚡ Fast, local LLM inference via Ollama (`mistral`, etc.)  
- 🔍 Vector similarity search using ChromaDB  
- 🧠 Embedding via HuggingFace `sentence-transformers`

---

## 📦 Tech Stack

| Component     | Tool                      |
|---------------|---------------------------|
| LLM           | Ollama (`mistral`)        |
| Framework     | LangChain                 |
| Embeddings    | HuggingFace Transformers  |
| Vector Store  | ChromaDB                  |
| Interface     | Command Line              |
| Language      | Python                    |

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```
###2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

    On Windows:

```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
    source venv/bin/activate
```
###3. Install dependencies
```bash
pip install -r requirements.txt
```
###4. Start Ollama and download a model

Make sure Ollama is installed and running:
```bash
ollama run mistral
```
You can replace mistral with any supported model like gemma, llama3, etc.

##🔧 Usage
###1. Add your documents

Place your .pdf, .txt, or .md files inside the data/ folder.
###2. Ingest documents
```bash
python ingest.py
```
This will chunk, embed, and store the content into a Chroma vector DB.
###3. Ask questions
```bash
python test_query.py
```
You'll be able to type a natural language question and get an answer based on the content you've ingested.
🧠 Example

> how does climate change affect newborns?

