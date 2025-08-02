# qa_service.py
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from ollama_llm import OllamaLLM  # your custom class

PERSIST_DIR = "vectordb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # or similar

# 1. Load Vector Store
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

# 2. Create Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 3. Use your local LLM
llm = OllamaLLM(model_name="mistral")  # or gemma:2b for lighter testing

# 4. Create prompt template (optional)
template = """Use the following context to answer the question. Be concise and accurate.

Context:
{context}

Question:
{question}

Helpful Answer:"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# 5. RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 6. Function to query
def ask_question(query: str) -> str:
    result = rag_chain.invoke({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])
    source_names = [doc.metadata.get("source", "unknown") for doc in sources]
    return f"Answer: {answer}\n\nSources: {source_names}"
