import os
import re
import chromadb
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from transformers import AutoTokenizer, AutoModel
from chromadb.config import Settings
from chromadb import Client
import chromadb.utils.embedding_functions as embedding_functions
import torch

collection_name="document_collection"
persist_directory=r"chroma_db"
client = chromadb.PersistentClient(path=persist_directory)


def extract_text_from_pdf_with_loader(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    docs = []
    docs_lazy = loader.lazy_load()
    
    for idx, doc in enumerate(docs_lazy):
        if isinstance(doc, Document):
            raw_content = doc.page_content
            cleaned_content = re.sub(r'[^\x20-\x7E]+', '', raw_content)
            docs.append(cleaned_content)
        else:
            raise ValueError("Unexpected document format encountered.")
    return docs

# Create text chunks
def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(
        [Document(page_content=doc) for doc in docs]
    )
    return chunks

def embed_text(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    return vectors


# Embed and store chunks in ChromaDB
def embed_and_store_chunks(chunks):
    persist_directory=r"chroma_db"
    # Initialize ChromaDB client with explicit settings
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get collection
    collection = client.get_or_create_collection(name=collection_name)

    
    vectors = embed_text(chunks)

    # Prepare data for upsert
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"ID{i}" for i in range(len(chunks))]
    
    # Upsert into ChromaDB
    collection.upsert(
        documents=documents,
        embeddings=vectors,
        ids=ids
    )
    
    return collection
    
# Retrieve context from ChromaDB
def retrieve_context(query, collection_name):
    collection = client.get_collection(name=collection_name)
    query_embedding = embed_text([query])[0]
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    if "documents" not in results:
        raise ValueError("Query results do not contain 'documents'. Debugging response: {}".format(results))

    # Flatten the results["documents"] and extract the context
    documents = results["documents"]
    context = " ".join(doc for doc_list in documents for doc in doc_list)  # Flatten nested lists

    return context

# Generate response with context using the LLM
def generate_response_with_context(query, context):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
            {"role": "assistant", "content": f"Context: {context}"},
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content


# Streamlit app
st.set_page_config(page_title="AI Document Query System", layout="wide")

# Streamlit session state for outputs
if "outputs" not in st.session_state:
    st.session_state["outputs"] = []

# UI for PDF Upload and Query
st.title("AI Document Query System 📄🤖")
st.sidebar.header("Upload PDF and Query")

uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text from the PDF..."):
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        text_list = extract_text_from_pdf_with_loader(pdf_path)
        st.sidebar.success("Text extracted successfully!")

        with st.spinner("Processing text and storing embeddings..."):
            chunks = create_chunks(text_list)
            st.sidebar.success("Text chunked successfully!")

            collect=embed_and_store_chunks(chunks)
            st.write(collect)
        st.sidebar.success("Text embedded and stored in the database!")

st.header("Ask a Question")
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a query.")
    elif not uploaded_file:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Retrieving context and generating response..."):
            context = retrieve_context(query,collection_name)
            response = generate_response_with_context(query, context)
            st.markdown("### Response")
            st.write(response)
            st.session_state["outputs"].append({"query": query, "response": response})

# if st.session_state["outputs"]:
#     st.markdown("## Previous Interactions")
#     for i, interaction in enumerate(st.session_state["outputs"]):
#         with st.expander(f"Interaction {i + 1}"):
#             st.write(f"**Query:** {interaction['query']}")
#             st.write(f"**Response:** {interaction['response']}")