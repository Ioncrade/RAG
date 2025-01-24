# AI Document Query System

## Overview
This project is an **AI-powered Retrieval-Augmented Generation (RAG) system** that allows users to query documents uploaded as PDFs. The system processes the PDFs, stores the data in **ChromaDB**, and uses **Hugging Face embeddings** to retrieve relevant context for answering queries. The application is built using **Streamlit** for the front-end interface, **LangChain** for text processing, and **ChromaDB** for vector storage.

## Features
- Upload PDF documents and extract their text.
- Split large documents into smaller chunks for processing.
- Generate vector embeddings using Hugging Face models.
- Store and manage document embeddings using ChromaDB.
- Query the stored data and retrieve relevant information.
- Generate responses with the help of a language model.

---

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- GPU support for Hugging Face embeddings(Cuda 12.1)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ioncrade/RAG.git
   cd RAG
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up ChromaDB Directory**:
   Create a folder for persistent storage of embeddings:
   ```bash
   mkdir chroma_db
   ```

5. **Set Environment Variables**:
   Set your API key for Groq (used for generating responses):
   ```bash
   export GROQ_API_KEY=<your_api_key>  # On Windows, use `set GROQ_API_KEY=<your_api_key>`
   ```

---

## How It Works

1. **PDF Upload**:
   Users upload PDF files via the Streamlit sidebar.

2. **Text Extraction**:
   The application extracts text from the PDF using **LangChain's PyPDFLoader**.

3. **Text Chunking**:
   Extracted text is split into manageable chunks using **RecursiveCharacterTextSplitter**.

4. **Embedding Generation**:
   Each chunk is transformed into vector embeddings using **Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`** model.

5. **Data Storage**:
   The vector embeddings and corresponding text are stored in **ChromaDB** for efficient retrieval.

6. **Query Processing**:
   Users input a query, which is embedded and matched against the stored vectors in **ChromaDB** to retrieve the most relevant chunks.

7. **Response Generation**:
   The retrieved context is passed to a language model (Groq) to generate a detailed response.

---

## Code Walkthrough

### Key Functions
1. **`extract_text_from_pdf_with_loader(pdf_path)`**:
   Extracts and cleans text from a PDF file.

2. **`create_chunks(docs)`**:
   Splits text into smaller chunks with overlap to improve retrieval accuracy.

3. **`embed_text(texts)`**:
   Generates vector embeddings for a list of texts using Hugging Face embeddings.

4. **`embed_and_store_chunks(chunks)`**:
   Stores the text chunks and their embeddings in ChromaDB.

5. **`retrieve_context(query, collection_name)`**:
   Queries ChromaDB to find relevant text chunks for a given user query.

6. **`generate_response_with_context(query, context)`**:
   Uses a language model to generate a response based on the retrieved context.

---

## Streamlit Interface

### Sidebar
- **PDF Upload**: Allows users to upload a PDF document.
- **Processing Status**: Displays extraction and embedding status.

### Main Page
- **Query Input**: Text input box for user queries.
- **Response Display**: Outputs the generated response to the query.

---

## Future Enhancements
- Support for additional file formats (e.g., DOCX, TXT).
- Integration with other vector databases for scalability.
- Advanced query ranking and filtering.
- Visualization of embeddings and document relevance.

---

## Screenshots



---

## Requirements
- `chromadb`
- `langchain`
- `streamlit`
- `sentence-transformers`
- `torch`
- `transformers`

Install all requirements using:
```bash
pip install -r requirements.txt
```

---

## How to Run
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser at [http://localhost:8501](http://localhost:8501).

3. Upload a PDF, input a query, and get your answers instantly!

---

## Contributing
Feel free to contribute by submitting pull requests or reporting issues. For major changes, please open an issue to discuss your ideas.

---
