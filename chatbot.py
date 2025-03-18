import streamlit as st
import openai
import os
import fitz
import requests
from functools import lru_cache
from typing import List, Optional, Dict, Any
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

# Configuration and environment setup
def setup_environment():
    # Set environment variables from secrets
    os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
    
    # OpenAI configuration
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = st.secrets["OPENAI_API_BASE"]
    
    return {
        'pinecone_api_key': os.environ['PINECONE_API_KEY'],
        'openai_api_key': os.environ['OPENAI_API_KEY'],
        'openai_api_base': st.secrets["OPENAI_API_BASE"],
        'index_name': st.secrets["PINECONE_INDEX_NAME"],
        'text_model': st.secrets["TEXT_MODEL"],
        'chat_model': st.secrets["MODEL"],
        'system_prompt': st.secrets["SYSTEM_PROMPT"]
    }

# Document loading and processing
@st.cache_data
def load_documents(data_folder: str = "data") -> List[Document]:
    """Load and process PDF documents from the data folder"""
    documents = []
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            try:
                with fitz.open(os.path.join(data_folder, file)) as doc:
                    text = "".join([page.get_text("text").strip() for page in doc])
                    documents.append(Document(page_content=text))
            except Exception as e:
                st.warning(f"Error loading {file}: {str(e)}")
    return documents

# Fix for the unhashable type error
@st.cache_data
def split_text_documents(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split a single text document into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Split documents into smaller chunks"""
    all_splits = []
    # Process each document individually
    for doc in documents:
        # Split text using cached function
        splits = split_text_documents(doc.page_content, chunk_size, chunk_overlap)
        # Convert back to Documents and add to results
        all_splits.extend([Document(page_content=text) for text in splits])
    return all_splits

# Vector embeddings
class OpenAIEmbeddingModel:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.max_tokens_per_batch = 8000  # OpenAI's limit for text-embedding models
    
    @lru_cache(maxsize=100)
    def embed_query(self, query: str) -> Optional[List[float]]:
        try:
            response = openai.Embedding.create(
                input=[query],
                model=self.model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            st.error(f"Error embedding query: {str(e)}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches to avoid token limits"""
        all_embeddings = []
        
        # Process in smaller batches
        batch_size = 16  # Process fewer documents at once
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Check if any texts are too long and truncate if necessary
                processed_texts = []
                for text in batch_texts:
                    # Approximate token count (rough estimate)
                    if len(text) > 5000:  # Approximately 4000 tokens
                        text = text[:5000]  # Truncate long texts
                    processed_texts.append(text)
                
                # Generate embeddings for this batch
                response = openai.Embedding.create(
                    input=processed_texts,
                    model=self.model
                )
                
                # Extract embeddings
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                
                # Add a small delay to avoid rate limiting
                if i + batch_size < len(texts):
                    import time
                    time.sleep(0.5)
                    
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            # Return partial embeddings if we got any
            if not all_embeddings:
                return []
        
        return all_embeddings

# Pinecone vector store setup
def setup_pinecone(config: Dict[str, str], documents: List[Document], embedding_model: OpenAIEmbeddingModel):
    """Setup or connect to Pinecone index"""
    try:
        pc = Pinecone(api_key=config['pinecone_api_key'])
        index_name = config['index_name']
        
        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west1')
            )
        
        index = pc.Index(index_name)
        
        # Generate and store embeddings
        texts_content = [doc.page_content for doc in documents]
        
        # Show progress for embedding generation
        with st.spinner("Generating embeddings... This may take a while."):
            embeddings = embedding_model.generate_embeddings(texts_content)
        
        # Upsert embeddings to Pinecone
        if embeddings:
            if len(embeddings) != len(texts_content):
                st.warning(f"Warning: Generated {len(embeddings)} embeddings for {len(texts_content)} documents. Some documents may be missing.")
                
            # Upsert available embeddings
            with st.spinner("Storing embeddings in Pinecone..."):
                batch_size = 50  # Smaller batch size to avoid timeouts
                for i in range(0, len(embeddings), batch_size):
                    batch_docs = documents[i:min(i+batch_size, len(embeddings))]
                    batch_embeddings = embeddings[i:i+batch_size]
                    
                    batch_items = []
                    for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                        # Create a unique ID for each document chunk
                        doc_id = f"doc_{i+j}"
                        # Include metadata with the document
                        metadata = {"text": doc.page_content[:100]}  # Include a preview of content
                        batch_items.append((doc_id, embedding, metadata))
                    
                    # Upsert this batch
                    index.upsert(vectors=[(id, vector, meta) for id, vector, meta in batch_items])
        
        return PineconeVectorStore(index, embedding_model, text_key="page_content")
    
    except Exception as e:
        st.error(f"Error setting up Pinecone: {str(e)}")
        return None

def check_openai_connection(api_base: str) -> bool:
    """Check if OpenAI API is accessible"""
    try:
        response = requests.get(api_base, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Chatbot response generation
def generate_response(
    user_input: str, 
    vectorstore: PineconeVectorStore,
    bm25: BM25Okapi, 
    documents: List[Document],
    config: Dict[str, Any]
) -> str:
    """Generate a response to user input using both BM25 and vector search"""
    user_input = user_input.strip()
    if not user_input:
        return "Input tidak boleh kosong."
    
    # BM25 search
    tokenized_query = user_input.split()
    top_bm25_docs = bm25.get_top_n(tokenized_query, documents, n=3)
    
    # Vector search
    vector_docs = []
    if vectorstore:
        vector_docs = vectorstore.similarity_search(user_input, k=10)
    
    # Combine results and remove duplicates
    all_docs = top_bm25_docs + vector_docs
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    
    # Create context from combined documents
    context = "".join([doc.page_content for doc in unique_docs if hasattr(doc, 'page_content')])
    
    # Check connection before making API call
    if not check_openai_connection(config['openai_api_base']):
        return "Koneksi ke model gagal. Silakan coba lagi nanti."
    
    # Generate response with OpenAI
    try:
        messages = [
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": f"Konteks: {context}\nPertanyaan: {user_input}"}
        ]
        response = openai.ChatCompletion.create(
            model=config['chat_model'],
            messages=messages
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

# Streamlit UI
def create_ui():
    st.title("Ridwan's Chatbot")
    st.write("Ask me anything about my profile. All information is sourced from my LinkedIn CV and Posts.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Input form in the center
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    with st.form(key='chat_form'):
        user_input = st.text_input("You:", key="input")
        submit_button = st.form_submit_button(label='Send')
    st.markdown("</div>", unsafe_allow_html=True)
    
    return user_input, submit_button

def main():
    # Setup environment and load configuration
    config = setup_environment()
    
    # Load and process documents
    documents = load_documents()
    
    # Process documents - using the fixed approach
    texts = split_documents(documents)
    
    # Setup embedding model
    embedding_model = OpenAIEmbeddingModel(model=config['text_model'], api_key=config['openai_api_key'])
    
    # Setup vector store and BM25
    vectorstore = setup_pinecone(config, texts, embedding_model)
    
    # Set up BM25 regardless of vector store success
    tokenized_texts = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_texts)
    
    # Create UI 
    user_input, submit_button = create_ui()
    
    if submit_button and user_input:
        # Clear previous messages
        st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a placeholder for the loading message
        loading_placeholder = st.empty()
        loading_placeholder.markdown("⏳ Reading the CV and Posts, please wait.")
        
        # Get response
        response = generate_response(user_input, vectorstore, bm25, documents, config)
        
        # Remove loading message
        loading_placeholder.empty()
        
        st.session_state.messages.append({"role": "bot", "content": response})
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")

if __name__ == "__main__":
    main()
