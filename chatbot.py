import streamlit as st
import openai
import os
import fitz
import requests
import time  # Added explicit import for time
import json  # For JSON handling
from functools import lru_cache
from typing import List, Optional, Dict, Any, Tuple
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
        openai.api_key = api_key  # Ensure API key is set for this instance
    
    @lru_cache(maxsize=100)
    def embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a single query string"""
        if not query or not query.strip():
            return None
            
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
        """Generate embeddings in batches with robust error handling"""
        if not texts:
            return []
            
        all_embeddings = []
        batch_size = 5  # Smaller batch size for reliability
        
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                try:
                    # Update progress
                    progress = min(i / len(texts), 1.0) if len(texts) > 0 else 0
                    progress_bar.progress(progress)
                    
                    # Get batch of texts
                    batch_texts = texts[i:i+batch_size]
                    
                    # Process each text to ensure it's valid
                    processed_texts = []
                    for text in batch_texts:
                        if not text or not isinstance(text, str):
                            processed_texts.append(" ")  # Use blank space for empty/invalid texts
                            continue
                            
                        # Truncate long texts (more conservative limit)
                        if len(text) > 4000:  
                            text = text[:4000]
                        processed_texts.append(text)
                    
                    # Skip if all texts were invalid
                    if not processed_texts or all(not text.strip() for text in processed_texts):
                        continue
                        
                    # Generate embeddings for this batch
                    response = openai.Embedding.create(
                        input=processed_texts,
                        model=self.model
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Add a delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    # Continue with next batch
                    time.sleep(1)  # Longer delay after error
            
            # Final progress update
            if progress_bar is not None:
                progress_bar.progress(1.0)
                
        except Exception as e:
            st.error(f"Error setting up progress bar: {str(e)}")
            # Fallback without progress bar
            for i in range(0, len(texts), batch_size):
                try:
                    batch_texts = texts[i:i+batch_size]
                    processed_texts = [text[:4000] if isinstance(text, str) and len(text) > 4000 else (text if isinstance(text, str) else " ") for text in batch_texts]
                    
                    if not processed_texts:
                        continue
                    
                    response = openai.Embedding.create(
                        input=processed_texts,
                        model=self.model
                    )
                    
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    all_embeddings.extend(batch_embeddings)
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Error in batch (fallback): {str(e)}")
                    time.sleep(1)
        
        return all_embeddings

# Pinecone vector store setup
def setup_pinecone(config: Dict[str, str], documents: List[Document], embedding_model: OpenAIEmbeddingModel):
    """Setup or connect to Pinecone index with better error handling"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=config['pinecone_api_key'])
        index_name = config['index_name']
        
        # Check if index exists
        try:
            indexes = pc.list_indexes()
            index_exists = hasattr(indexes, 'names') and index_name in indexes.names()
        except:
            # Try alternative approach if the first one fails
            try:
                all_indexes = pc.list_indexes().names()
                index_exists = index_name in all_indexes
            except:
                index_exists = False
                
        if not index_exists:
            st.info(f"Creating new Pinecone index: {index_name}")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-west-2')  # Changed to us-west-2
                )
                # Wait for index to be ready
                time.sleep(10)
            except Exception as e:
                st.error(f"Error creating index: {str(e)}")
                return None
        
        # Connect to index
        try:
            index = pc.Index(index_name)
        except Exception as e:
            st.error(f"Error connecting to index: {str(e)}")
            return None
        
        # Prepare documents for embedding
        if not documents:
            st.warning("No documents to embed")
            return None
            
        # Extract text content
        texts_content = []
        for doc in documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                texts_content.append(doc.page_content)
        
        if not texts_content:
            st.warning("No valid content found in documents")
            return None
        
        # Generate embeddings
        with st.spinner("Generating embeddings... This may take a while."):
            embeddings = embedding_model.generate_embeddings(texts_content)
        
        if not embeddings:
            st.error("Failed to generate embeddings")
            return None
        
        if len(embeddings) != len(texts_content):
            st.warning(f"Warning: Generated {len(embeddings)} embeddings for {len(texts_content)} documents")
            # Adjust document list to match embeddings length
            documents = documents[:len(embeddings)]
            
        # Upsert to Pinecone
        with st.spinner(f"Storing {len(embeddings)} embeddings in Pinecone..."):
            # Smaller batches for Pinecone
            batch_size = 20
            
            for i in range(0, len(embeddings), batch_size):
                end_idx = min(i + batch_size, len(embeddings))
                batch_docs = documents[i:end_idx] if i < len(documents) else []
                batch_embeddings = embeddings[i:end_idx]
                
                if not batch_docs or not batch_embeddings or len(batch_docs) != len(batch_embeddings):
                    continue
                
                vectors_to_upsert = []
                for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                    try:
                        # Create unique ID and metadata
                        doc_id = f"doc_{i+j}"
                        metadata = {"text": doc.page_content[:100] if hasattr(doc, 'page_content') else ""}
                        
                        # Add to batch
                        vectors_to_upsert.append({
                            "id": doc_id,
                            "values": embedding,
                            "metadata": metadata
                        })
                    except Exception as e:
                        st.error(f"Error processing document {i+j}: {str(e)}")
                
                # Upsert batch if we have vectors
                if vectors_to_upsert:
                    try:
                        index.upsert(vectors=vectors_to_upsert)
                        time.sleep(0.5)  # Small delay between batches
                    except Exception as e:
                        st.error(f"Error upserting batch: {str(e)}")
        
        # Create PineconeVectorStore
        return PineconeVectorStore(index, embedding_model, text_key="text")
        
    except Exception as e:
        st.error(f"Error in Pinecone setup: {str(e)}")
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
    vectorstore: Optional[PineconeVectorStore],
    bm25: Optional[BM25Okapi], 
    documents: List[Document],
    config: Dict[str, Any]
) -> str:
    """Generate a response to user input using available search methods"""
    # Input validation
    user_input = user_input.strip() if user_input else ""
    if not user_input:
        return "Input tidak boleh kosong."
    
    context = ""  # Default empty context
    
    try:
        # BM25 search if available
        if bm25:
            tokenized_query = user_input.split()
            top_bm25_docs = bm25.get_top_n(tokenized_query, documents, n=3)
        else:
            top_bm25_docs = []
        
        # Vector search if available
        vector_docs = []
        if vectorstore:
            try:
                vector_docs = vectorstore.similarity_search(user_input, k=5)
            except Exception as e:
                st.warning(f"Vector search failed: {str(e)}")
        
        # Combine results and remove duplicates
        all_docs = top_bm25_docs + vector_docs
        if all_docs:
            # Create context from documents with more robust approach
            context_parts = []
            seen_texts = set()
            
            for doc in all_docs:
                if hasattr(doc, 'page_content') and doc.page_content:
                    text = doc.page_content.strip()
                    if text and text not in seen_texts:
                        context_parts.append(text)
                        seen_texts.add(text)
            
            context = "\n\n".join(context_parts)
        
        # Check connection before making API call
        if not context:
            return "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan ini."
            
        # Generate response with OpenAI
        messages = [
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": f"Konteks: {context}\n\nPertanyaan: {user_input}"}
        ]
        
        response = openai.ChatCompletion.create(
            model=config['chat_model'],
            messages=messages,
            temperature=0.7,
            max_tokens=1000
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
    try:
        # Setup environment and load configuration
        config = setup_environment()
        
        # Load and process documents
        documents = load_documents()
        if not documents:
            st.error("No documents were loaded. Please check the data folder.")
            return
        
        # Process documents - using the fixed approach
        texts = split_documents(documents)
        if not texts:
            st.error("Failed to split documents into chunks.")
            return
        
        # Setup embedding model
        embedding_model = OpenAIEmbeddingModel(model=config['text_model'], api_key=config['openai_api_key'])
        
        # Set up BM25 first as a fallback
        tokenized_texts = [doc.page_content.split() for doc in documents if hasattr(doc, 'page_content')]
        bm25 = BM25Okapi(tokenized_texts) if tokenized_texts else None
        
        # Setup vector store - but allow the app to continue if this fails
        vectorstore = None
        try:
            st.info("Setting up Pinecone vector store...")
            vectorstore = setup_pinecone(config, texts, embedding_model)
            if vectorstore:
                st.success("Pinecone vector store setup successful!")
            else:
                st.warning("Vector store setup failed. Continuing with BM25 only.")
        except Exception as e:
            st.error(f"Failed to set up vector store: {str(e)}")
            vectorstore = None
        
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
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
