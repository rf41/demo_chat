import streamlit as st
import openai
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
import requests
import base64
from concurrent.futures import ThreadPoolExecutor

# Add this near the top of your file
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}

# Set environment variables from secrets
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# Decode the OpenAI API base URL from secrets
OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY

# Set the custom base URL for OpenAI
openai.api_base = OPENAI_API_BASE

# Memuat dan memproses dokumen PDF
data_folder = "data"
documents = []
for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        with fitz.open(os.path.join(data_folder, file)) as doc:
            text = "".join([page.get_text("text").strip() for page in doc])
            documents.append(Document(page_content=text))

# Memecah teks menjadi chunk kecil
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# Create or load the Pinecone index
index_name = st.secrets["PINECONE_INDEX_NAME"]  # Replace with your index name
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust dimension as needed
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west1')  # Change to a supported region
    )

index = pc.Index(index_name)

# Generate embeddings using OpenAI
def generate_embeddings(texts):
    try:
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input texts must be a list of strings.")
        print(f"Generating embeddings for texts: {texts}")  # Debug statement
        response = openai.Embedding.create(
            input=texts,
            model=st.secrets["TEXT_MODEL"]
        )
        embeddings = [embedding['embedding'] for embedding in response['data']]
        print(f"Generated embeddings: {embeddings}")  # Debug statement
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# Membuat atau memuat Pinecone index
texts_content = [doc.page_content for doc in documents]
print(f"texts_content: {texts_content}")  # Debug statement
embeddings = generate_embeddings(texts_content)

# Ensure embeddings are added to Pinecone
if embeddings:
    for i, embedding in enumerate(embeddings):
        index.upsert([(str(i), embedding)])
    print("Embeddings successfully upserted to Pinecone.")

# Initialize PineconeVectorStore with the correct embedding model
class OpenAIEmbeddingModel:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        openai.api_key = api_key

    # Modify your embed_query method in OpenAIEmbeddingModel
    def embed_query(self, query):
        try:
            # Check if query is in cache
            if query in st.session_state.query_cache:
                return st.session_state.query_cache[query]
                
            if not isinstance(query, str):
                raise ValueError("Query must be a string.")
            print(f"Embedding query: {query}")  # Debug statement
            response = openai.Embedding.create(
                input=[query],
                model=self.model
            )
            embedding = response['data'][0]['embedding']
            # Cache the result
            st.session_state.query_cache[query] = embedding
            return embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            return None

embedding_model = OpenAIEmbeddingModel(model=st.secrets["TEXT_MODEL"], api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(index, embedding_model, text_key="page_content")

# Initialize BM25
tokenized_texts = [doc.page_content.split() for doc in documents]
bm25 = BM25Okapi(tokenized_texts)

def check_openai_connection():
    try:
        response = requests.get(f"{OPENAI_API_BASE}", timeout=5)
        if response.status_code == 200:
            return True
    except requests.RequestException as e:
        print(f"Koneksi ke model gagal: {e}")
        return False

# New helper functions
def bm25_search(query):
    tokenized_query = query.split()
    return bm25.get_top_n(tokenized_query, documents, n=3)
    
def vector_search(query):
    if vectorstore and query:
        return vectorstore.similarity_search(query, k=10)
    return []

# Modify the chatbot_response function
def chatbot_response(user_input):
    user_input = user_input.strip()
    if not user_input:
        return "Input tidak boleh kosong."
    
    # Use ThreadPoolExecutor to run searches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both search tasks
        bm25_future = executor.submit(bm25_search, user_input)
        vector_future = executor.submit(vector_search, user_input)
        
        # Show detailed progress
        loading_placeholder.markdown("⏳ Searching knowledge base...")
        
        # Get results
        top_n = bm25_future.result()
        retrieved_docs = vector_future.result()
        
        # Show detailed progress
        loading_placeholder.markdown("⏳ Analyzing relevant information...")
    
    # Combine BM25 and vectorstore results
    combined_results = top_n + retrieved_docs
    combined_results = list({doc.page_content: doc for doc in combined_results}.values())  # Remove duplicates based on page_content
    
    # Update context with combined results
    context = "".join([doc.page_content for doc in combined_results if hasattr(doc, 'page_content')])
    system_prompt = st.secrets["SYSTEM_PROMPT"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Konteks: {context}\nPertanyaan: {user_input}"}
    ]
    try:
        if not check_openai_connection():
            return "Koneksi ke model gagal. Silakan coba lagi nanti."
        
        # Show detailed progress
        loading_placeholder.markdown("⏳ Generating response...")
        
        response = openai.ChatCompletion.create(
            model=st.secrets["MODEL"],
            messages=messages
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

# Streamlit UI
st.title("Ridwan's Chatbot")
st.write("Ask me anything about my profile. All information is sourced from my LinkedIn CV and Posts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS untuk mempercantik tampilan chat (seperti ChatGPT)
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
    max-width: 800px;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    position: relative;
    font-size: 1rem;
    max-width: 80%;
}

.chat-message.user {
    background-color: #1e88e5;
    color: white;
    border: 1px solid #106cc8;
    align-self: flex-end;
    margin-left: auto;
    border-top-right-radius: 0;
}

.chat-message.bot {
    background-color: #f0f0f0;
    color: #333;
    border: 1px solid #e0e0e0;
    align-self: flex-start;
    margin-right: auto;
    border-top-left-radius: 0;
}

.chat-message .message-content {
    display: block;
    margin-top: 0;
    width: 100%;
    white-space: pre-wrap;
    overflow-wrap: break-word;
}

.chat-message .message-header {
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

.chat-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding-bottom: 7rem;
    margin-bottom: 60px;
}

.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: white;
    padding: 1rem 1rem 1.5rem 1rem;
    z-index: 1000;
    border-top: 1px solid #e5e5e5;
    box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: center;
}

.input-container form {
    max-width: 800px;
    width: 100%;
}

.stTextInput input {
    border-radius: 0.75rem !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    border: 1px solid #e5e5e5 !important;
}

.stButton button {
    border-radius: 0.75rem !important;
    padding: 0.25rem 1rem !important;
}

.stForm {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[data-testid="stFormSubmitButton"] > button {
    visibility: hidden;
    position: absolute;
}

/* Fix untuk memastikan chat tidak tertutup input */
.main {
    padding-bottom: 70px;
}

footer {
    visibility: hidden;
}

.stApp {
    margin-bottom: 60px;
}
</style>
""", unsafe_allow_html=True)

# Chat container with proper spacing for fixed input
st.markdown('<div style="height: calc(100vh - 200px); overflow-y: auto; padding-bottom: 80px;">', unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display messages in chronological order (oldest first)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message-header">You</div>
            <div class="message-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Proses konten bot untuk memastikan paragraf baru tampil dengan benar
        content = message['content'].replace('\n', '<br>')
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="message-header">Bot</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Add a placeholder for loading messages
loading_placeholder = st.empty()

# Input form at the bottom - dengan wrapper tambahan untuk memastikan posisi fixed
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: white; z-index: 9999;">
    <div class='input-container'>
        <div style="width: 100%; max-width: 800px;">
""", unsafe_allow_html=True)

# Input form
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("", placeholder="Type your message here...", key="input")
with col2:
    submit_button = st.button("Send", key="send")
    
st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if submit_button and user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for the loading message
    loading_placeholder.markdown("""
    <div style="display: flex; justify-content: center; margin: 1rem;">
        <div style="background-color: #f0f0f0; padding: 0.5rem 1rem; border-radius: 1rem;">
            ⏳ Reading the CV, please wait...
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get response
    response = chatbot_response(user_input)
    
    # Remove loading message
    loading_placeholder.empty()
    
    # Add bot response to session state
    st.session_state.messages.append({"role": "bot", "content": response})
    
    # Trigger a rerun to refresh the chat display
    st.experimental_rerun()