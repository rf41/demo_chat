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

# CSS yang lebih robust untuk responsivitas dan floating input
st.markdown("""
<style>
/* Container utama */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 8rem !important; /* Pastikan ada ruang untuk input */
    max-width: 800px;
}

/* Batas utama untuk chat */
.chat-area {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 180px);
    overflow-y: auto;
    padding-bottom: 100px;
    margin-bottom: 0;
    position: relative;
}

/* Container chat */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding-bottom: 1rem;
    width: 100%;
}

/* Pesan chat */
.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 0.5rem;
    position: relative;
    font-size: 1rem;
    max-width: 85%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Pesan user */
.chat-message.user {
    background-color: #1e88e5;
    color: white;
    border: none;
    align-self: flex-end;
    margin-left: auto;
    border-top-right-radius: 0.2rem;
}

/* Pesan bot */
.chat-message.bot {
    background-color: #f5f5f5;
    color: #333;
    border: none;
    align-self: flex-start;
    margin-right: auto;
    border-top-left-radius: 0.2rem;
}

/* Konten pesan */
.chat-message .message-content {
    display: block;
    width: 100%;
    white-space: pre-wrap;
    overflow-wrap: break-word;
}

/* Header pesan */
.chat-message .message-header {
    font-size: 0.8rem;
    font-weight: bold;
    margin-bottom: 0.3rem;
    opacity: 0.8;
}

/* Input container dengan posisi fixed */
.input-floating {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: white;
    padding: 1rem;
    z-index: 1000;
    border-top: 1px solid #e0e0e0;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}

/* Container input di tengah */
.input-center {
    max-width: 800px;
    margin: 0 auto;
    display: flex;
}

/* Style untuk input field */
.stTextInput {
    flex-grow: 1;
}

.stTextInput input {
    border-radius: 20px !important;
    padding: 0.75rem 1.2rem !important;
    border: 1px solid #ddd !important;
    box-shadow: none !important;
    font-size: 1rem !important;
}

/* Style untuk send button */
.send-button {
    margin-left: 8px;
    align-self: center;
}

.send-button button {
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: #1e88e5 !important;
    color: white !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
}

/* Hide Streamlit elements */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Pastikan stApp container memiliki overflow yang benar */
.stApp {
    overflow: hidden;
}

/* Loading indicator */
.loading-indicator {
    display: flex;
    justify-content: center;
    margin: 1rem;
}

.loading-bubble {
    background-color: #f0f0f0;
    padding: 0.5rem 1rem;
    border-radius: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Responsivitas untuk layar kecil */
@media (max-width: 640px) {
    .chat-message {
        max-width: 90%;
        padding: 0.75rem;
    }
    
    .input-floating {
        padding: 0.75rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Main chat area with scroll
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message-header">You</div>
            <div class="message-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format content properly
        content = message['content'].replace('\n', '<br>')
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="message-header">Bot</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
st.markdown('</div>', unsafe_allow_html=True)  # Close chat-area

# Placeholder for loading message
loading_placeholder = st.empty()

# Custom HTML untuk input floating
st.markdown("""
<div class="input-floating">
    <div class="input-center">
    <!-- Input form diatur melalui Streamlit -->
    </div>
</div>
""", unsafe_allow_html=True)

# Input form - dibuat terpisah dari HTML di atas
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("", placeholder="Type your message here...", key="input", label_visibility="collapsed")
with col2:
    # HTML untuk tombol kirim kustom
    submit_button = st.button("➤", key="send")

if submit_button and user_input:
    # Simpan input saat ini
    current_input = user_input
    
    # Tambahkan pesan user ke session state
    st.session_state.messages.append({"role": "user", "content": current_input})
    
    # Reset input field
    st.session_state.input = ""
    
    # Tampilkan loading
    loading_placeholder.markdown("""
    <div class="loading-indicator">
        <div class="loading-bubble">
            ⏳ Processing your request...
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dapatkan respons
    response = chatbot_response(current_input)
    
    # Hapus loading
    loading_placeholder.empty()
    
    # Tambahkan respons bot
    st.session_state.messages.append({"role": "bot", "content": response})
    
    # Refresh halaman
    st.experimental_rerun()

# JavaScript untuk scroll ke pesan terbaru dan fokus pada input
st.markdown("""
<script>
    // Tunggu halaman dimuat
    window.addEventListener('load', function() {
        // Scroll ke bawah untuk melihat pesan terbaru
        const chatArea = document.querySelector('.chat-area');
        if (chatArea) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        // Fokus pada input
        setTimeout(function() {
            const inputElement = document.querySelector('input[data-baseweb="input"]');
            if (inputElement) {
                inputElement.focus();
            }
        }, 500);
    });
</script>
""", unsafe_allow_html=True)