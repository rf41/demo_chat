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

# Checkbox to use knowledge base (default checked and disabled)
# use_knowledge_base = st.checkbox("Use Knowledge Base", value=True, disabled=True)

# Input form in the center
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
with st.form(key='chat_form'):
    user_input = st.text_input("You:", key="input")
    submit_button = st.form_submit_button(label='Send')
st.markdown("</div>", unsafe_allow_html=True)

if submit_button and user_input:
    # Add to existing messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for the loading message
    loading_placeholder = st.empty()
    loading_placeholder.markdown("⏳ Reading the CV, please wait.")
    
    # Get response
    response = chatbot_response(user_input)
    
    # Remove loading message
    loading_placeholder.empty()
    
    st.session_state.messages.append({"role": "bot", "content": response})

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Bot: {message['content']}")