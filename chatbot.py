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

# Custom CSS untuk mempercantik tampilan chat
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #e6f7ff;
    border-left: 5px solid #2196F3;
}
.chat-message.bot {
    background-color: #f0f0f0;
    border-left: 5px solid #4CAF50;
}
.chat-message .message-content {
    display: block;  /* Changed from flex to block */
    margin-top: 0;
}
.chat-message .message-header {
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.5rem;
}
.chat-container {
    display: flex;
    flex-direction: column-reverse;
}
.st-emotion-cache-12fmjuu{
    display: none!important;
}
header{
    display: none!important;
}
</style>
""", unsafe_allow_html=True)

# Input form in the center
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
with st.form(key='chat_form'):
    user_input = st.text_input("Your question:", 
                          key="input", 
                          placeholder="Ask me about Ridwan's experience, skills, or background...")
    submit_button = st.form_submit_button(label='Ask')
st.markdown("</div>", unsafe_allow_html=True)

# Modify the Submit button handling section
if submit_button and user_input:
    # Clear previous messages before adding new ones
    st.session_state.messages = []
    
    # Add only the user input to session state (for processing)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for the loading message
    loading_placeholder = st.empty()
    loading_placeholder.markdown("⏳ Reading the CV, please wait.")
    
    # Get response
    response = chatbot_response(user_input)
    
    # Remove loading message
    loading_placeholder.empty()
    
    # Add only the bot response to display
    st.session_state.messages = [{"role": "bot", "content": response}]

# Display messages section - modify to only show bot responses
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display only bot messages
for message in st.session_state.messages:
    if message["role"] == "bot":
        # Create a container for each message
        message_container = st.container()
        
        # Use a vertical layout instead of columns
        with message_container:
            # Add separator for visual distinction
            st.markdown("<hr style='margin: 10px 0; opacity: 0.8'> </hr>", unsafe_allow_html=True)

            # Display "Bot" label above the message
            st.markdown("<div style='font-weight: bold; margin-bottom: 5px;'>Answer:</div>", unsafe_allow_html=True)
            
            # Display the message content with markdown support
            st.markdown(message['content'])
            
            # Add separator for visual distinction
            st.markdown("<hr style='margin: 10px 0; opacity: 0.8'> </hr>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)