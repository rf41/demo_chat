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
            model=st.secrets["TEXT_MODEL "]
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

    def embed_query(self, query):
        try:
            if not isinstance(query, str):
                raise ValueError("Query must be a string.")
            print(f"Embedding query: {query}")  # Debug statement
            response = openai.Embedding.create(
                input=[query],
                model=self.model
            )
            embedding = response['data'][0]['embedding']
            print(f"Generated query embedding: {embedding}")  # Debug statement
            return embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            return None

embedding_model = OpenAIEmbeddingModel(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(index, embedding_model, text_key="page_content")

# Initialize BM25
tokenized_texts = [doc.page_content.split() for doc in documents]
bm25 = BM25Okapi(tokenized_texts)

def check_openai_connection():
    try:
        response = requests.get(f"{OPENAI_API_BASE}/v1/engines", timeout=5)
        if response.status_code == 200:
            return True
    except requests.RequestException as e:
        print(f"Koneksi ke OpenAI gagal: {e}")
        return False

# Inisialisasi LLM
def chatbot_response(user_input):
    user_input = user_input.strip()  # Bersihkan input
    if not user_input:
        return "Input tidak boleh kosong."
    
    # BM25 search
    tokenized_query = user_input.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = bm25.get_top_n(tokenized_query, documents, n=3)
    
    retrieved_docs = []
    if vectorstore and user_input:
        print(f"Embedding user query: {user_input}")  # Debug statement
        retrieved_docs = vectorstore.similarity_search(user_input, k=10) if vectorstore else []
        print(f"Dokumen yang ditemukan: {retrieved_docs}")  # Debug statement
        faiss_results = ''.join([doc.page_content[:200] + '...' for doc in retrieved_docs if hasattr(doc, 'page_content')])  # Menampilkan 200 karakter pertama dari tiap hasil
        st.markdown(f"### Hasil Pencarian:")
        st.markdown(f"```{faiss_results}```")
        
        context = "".join([doc.page_content for doc in retrieved_docs if hasattr(doc, 'page_content')])
    else:
        context = ""
    
    # Combine BM25 and vectorstore results
    combined_results = top_n + retrieved_docs
    combined_results = list({doc.page_content: doc for doc in combined_results}.values())  # Remove duplicates based on page_content
    
    # Update context with combined results
    context = "".join([doc.page_content for doc in combined_results if hasattr(doc, 'page_content')])
    
    messages = [
        {"role": "system", "content": "Anda adalah AI Agen Teknologi Informasi (TIK) yang bertugas membantu pengguna di Sekretariat Pengadilan Pajak. Tugas utama Anda adalah membantu pengguna dalam menyelesaikan masalah terkait Teknologi Informasi (IT), memahami panduan-panduan IT yang ada, dan memberikan penjelasan teknis yang akurat berdasarkan informasi yang tersedia di knowledgebase.\n\nAturan:\n\nSumber Jawaban: Anda hanya diperbolehkan menggunakan data yang ada dalam knowledgebase untuk memberikan jawaban. Anda tidak dapat mencari atau menggunakan informasi eksternal selain yang sudah tersedia di dalam knowledgebase.\nMenjawab Pertanyaan: Berikan jawaban yang jelas, profesional, dan langsung sesuai dengan konteks pertanyaan pengguna.\nPenjelasan Teknis: Apabila diberikan pertanyaan teknis terkait IT, Anda memberikan penjelasan yang sesuai dengan panduan yang ada, menggunakan bahasa yang sederhana namun tetap teknis.\nTidak Memberikan Opini Tambahan: Anda tidak boleh memberikan opini pribadi atau pendapat tambahan yang tidak relevan dengan pertanyaan yang diajukan.\nJika Tidak Ada Jawaban di Knowledgebase: Jika Anda tidak dapat menemukan jawaban yang relevan dalam knowledgebase, beri penjelasan dengan sopan, dan beri tahu pengguna bahwa informasi yang diminta tidak tersedia dalam knowledgebase dan arahkan user untuk menghubungi OPTIK di Internal 6523 atau kunjungi Service Desk Set. PP di https://s.id/servicedesksetpp\nApabila konteks pertanyaan user seputar manajemen akun, maka tanya terlebih dahulu akun apa yang dimaksud. apakah akun kemenkeu/bssn/microsoft. Setelah user memberikan konfirmasi baru cek di knowledge base sesuai dengan konteks akunnya.\nApabila konteks pertanyaan user seputar kendala jaringan, maka tanya terlebih dahulu detailnya. konfirmasi jenis jaringannya apakah lan/wifi. apabila wifi maka konfirmasi kembali kemenkeu/kemenkeu public. kemudian konfirmasi perangkat yang digunakan apakah laptop/desktop/smartphone. Setelah user memberikan konfirmasi baru cek di knowledge base sesuai dengan konteks akunnya.\nApabila konteks pertanyaan seputar aplikasi tanpa ada informasi apa konteks aplikasinya, maka tanya terlebih dahulu aplikasi apa yang dimaksud. Setelah user memberikan konfirmasi baru cek di knowledge base sesuai dengan konteks akunnya.\n\ncatatan: semua konteks pertanyaan adalah seputar sistem teknologi informasi baik hardware maupun software di sekretariat pengadilan pajak."},
        {"role": "user", "content": f"Konteks: {context}\nPertanyaan: {user_input}"}
    ]
    try:
        if not check_openai_connection():
            return "Koneksi gagal. Silakan coba lagi nanti."
        response = openai.ChatCompletion.create(
            model=st.secrets["PINECONE_API_KEY"],
            messages=messages
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

# Streamlit UI
st.title("Chatbot Interface")
st.write("Chat with the chatbot like ChatGPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Checkbox to use knowledge base (default checked and disabled)
use_knowledge_base = st.checkbox("Use Knowledge Base", value=True, disabled=True)

# Input form in the center
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
with st.form(key='chat_form'):
    user_input = st.text_input("You:", key="input")
    submit_button = st.form_submit_button(label='Send')
st.markdown("</div>", unsafe_allow_html=True)

if submit_button and user_input:
    # Clear previous messages
    st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = chatbot_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Bot: {message['content']}")
