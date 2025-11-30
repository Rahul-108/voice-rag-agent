import streamlit as st
import os
from dotenv import load_dotenv
import speech_recognition as sr
import edge_tts
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ API Key missing! Please create a .env file with GOOGLE_API_KEY")
    st.stop()

st.set_page_config(page_title="Voice Brain", page_icon="🧠", layout="wide")

st.markdown("""
<style>
        /*Modern Chat UI Styling*/
        .stChatMessage{
            background-color: #262730;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 10px;    
        }
        .stChatMessage p, .stChatMessage div {
            color: #FFFFFF !important;
        }
        .stButton>button{
            width: 100%;
            border-radius: 20px;
            height: 50px;
            font-weight: bold;
            background-color: #FF4B4B;
            color: white;
            border: none;    
        }
        .stButton>button:hover {
            background-color: #FF2B2B;
        }
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


async def text_to_speech(text, output_file="response.mp3"):
    """Converts text to speech using Microsoft Edge's free neural voices."""
    # 'en-US-AvaNeural' is a very natural sounding voice
    communicate = edge_tts.Communicate(text, "en-US-AvaNeural")
    await communicate.save(output_file)

def record_audio():
    """Records audio from the system microphone."""
    try:
        r= sr.Recognizer()
        #Dynamic energy threshold adjustment for noisy environments
        r.dynamic_energy_threshold = True

        with sr.Microphone() as source:
            st.toast("Listening......Speak now!")
            try:
                # audio = r.listen(source, timeout=40, phrase_time_limit=400)
                audio = r.listen(source, timeout=40)
                st.toast("Processing audio...")
                text = r.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                st.warning("Time out! No speech detected.")
                return None
            except sr.UnknownValueError:
                st.warning("Could not understand audio.")
                return None
            except Exception as e:
                st.error(f"Microphone error: {e}")
                return None
    except AttributeError:
        st.error("⚠️ Microphone not available. Voice input only works when running locally with PyAudio installed.")
        return None

@st.cache_resource
def get_embeddings_model():
    """Cached embeddings model to avoid re-downloading."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store(uploaded_file):
    """Processes uploaded PDF and creates a FAISS vector store."""
    # Lazy imports - only load when needed
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Use cached embeddings model
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    os.unlink(tmp_file_path)
    return vector_store


with st.sidebar:
    st.title("Knowledge Base")
    st.markdown("Upload a PDF to give your agent a 'brain'.")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner("Reading document....."):
            st.session_state.vector_store = get_vector_store(uploaded_file)
        st.success("Brain Loaded! Ready to chat.")

    if st.button("Clear Chat Memory"):
        st.session_state.chat_history = []
        st.rerun()


st.title("AI Voice Agent")
st.caption("I can read your documents and answer questions via voice.")

container = st.container()
with container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("Tap to Speak"):
        if st.session_state.vector_store is None:
            st.error("Please upload a PDF in the sidebar first!")
        else:
            user_input = record_audio()

        if user_input:
            st.session_state.chat_history.append({"role":"user", "content": user_input})

            retriever =st.session_state.vector_store.as_retriever()
            relevant_docs = retriever.invoke(user_input)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

            system_prompt = (
                "You are a helpful voice assistant. Answer the question concisely based ONLY on the provided context."
                "If answer is not in the context, say 'I don't see that in the document'. "
                "Keep your answer under 3 sentences for better voice experience."
            )

            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {user_input}"
            response = llm.invoke(full_prompt)
            ai_text = response.content

            st.session_state.chat_history.append({"role":"assistant", "content": ai_text})
            st.rerun()


if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
    last_msg = st.session_state.chat_history[-1]["content"]

    if "last_spoken" not in st.session_state or st.session_state.last_spoken != len(st.session_state.chat_history):
        asyncio.run(text_to_speech(last_msg))
        st.audio("response.mp3", format="audio/mp3", autoplay=True)
        st.session_state.last_spoken = len(st.session_state.chat_history)

    