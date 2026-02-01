import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the System Prompt
system_prompt = (
    "You are the Space42 Orbital Twin. Use the following pieces of retrieved context "
    "to answer the question. If you don't know the answer, say you don't know."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 2. Create the "Stuff" chain (combines docs into the prompt)
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

# 3. Create the final Retrieval Chain
rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

# To run it later in your code:
# response = rag_chain.invoke({"input": prompt})
# answer = response["answer"]

# --- INITIALIZATION & BRANDING ---
st.set_page_config(page_title="Space42 Orbital Twin", layout="wide")

# Custom CSS for a "Space-Age" feel
st.markdown("""
    <style>
    .stApp { background-color: #010409; color: #E6EDF3; }
    .stChatFloatingInputContainer { background-color: #0D1117; }
    .source-box { background-color: #161B22; border-left: 5px solid #1F6FEB; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND BRAIN ---
@st.cache_resource
def build_orbital_brain(pdf_path, _api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key
    
    # 1. Ingestion: Loading the Space42 Ground Truth
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Chunking: Recursive splitting for semantic context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    # 3. Vector Space: Storing data in a searchable 'Orbit'
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 4. Agentic Reasoning: Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    # 5. The Chain: Integration of retrieval and response
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# --- UI LAYOUT ---
st.title("🚀 Space42 Orbital Twin")
st.write("Mission Control: Managing the future of AI-powered SpaceTech.")

# Sidebar for Config
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_pdf = st.file_uploader("Upload Space42 Mission Brief", type="pdf")
    
    if st.button("Initialize Orbital Twin") and uploaded_pdf and user_api_key:
        with st.spinner("Calibrating Reasoning Engine..."):
            # Save temp file
            with open("current_brief.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.session_state.agent = build_orbital_brain("current_brief.pdf", user_api_key)
            st.success("System Online.")

# Main Interface: Chat vs Explainability HUD
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 👨‍🚀 Mission Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Query the Navigator..."):
        if "agent" not in st.session_state:
            st.error("Please initialize the system in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Processing query through Reasoning Engine
            with st.spinner("Analyzing Mission Data..."):
                response = st.session_state.agent({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.session_state.last_sources = sources # Store for the HUD

            st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.markdown("### 🛰️ Explainability HUD")
    if "last_sources" in st.session_state:
        for idx, doc in enumerate(st.session_state.last_sources):
            st.markdown(f"**Source {idx+1} (Page {doc.metadata['page']+1})**")
            st.markdown(f"<div class='source-box'>{doc.page_content}</div>", unsafe_allow_html=True)
    else:
        st.info("Source data will appear here after your first query.")
