import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# LangChain / Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load API Key
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

FAISS_DIR = "faiss_store_gemini"

# Streamlit setup
st.set_page_config(page_title="NewsAnalyzer AI", layout="wide")
st.title("📰 NewsAnalyzer AI")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: URLs & model
with st.sidebar:
    st.header("News Sources & Settings")
    urls = [st.text_input(f"URL {i+1}") for i in range(3)]
    model_choice = st.selectbox("Select Model", ["gemini-2.5-flash", "gemini-2.0", "gemini-1.5-pro"])
    process_btn = st.button("Process URLs")

# --- Process URLs ---
if process_btn:
    valid_urls = [u.strip() for u in urls if u and u.strip()]
    if not valid_urls:
        st.warning("Enter at least one valid URL.")
    else:
        try:
            with st.spinner("Loading webpages..."):
                loader = UnstructuredURLLoader(urls=valid_urls)
                docs = loader.load()
            with st.spinner("Splitting text..."):
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ","],
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                split_docs = splitter.split_documents(docs)
            with st.spinner("Creating embeddings and FAISS vector store..."):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", google_api_key=API_KEY
                )
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local(FAISS_DIR)
            st.success("✅ FAISS vector store saved successfully!")
        except Exception as e:
            st.error(f"Error while processing URLs: {e}")

st.write("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Ask Question", "📚 Sources", "🕓 ChatHistory"])

K = 7  # fixed retriever chunk count

# --- Load vectorstore ---
vectorstore = None
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=API_KEY
    )
    if os.path.exists(FAISS_DIR):
        vectorstore = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
except Exception as e:
    st.sidebar.warning("FAISS vectorstore not found. Process URLs first.")

# --- Tab 1: Ask Question ---
with tab1:
    query = st.text_input("Ask a question about the news articles:")
    answer_btn = st.button("Get Answer")  # ✅ User triggers LLM

    if answer_btn and query and vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": K})
        try:
            docs = retriever.invoke(query)
            if not docs:
                st.info("No relevant information found.")
            else:
                # assemble context
                context_parts = []
                for i, d in enumerate(docs):
                    src = d.metadata.get("source", f"doc_{i+1}")
                    text = d.page_content.strip()
                    snippet = text if len(text) < 2500 else text[:2500] + " ...[truncated]"
                    context_parts.append(f"--- Source {i+1}: {src}\n{snippet}")
                context = "\n\n".join(context_parts)

                prompt = (
                    "You are a helpful research assistant. Use ONLY the provided snippets.\n\n"
                    f"DOCUMENTS:\n{context}\n\n"
                    "INSTRUCTIONS:\n"
                    "1) Answer concisely (3–6 sentences).\n"
                    "2) List sources under 'SOURCES:'.\n\n"
                    f"QUESTION: {query}\n\nProvide the answer now."
                )

                llm = ChatGoogleGenerativeAI(
                    model=model_choice, temperature=0.7, google_api_key=API_KEY
                )
                resp = llm.invoke(prompt)
                answer_text = getattr(resp, "content", str(resp))
                st.subheader("Answer")
                st.write(answer_text)

                # Save chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": query,
                    "answer": answer_text
                })
        except Exception as e:
            st.error(f"Retriever / LLM error: {e}")

# --- Tab 2: Sources ---
with tab2:
    st.subheader("Retrieved Sources")
    if vectorstore and st.session_state.chat_history:
        last_docs = retriever.invoke(st.session_state.chat_history[-1]["question"])
        for i, d in enumerate(last_docs):
            st.markdown(f"**Source {i+1}** — {d.metadata.get('source', 'Unknown')}")
            st.write(d.page_content[:2000])
    else:
        st.info("No sources to display.")

# --- Tab 3: Summary ---
with tab3:
    st.subheader("Chat History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history[::-1]:
            st.markdown(f"**{chat['timestamp']}**")
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.write("---")
    else:
        st.info("No questions asked yet.")
