import streamlit as st
import requests
from exception.exceptions import TradingBotException
import sys
from custom_logging.logging import logger

# Backend FastAPI service endpoint
BASE_URL = "http://localhost:8000"

# Streamlit page configuration
st.set_page_config(
    page_title="📈 Stock Market Agentic Chatbot",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📈 Stock Market Agentic Chatbot")

# Initialize chat history session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Sidebar Section: File Upload
# ---------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    st.markdown("Upload **stock market PDFs or DOCX** to create knowledge base.")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Upload and Ingest"):
        if uploaded_files:
            files = []
            for f in uploaded_files:
                file_data = f.read()
                if not file_data:
                    logger.warning(f"Skipped empty file: {getattr(f, 'name', 'unknown')}")
                    continue
                files.append(("files", (getattr(f, "name", "file.pdf"), file_data, f.type)))

            if files:
                try:
                    with st.spinner("Uploading and processing files..."):
                        logger.info("Uploading %d files to backend.", len(files))
                        response = requests.post(f"{BASE_URL}/upload", files=files)

                    if response.status_code == 200:
                        logger.info("File upload successful.")
                        st.success("✅ Files uploaded and processed successfully!")
                    else:
                        logger.error("Upload failed with status %s: %s", response.status_code, response.text)
                        st.error("❌ Upload failed: " + response.text)
                except Exception as e:
                    logger.exception("Exception during file upload.")
                    raise TradingBotException(e, sys)
            else:
                logger.warning("No valid files found for upload.")
                st.warning("Some files were empty or unreadable.")
        else:
            logger.warning("Upload button clicked without selecting any files.")
            st.warning("Please upload at least one file.")

# ---------------------------
# Main Section: Chat Interface
# ---------------------------
st.header("💬 Chat")

# Display chat history from session state
for chat in st.session_state.messages:
    if chat["role"] == "user":
        st.markdown(f"**🧑 You:** {chat['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {chat['content']}")

# ---------------------------
# Chat Input Form
# ---------------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message", placeholder="e.g. Tell me about NIFTY 50")
    submit_button = st.form_submit_button("Send")

# ---------------------------
# Chatbot Query Handler
# ---------------------------
if submit_button and user_input.strip():
    try:
        logger.info("User asked: %s", user_input)

        # Add user's message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Spinner while processing
        with st.spinner("Bot is thinking..."):
            payload = {"question": user_input}
            response = requests.post(f"{BASE_URL}/query", json=payload)

        if response.status_code == 200:
            answer = response.json().get("answer", "No answer returned.")
            logger.info("Bot answered: %s", answer)
            st.session_state.messages.append({"role": "bot", "content": answer})
            st.rerun()  # Refresh to show new messages
        else:
            logger.error("Bot response failed with status %s: %s", response.status_code, response.text)
            st.error("❌ Bot failed to respond: " + response.text)

    except Exception as e:
        logger.exception("Exception during chatbot interaction.")
        raise TradingBotException(e, sys)
