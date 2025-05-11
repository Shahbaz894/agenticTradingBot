import os
import sys
import tempfile
from typing import List
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone

from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from exception.exceptions import TradingBotException
from custom_logging.logging import logger


class DataIngestion:
    """
    Handles loading, splitting, and storing documents into Pinecone vector store.
    """

    def __init__(self):
        try:
            logger.info("Initializing DataIngestion...")
            self.model_loader = ModelLoader()  # Load the embedding model
            self._load_env_variables()         # Load required environment variables
            self.config = load_config()        # Load application config
            logger.info("DataIngestion initialized successfully.")
        except Exception as e:
            logger.error("Error during initialization of DataIngestion.")
            raise TradingBotException(e, sys)

    def _load_env_variables(self):
        """
        Loads and validates required environment variables from a .env file.
        """
        try:
            logger.info("Loading environment variables...")
            load_dotenv()

            required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
            missing_vars = [var for var in required_vars if os.getenv(var) is None]

            if missing_vars:
                raise EnvironmentError(f"Missing environment variables: {missing_vars}")

            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            logger.info("Environment variables loaded successfully.")
        except Exception as e:
            logger.error("Error loading environment variables.")
            raise TradingBotException(e, sys)

    def load_documents(self, uploaded_files) -> List[Document]:
        """
        Loads documents from uploaded files and converts them to LangChain Document objects.
        """
        try:
            logger.info("Starting document loading...")
            documents = []

            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
                suffix = file_ext if file_ext in ['.pdf', '.docx'] else ".tmp"

                # Write uploaded file to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name

                # Load content using appropriate loader
                if file_ext == '.pdf':
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                    logger.info(f"Loaded PDF file: {uploaded_file.filename}")
                elif file_ext == '.docx':
                    loader = Docx2txtLoader(temp_path)
                    documents.extend(loader.load())
                    logger.info(f"Loaded DOCX file: {uploaded_file.filename}")
                else:
                    logger.warning(f"Unsupported file type: {uploaded_file.filename}")

            logger.info(f"Loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            logger.error("Error during document loading.")
            raise TradingBotException(e, sys)

    def store_in_vector_db(self, documents: List[Document]):
        """
        Splits documents and stores them in the Pinecone vector database.
        """
        try:
            logger.info("Starting document ingestion to Pinecone...")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            documents = text_splitter.split_documents(documents)
            logger.info(f"Documents split into {len(documents)} chunks.")

            # Initialize Pinecone client
            pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            index_name = self.config["vector_db"]["index_name"]

            # Create index if it doesn't exist
            if index_name not in [i.name for i in pinecone_client.list_indexes()]:
                logger.info(f"Creating new index: {index_name}")
                pinecone_client.create_index(
                    name=index_name,
                    dimension=786,
                    metric='cosine',
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                logger.info(f"Using existing index: {index_name}")

            # Get Pinecone index
            index = pinecone_client.Index(index_name)

            # Load embeddings model
            embeddings = self.model_loader.load_embeddings()

            # Create Pinecone vector store
            vector_store = PineconeVectorStore(index=index, embedding=embeddings)

            # Generate UUIDs for each document
            uuids = [str(uuid4()) for _ in range(len(documents))]

            # Add documents to Pinecone
            vector_store.add_documents(documents=documents, ids=uuids)
            logger.info(f"{len(documents)} documents successfully ingested into Pinecone.")
        except Exception as e:
            logger.error("Error during storing documents in Pinecone.")
            raise TradingBotException(e, sys)
