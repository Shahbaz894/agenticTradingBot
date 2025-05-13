import os
import tempfile
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from pinecone import ServerlessSpec, Pinecone
from custom_logging.logging import logger
from uuid import uuid4
import sys
from exception.exceptions import TradingBotException

class DataIngestion:
    """
    Class to handle document loading, transformation, and ingestion into the Pinecone vector store.
    """

    def __init__(self):
        try:
            logger.info("Initializing DataIngestion pipeline...")
            self.model_loader = ModelLoader()
            self._load_env_variables()
            self.config = load_config()
            logger.info("DataIngestion pipeline initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize DataIngestion pipeline.")
            raise TradingBotException(e, sys)

    def _load_env_variables(self):
        """
        Load and validate environment variables required for ingestion.
        """
        try:
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
        Load and parse uploaded files into LangChain Document objects.
        """
        try:
            documents = []
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
                suffix = file_ext if file_ext in [".pdf", ".docx"] else ".tmp"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.file.read())
                    temp_path = temp_file.name

                if file_ext == ".pdf":
                    logger.info(f"Loading PDF file: {uploaded_file.filename}")
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                elif file_ext == ".docx":
                    logger.info(f"Loading DOCX file: {uploaded_file.filename}")
                    loader = Docx2txtLoader(temp_path)
                    documents.extend(loader.load())
                else:
                    logger.warning(f"Unsupported file type skipped: {uploaded_file.filename}")
            logger.info(f"Total documents loaded: {len(documents)}")
            return documents
        except Exception as e:
            logger.error("Failed to load documents.")
            raise TradingBotException(e, sys)

    def store_in_vector_db(self, documents: List[Document]):
        """
        Split documents and store embeddings in Pinecone vector store.
        """
        try:
            logger.info("Splitting documents for ingestion...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            documents = text_splitter.split_documents(documents)
            logger.info(f"Total document chunks created: {len(documents)}")

            logger.info("Connecting to Pinecone...")
            pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            index_name = self.config["vector_db"]["index_name"]

            # Create index if not exists
            if index_name not in [i.name for i in pinecone_client.list_indexes()]:
                logger.info(f"Creating new Pinecone index: {index_name}")
                pinecone_client.create_index(
                    name=index_name,
                    dimension=768,  # Adjust based on embedding model
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                logger.info(f"Pinecone index '{index_name}' already exists.")

            index = pinecone_client.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=self.model_loader.load_embeddings())

            uuids = [str(uuid4()) for _ in range(len(documents))]
            logger.info("Adding documents to Pinecone vector store...")
            vector_store.add_documents(documents=documents, ids=uuids)
            logger.info("Documents successfully ingested into Pinecone.")
        except Exception as e:
            logger.error("Failed to store documents in vector database.")
            raise TradingBotException(e, sys)

    def run_pipeline(self, uploaded_files):
        """
        Execute the full ingestion pipeline: load, split, and store.
        """
        try:
            logger.info("Running data ingestion pipeline...")
            documents = self.load_documents(uploaded_files)
            if not documents:
                logger.warning("No valid documents found for ingestion.")
                return
            self.store_in_vector_db(documents)
            logger.info("Data ingestion pipeline completed successfully.")
        except Exception as e:
            logger.error("Data ingestion pipeline failed.")
            raise TradingBotException(e, sys)

if __name__ == '__main__':
    pass
