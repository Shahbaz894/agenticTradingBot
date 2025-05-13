import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from custom_logging.logging import logger

class ModelLoader:
    """
    A utility class to load embedding models and LLM models.
    Handles environment setup, configuration loading, and model initialization.
    """

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        logger.info("ModelLoader initialized successfully.")

    def _validate_env(self):
        """
        Ensure required environment variables are present.
        """
        required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error("Missing environment variables: %s", missing_vars)
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        logger.debug("All required environment variables are set.")

    def load_embeddings(self):
        """
        Load and return the Google Generative AI embedding model.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            logger.info("Loading embedding model: %s", model_name)
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            logger.info("Embedding model loaded successfully.")
            return embeddings
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", str(e))
            raise

    def load_llm(self):
        """
        Load and return the Groq LLM.
        """
        try:
            model_name = self.config["llm"]["groq"]["model_name"]
            logger.info("Loading Groq LLM: %s", model_name)
            groq_model = ChatGroq(model=model_name, api_key=self.groq_api_key)
            logger.info("Groq LLM initialized successfully.")
            return groq_model
        except Exception as e:
            logger.exception("Failed to load Groq LLM: %s", str(e))
            raise
