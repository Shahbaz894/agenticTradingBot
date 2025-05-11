import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config_loader import load_config
from custom_logging.logging import logger
import sys
from exception.exceptions import TradingBotException
class ModelLoader():
    """
    A utility class to load embedding models and LLM models.
    """
    
    def __init__(self):
        logger.info('initialized ModelLoader')
        load_dotenv()
        self._validate_env()
        self.config=load_config()
        
    def _validate_env(self):
        """
        Validate necessary environment variables.
        """
        try:
            logger.info('validating enviroment variables')
            required_var=['GOOGLE_API_KEY']
            missing_var=[var for var in required_var if not os.getenv(var) is None]
            if missing_var:
                raise EnvironmentError(f"missing enviroment variables")
            
        except Exception as e:
            logger.error('Error validating enviroment variables')
            raise TradingBotException(e,sys)
        
    def load_embedding_model(self):
        logger.info('loading embbeding model')
        try:
            model_name=self.config['embedding_model']['model_name']
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            logger.error('Error loading embedding model')
            raise TradingBotException(e,sys)
        
    def load_llm(self):
        """
        Load and return the LLM model.
        """
        try:
            logger.info('loading llm model')
            model_name=self.config["llm"]["google"]["model_name"]
            gemini_model=ChatGoogleGenerativeAI(model=model_name)
            logger.info('llm model loaded successfully')
            return gemini_model
        except Exception as e:
            logger.error('Error loading LLM model')
            raise TradingBotException(e,sys)
        
        
        