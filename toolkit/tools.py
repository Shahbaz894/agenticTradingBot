import os
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from data_models.models import RagToolSchema
from langchain_pinecone import PineconeVectorStore
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from dotenv import load_dotenv
from pinecone import Pinecone
from custom_logging.logging import logger

# Load environment variables from .env file
load_dotenv()

# Initialize Polygon API wrapper for financial data
api_wrapper = PolygonAPIWrapper()

# Initialize model loader for embeddings
model_loader = ModelLoader()

# Load application configuration from YAML or JSON
config = load_config()

@tool(args_schema=RagToolSchema)
def retriever_tool(question):
    """
    Retrieves relevant documents from Pinecone vector store based on a user's question.
    
    Parameters:
        question (str): The question input used for similarity search.

    Returns:
        List[Document]: List of relevant documents retrieved from Pinecone.
    """
    try:
        logger.info("Starting retriever tool with question: %s", question)

        # Load Pinecone API key from environment and initialize client
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set in the environment.")
        pc = Pinecone(api_key=pinecone_api_key)

        # Set up the Pinecone Vector Store with embedding model
        index_name = config["vector_db"]["index_name"]
        vector_store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=model_loader.load_embeddings()
        )

        # Create a retriever with score threshold and top-k filtering
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config["retriever"]["top_k"],
                "score_threshold": config["retriever"]["score_threshold"]
            }
        )

        # Perform similarity search and return documents
        retriever_result = retriever.invoke(question)
        logger.info("Retriever tool returned %d documents", len(retriever_result))
        return retriever_result

    except Exception as e:
        logger.error("Error in retriever_tool: %s", str(e), exc_info=True)
        return []  # You can also raise a custom exception if preferred

# Tavily tool for web search with deep content fetching
tavilytool = TavilySearchResults(
    max_results=config["tools"]["tavily"]["max_results"],
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)
logger.info("TavilySearchResults tool initialized with max_results=%d", config["tools"]["tavily"]["max_results"])

# Financials tool for retrieving company financial data via Polygon API
financials_tool = PolygonFinancials(api_wrapper=api_wrapper)
logger.info("PolygonFinancials tool initialized using Polygon API")
