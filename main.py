from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.responses import JSONResponse
from data_ingestion.ingestion_pipeline import DataIngestion  # Handles data ingestion and storage
from agent.workflow import GraphBuilder  # Manages the LLM workflow or agent stream
from data_models.models import *  # Includes data schemas like QuestionRequest
from custom_logging.logging import logger  # Custom logger for application-level logging

app = FastAPI()

# Enable CORS to allow cross-origin requests (use specific origins in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/upload')
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload and ingest files (e.g., PDFs, docs) into the vector store.
    """
    try:
        logger.info("Received %d files for ingestion.", len(files))

        ingestion = DataIngestion()
        ingestion.run_pipeline(files)

        logger.info("Files successfully processed and stored.")
        return {"message": "Files successfully processed and stored."}
    
    except Exception as e:
        logger.error("Error during file upload: %s", str(e), exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/query")
async def query_chatbot(request: QuestionRequest):
    """
    Endpoint to query the chatbot/LLM with a natural language question.
    """
    try:
        logger.info("Received query: %s", request.question)

        # Initialize graph workflow and build computation graph
        graph_service = GraphBuilder()
        graph_service.build()
        graph = graph_service.get_graph()

        # Format message for graph input (may vary based on your graph implementation)
        messages = {"messages": [request.question]}

        logger.info("Invoking graph with message: %s", messages)
        result = graph.invoke(messages)

        # Parse result depending on graph's return type
        if isinstance(result, dict) and "messages" in result:
            final_output = result["messages"][-1].content  # Extract final answer
        else:
            final_output = str(result)

        logger.info("Query response: %s", final_output)
        return {"answer": final_output}

    except Exception as e:
        logger.error("Error during chatbot query: %s", str(e), exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
