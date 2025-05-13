from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loader import ModelLoader
from toolkit.tools import *
from custom_logging.logging import logger

# Define the state for the graph, containing message history
class State(TypedDict):
    messages: Annotated[list, add_messages]

class GraphBuilder:
    def __init__(self):
        # Initialize model loader and load base LLM
        logger.info("Initializing GraphBuilder...")
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        logger.info("LLM loaded successfully.")

        # Define tools for the agent to use
        self.tools = [retriever_tool, financials_tool, tavilytool]
        logger.info("Tools loaded: %s", [tool.name for tool in self.tools])

        # Bind tools with LLM for reasoning + tool usage
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        logger.info("LLM successfully bound with tools.")

        self.graph = None

    def _chatbot_node(self, state: State):
        """
        Chatbot node that processes the state and generates AI response.
        """
        logger.debug("Processing chatbot node with state: %s", state)
        try:
            # Use the LLM with tools to generate a response
            result = self.llm_with_tools.invoke(state["messages"])
            logger.debug("Chatbot node response generated.")
            return {"messages": [result]}
        except Exception as e:
            logger.exception("Error in chatbot node.")
            raise

    def build(self):
        """
        Builds the LangGraph with conditional tool execution.
        """
        logger.info("Building graph...")
        graph_builder = StateGraph(State)

        # Add main chatbot node
        graph_builder.add_node("chatbot", self._chatbot_node)
        logger.info("Chatbot node added to graph.")

        # Add tool-handling node
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        logger.info("Tool node added to graph.")

        # Add conditional edge to call tools if needed
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        logger.info("Conditional edges from chatbot to tools added.")

        # Return flow from tools to chatbot
        graph_builder.add_edge("tools", "chatbot")

        # Define graph start point
        graph_builder.add_edge(START, "chatbot")

        # Compile graph
        self.graph = graph_builder.compile()
        logger.info("Graph successfully compiled.")

    def get_graph(self):
        """
        Returns the compiled graph if available.
        """
        if self.graph is None:
            logger.error("Attempted to access graph before building.")
            raise ValueError("Graph not built. Call build() first.")
        logger.info("Graph retrieved successfully.")
        return self.graph
