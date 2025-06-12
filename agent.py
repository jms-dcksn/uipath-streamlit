from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from uipath_langchain.chat.models import UiPathAzureChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from uipath_langchain.retrievers import ContextGroundingRetriever

def create_retriever_with_tool(
    index_name: str,
    folder_path: str,
    tool_name: str,
    tool_description: str,
    number_of_results: int = 5
) -> Tool:
    """
    Create a ContextGroundingRetriever and wrap it in a retriever tool.
    
    Args:
        index_name (str): Name of the Pinecone index to use
        folder_path (str): Path to the folder containing the documents
        tool_name (str): Name of the retriever tool
        tool_description (str): Description of what the tool does
        number_of_results (int): Number of results to return from the retriever
        
    Returns:
        Tool: A configured retriever tool
    """
    retriever = ContextGroundingRetriever(
        index_name=index_name,
        folder_path=folder_path,
        number_of_results=number_of_results
    )

    return create_retriever_tool(
        retriever,
        name=tool_name,
        description=tool_description
    )

def create_graph(
        model_name: str = "openai:gpt-4o", 
        prompt_text: str = "You are a helpful assistant.", 
        retriever_config: dict = None):
    """
    Create a reactive agent graph with specified model and prompt.
    
    Args:
        model_name (str): Name of the OpenAI model to use
        prompt_text (str): Prompt text for the assistant
        retriever_config (dict): Configuration for the retriever tool
        
    Returns:
        Graph: A configured reactive agent graph
    """
    if retriever_config is None:
        retriever_config = {
            "index_name": "TRMappings",
            "folder_path": "Demos/EFX",
            "tool_name": "Context for Technical RequirementsMappings",
            "tool_description": "Use this tool to search for technical requirements mappings of NIST codes",
            "number_of_results": 5
        }

    # Initialize search tool
    search_tool = DuckDuckGoSearchRun()
    
    # Initialize retriever tool
    retriever_tool = create_retriever_with_tool(
        index_name=retriever_config["index_name"],
        folder_path=retriever_config["folder_path"],
        tool_name=retriever_config["tool_name"],
        tool_description=retriever_config["tool_description"],
        number_of_results=retriever_config["number_of_results"]
    )
    
    # Combine tools
    tools = [search_tool, retriever_tool]
    
    # Initialize model
    try:
        provider, model_name = model_name.split(":")
        print(provider)
        if provider.lower() == "openai":
            model = ChatOpenAI(model=model_name)
        elif provider.lower() == "uipath":
            model = UiPathAzureChatOpenAI(
                model=model_name
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Provider must be either 'openai' or 'uipath'")
    except ValueError:
        raise ValueError("Model name must be in format 'provider:model_name' (e.g. 'openai:gpt-4' or 'uipath:gpt-4')")
    
    uipath_agent = create_react_agent(model, tools=tools, prompt=prompt_text)
    # Build the state graph
    builder = StateGraph(input=MessagesState, output=MessagesState)

    builder.add_node("uipath_agent", uipath_agent)

    builder.add_edge(START, "uipath_agent")

    builder.add_edge("uipath_agent", END)

# Compile the graph
    graph = builder.compile()

    return graph
