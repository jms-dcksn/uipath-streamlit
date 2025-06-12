from venv import logger
import streamlit as st
from typing import List, Dict
from agent import create_graph

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False
    if "retriever_config" not in st.session_state:
        st.session_state.retriever_config = {
            "index_name": "demo-agent-knowledge-hybrid-search",
            "folder_path": "Demos/EFX",
            "tool_name": "ContextTechnicalRequirementsMappings",
            "tool_description": "Use this tool to search for technical requirements mappings of NIST codes",
            "number_of_results": 5
        }

tools_list = str("## Available Tools: \n"
            "* DuckDuckGo Web Search\n * UiPath Context Grounding Search")


def initialize_agent(model_id, system_prompt, retriever_config):
    agent = create_graph(
        model_name=model_id, 
        prompt_text=system_prompt,
        retriever_config=retriever_config
    )
    print("Agent initialized with " + model_id)
    return agent

def stream_response(graph, user_input: str) -> None:
    """Stream the assistant's response"""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()
            # Get response from the graph
            response = graph.invoke({"messages": [{"role": "user", "content": user_input}]})            
            # Extract the final response content
            if response and "messages" in response:
                # Get the last message which should be the AI's response
                final_message = response["messages"][-1]
                if hasattr(final_message, "content"):
                    message_placeholder.markdown(final_message.content)
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": final_message.content})

def main():
    st.title(":mechanical_arm: UiPath-Powered LangGraph Chat Assistant")
    initialize_session_state()
    
    with st.sidebar:
        st.header("Agent Configuration", divider=True)
        system_prompt = st.text_area("System Prompt", value='You are a helpful assistant.')

        # Get Model Id
        model_id = st.sidebar.selectbox("Model", options=["openai:gpt-4o", "uipath:gpt-4o-2024-08-06"])
        
        # Retriever Configuration
        st.subheader("Context Configuration", divider=True)
        retriever_config = st.session_state.retriever_config
        
        retriever_config["index_name"] = st.text_input(
            "Index Name",
            value=retriever_config["index_name"],
            help="Name of the Pinecone index to use"
        )
        
        retriever_config["folder_path"] = st.text_input(
            "Folder Path",
            value=retriever_config["folder_path"],
            help="Path to the folder containing the documents"
        )
        
        retriever_config["tool_name"] = st.text_input(
            "Tool Name",
            value=retriever_config["tool_name"],
            help="Name of the retriever tool"
        )
        
        retriever_config["tool_description"] = st.text_area(
            "Index Description",
            value=retriever_config["tool_description"],
            help="Description of what the tool does"
        )
        
        retriever_config["number_of_results"] = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=retriever_config["number_of_results"],
            help="Number of results to return from the index"
        )
        
        with st.popover(":hammer_and_wrench: Tools"):
            st.markdown(tools_list)
            


    # Set model_id in session state
    if "model_id" not in st.session_state:
        st.session_state["model_id"] = model_id
    elif st.session_state["model_id"] != model_id:
        print("restarting agent with " + model_id)
        st.session_state["model_id"] = model_id
        st.session_state["agent"] = initialize_agent(
            model_id=model_id, 
            system_prompt=system_prompt,
            retriever_config=retriever_config
        )

    # Set system_prompt in session state
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = system_prompt
    elif st.session_state["system_prompt"] != system_prompt:
        st.session_state["system_prompt"] = system_prompt
        st.session_state["agent"] = initialize_agent(
            model_id=model_id, 
            system_prompt=system_prompt,
            retriever_config=retriever_config
        )

    # Check if retriever config has changed
    if st.session_state.retriever_config != retriever_config:
        st.session_state.retriever_config = retriever_config
        st.session_state["agent"] = initialize_agent(
            model_id=model_id, 
            system_prompt=system_prompt,
            retriever_config=retriever_config
        )
        print("Retriever config updated.")

    if "agent" not in st.session_state or st.session_state["agent"] is None:
        logger.info(f"---*--- Creating {model_id} Agent ---*---")
        st.session_state["agent"] = initialize_agent(
            model_id=model_id, 
            system_prompt=system_prompt,
            retriever_config=retriever_config
        )
        print("Agent retrieved.")
    
    agent = st.session_state["agent"]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("What's on your mind?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from the graph using stream_response
        stream_response(agent, user_input)


if __name__ == "__main__":
    main()
