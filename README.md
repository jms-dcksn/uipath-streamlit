# UiPath LangGraph Agent Example

## Overview

This example demonstrates how to deploy a LangGraph agent that leverages UiPath's agentic assets. The implementation showcases:

- Integration with UiPath AI Trust Layer for managed model connections
- Utilization of UiPath Context Grounding for Retrieval-Augmented Generation (RAG)
- Deployment of a Streamlit-based web interface for agent interaction

## Prerequisites

- Python 3.10 or higher
- UiPath Automation Cloud with AI Trust Layer models enabled and AI Units
- UiPath Index for Context Grounding
- uv package manager (recommended for faster dependency installation)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uipath-streamlit.git
cd uipath-streamlit
```

2. Create and activate a virtual environment using uv:
```bash
uv venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies using uv:
```bash
uv pip install -e .
```

4. Authenticate with your UiPath Automation Cloud account:
```bash
uipath auth
```

5. Run the Streamlit application:
```bash
streamlit run main.py
```

## Usage

1. Access the web interface at `http://localhost:8501`
2. Input your query in the text field
3. The agent will process your request using UiPath's AI Trust Layer and Context Grounding
4. View the results and agent's reasoning process in the interface

## Documentation

### UiPath Resources
- [UiPath SDK Documentation](https://uipath.github.io/uipath-python/)

### LangGraph Resources
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
