from langchain_core.tools import Tool
from langchain_community.document_loaders import WebBaseLoader
from typing import Optional

def web_base_loader(url: str) -> str:
    """
    Load content from a URL and return the page content.
    
    Args:
        url (str): The webpage URL to load
        
    Returns:
        str: The page content or error message
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if docs and len(docs) > 0:
            title=docs[0].metadata.get("title")
            content = docs[0].page_content
            return f"Title: {title}\n\nContent: {content}"
        else:
            return "No content found at the provided URL."
            
    except Exception as e:
        return f"Error loading URL: {str(e)}"

# Create the tool
web_loader_tool = Tool(
    name="rag_web_loader",
    description="Loads and reads content from web pages. Input should be a complete URL starting with http:// or https://",
    func=web_base_loader
)



