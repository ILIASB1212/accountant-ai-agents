from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os
from langchain_core.tools import Tool
load_dotenv()
os.environ["SERPAPI_API_KEY"]=os.getenv("SERPAPI_API_KEY")

# Initialize the wrapper. It will use the SERPAPI_API_KEY environment variable.
search_wrapper = SerpAPIWrapper(params={'engine': 'google'})



search = Tool(
    name="google_search",
    description="Search Google for current information. Use this for questions about recent events, news, or facts.",
    func=search_wrapper.run
)



