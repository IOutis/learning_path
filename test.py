from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Any

def searchDuckDuckGo(a: str) -> Any:
    """Searches the web using DuckDuckGo and returns only the top 3 results."""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    # Add output_format="list" here
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")
    results = search.invoke(a)
    return results[:3]

print(searchDuckDuckGo("Price of tomato in telangana today:13th jan 2025"))