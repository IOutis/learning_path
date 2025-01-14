import os
import asyncio
import json
from pydantic import BaseModel
from typing import List, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

class Course_Data(BaseModel):
    module: str
    concepts: List[str]  # List of concepts for a single module

class Course_DataResponse(BaseModel):
    course_data: List[Course_Data]

async def crawl_website_async(website: str) -> Any:
    """
    Asynchronously crawls a website and concepts using an LLM extraction strategy and categorizes them into digestable modules to make a detailed course structure from beginner to advanced
    
    Args:
        website (str): The URL of the website to crawl.
    
    Returns:
        Any: A list of dictionaries containing the extracted modules and relevant concepts.
    """
    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        provider="gemini/gemini-2.0-flash-exp",            # e.g. "ollama/llama2"
        api_token=os.getenv('GEMINI_API_KEY'),
        schema=Course_DataResponse.model_json_schema(),  # Updated schema
        extraction_type="schema",
        instruction="Extract all concepts and categorize them into digestible modules. Each module should include a list of related concepts.",
        chunk_token_threshold=4000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",   # or "html", "fit_markdown"
        extra_args={"temperature": 0.0, "max_tokens": 4000}
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        try:
            # 4. Crawl the provided website
            result = await crawler.arun(
                url=website,  # Use the provided website URL
                config=crawl_config
            )

            if result.success:
                # 5. The extracted content is presumably JSON
                try:
                    # Convert the string output into a Python data structure
                    data = json.loads(result.extracted_content)
                    return data  # Return the extracted data
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse LLM output as JSON: {e}"}
                except Exception as e:
                    return {"error": f"An error occurred while processing the data: {e}"}
            else:
                return {"error": result.error_message}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def crawl_website(website: str) -> Any:
    """
    Synchronous wrapper for the async crawl_website function.
    
    Args:
        website (str): The URL of the website to crawl.
    
    Returns:
        Any: A list of dictionaries containing the extracted learning paths and concepts.
    """
    return asyncio.run(crawl_website_async(website))

# Example usage
if __name__ == "__main__":
    # Test the function
    result = crawl_website("https://www.geeksforgeeks.org/golang-tutorial-learn-go-programming-language/")
    print(result)