from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Any
from langchain_core.tools import Tool
from functools import lru_cache
# from litellm import retry
from playwright.sync_api import sync_playwright
import google.generativeai as genai
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from crawl_test import crawl_website

# Load environment variables
load_dotenv()

# Initialize the Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    max_tokens=None,
    timeout=None,
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def gemini_llm(query: str) -> str:
    """
    Generates detailed and structured content for a given query using the Gemini LLM.
    
    Args:
        query (str): The query or topic for which content needs to be generated.
    
    Returns:
        str: A detailed and structured response to the query, formatted in markdown.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction="""You are a highly detailed and thorough research assistant. When providing information:
        - Always verify information across multiple sources
        - Provide structured, comprehensive answers
        - Include specific examples and code snippets when relevant
        - Cross-reference all information
        - Format output consistently using proper markdown
        - Always cite sources inline as well as in references
        - Be concise and clear in your language
        - When reciting copy-righted materials please just understand the content and then give the interpreted content of the data. Paraphrase the key points in your own words, ensuring that the original meaning is preserved. Include proper attribution to the original source.
        """
    )

    chat = model.start_chat(history=[])
    response = chat.send_message("Provide detailed content for this: " + query, stream=True)
    for chunk in response:
        print(chunk.text, end="")

    print(chat.history)
    import time
    time.sleep(2)  # Wait for 2 seconds before exiting

def searchDuckDuckGo(query: str) -> Any:
    """
    Searches the web using DuckDuckGo and returns a list of relevant search results.
    
    Args:
        query (str): The search query to look up on DuckDuckGo.
    
    Returns:
        Any: A list of search results, including titles, URLs, and snippets.
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    return search.invoke(query)

@lru_cache(maxsize=100)
def scrape_url(website: str) -> Any:
    """
    Scrapes the content of a webpage and returns the raw HTML data.
    
    Args:
        website (str): The URL of the webpage to scrape.
    
    Returns:
        Any: The raw HTML content of the webpage.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(website)
            print(f"Scraping {website}...")
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"Error scraping {website}: {e}")
        return None

# Define the tools using the Tool class
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a thorough research assistant and an expert professional course creator and educational content developer that MUST follow this exact workflow for EVERY query:

1. REQUIRED FIRST STEP: Use the DuckDuckGo Search tool to find at least 3-5 relevant URLs about the topic.

2. REQUIRED SECOND STEP: For each useful URL found, use the Crawl Website tool to extract detailed information. Analyze at least 2-3 different sources.

3. REQUIRED FINAL STEP: Use the Gemini LLM tool to synthesize all the gathered information into a coherent, structured response.

IMPORTANT RULES:
- You MUST use all three tools in the above order for every query
- Never skip steps or try to answer without using the tools
- If the first search doesn't yield good results, try different search terms
- Always provide reasoning for your tool selection
- Track which URLs you've already crawled to avoid duplicates
- Before giving your final answer, verify you've used all three tools

For learning paths specifically:
- Use DuckDuckGo to find curriculum examples from multiple institutions
- Crawl each curriculum page to extract detailed module structures
- Use Gemini to organize and standardize the content into a clear learning path

Your output should always include:
- Sources used (URLs crawled)
- Tools used in order
- Synthesized final response

Remember: Incomplete tool usage will result in incomplete information. Always complete all three steps."""
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Modify tool descriptions to encourage sequential usage
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=searchDuckDuckGo,
        description="REQUIRED FIRST STEP: Search for 3-5 relevant URLs about the topic. Always use this tool first to find source material."
    ),
    Tool(
        name="Crawl Website",
        func=crawl_website,
        description="REQUIRED SECOND STEP: Extract detailed content from URLs found by DuckDuckGo. Use this on at least 2-3 different sources."
    ),
    Tool(
        name="Gemini LLM",
        func=gemini_llm,
        description="REQUIRED FINAL STEP: After gathering information with the other tools, use this to synthesize all findings into a structured response."
    ),
]


# Create the agent and executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=200,return_intermediate_steps=True)

def get_multiline_input():
    """
    Collects multiline input from the user.
    
    Returns:
        str: The user's input as a single string.
    """
    print("Enter your prompt (press Ctrl+D or Ctrl+Z on Windows when done):")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)

# For testing, you can either use the function or hardcode the prompt
USE_HARDCODED_PROMPT = True  # Set to True to use hardcoded prompt

if USE_HARDCODED_PROMPT:
    # prompt_text = """
    # Generate me a well structured and detailed from scratch to advanced learning path on AI in Blockchain : module wise and concepts in each module, with references and links, modules should be well structured and the concepts in each module should cover all the topics that may come in that module . Make sure the learning path you generate is just a overview or like a syllabus structure
    # """
    # prompt_text="""give me detailed structure of learning path of Cloud Computing. Scrape the relevant websites to get the content , understand them, categorize all the concepts in digestable modules which are ordered from beginner level to advanced level."""
    prompt_text = """
Find information about React Native for mobile development, summarize it, and provide a structured course. Use the DuckDuckGo Search tool to find relevant articles, the Crawl Website tool to extract content, and the Gemini LLM tool to summarize and structure the information.
"""
else:
    prompt_text = get_multiline_input()

# Execute the agent
answer = agent_executor.invoke({"input": prompt_text})

# Print the tools used by the agent
print("\nTools Used by the Agent:")
for step in answer["intermediate_steps"]:
    tool_name = step[0].tool  # The tool used
    tool_input = step[0].tool_input  # The input passed to the tool
    tool_output = step[1]  # The output from the tool
    print(f"- Tool: {tool_name}")
    print(f"  Input: {tool_input}")
    print(f"  Output: {tool_output}\n")

print("\nResponse:")
gemini_llm(answer['output'])

print("Tools in the tool belt : ",agent_executor.tools)