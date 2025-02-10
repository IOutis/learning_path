import json
import re
from groq import APIError
from langchain.agents import create_json_chat_agent
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

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# 
# def gemini_llm(query: str) -> Any:
#     """
#     Generates detailed and structured content for a given query using the Gemini LLM.
    
#     Args:
#         query (str): The query or topic for which content needs to be generated.
    
#     Returns:
#         str: A detailed and structured response to the query, formatted in markdown.
#     """
#     model = genai.GenerativeModel(
#         model_name="gemini-2.0-flash-exp",
#         system_instruction="""You are a professional course creator and educational content developer. Your task is to:
# 1. Provide an in-depth, multi-dimensional explanation for each topic that:
# - Delivers a precise, detailed theoretical explanation
# - Articulates the fundamental "why" behind the concept
# - Deconstructs complex ideas into digestible, sequential explanations
# - Embeds real-world context and practical applications
# - Reveals underlying mechanisms and core principles
# - Illuminates the broader significance and potential impact
# - Transcends surface-level descriptions
# - Incorporates technical depth
# - Connects theoretical knowledge with practical implementation
# - Uses clear, accessible language
# - Provides rich contextual understanding
# 2. Organize the content into modules and submodules.
# 3. Include examples, references, and practical applications.
# 4. STrict instruction : Format the output with the following structure (an object but dont include backticks in the beginning or in the end because its causing errors here."):

# {
#   "course": "Course Title",
#   "categories": [
#     {
#       "name": "Category Name",
#       "subcategories": [
#         {
#           "name": "Subcategory Name",
#           "content": "Detailed content for this subcategory."
#           "links":"URL Links to relevant resources including websites, journals, Youtube Videos. STRICT RULE : Before using the links first check to make sure that they are valid or not. Becuase most of the links you gave earlier were either removed or were invalid. 
#           "references": [
#                {
#                   "title": "Reference Title",
#                   "author": "Reference Author",
#                   "year": "Reference Year",
#                   "url": "Reference URL",
#                   "description": "Reference Description"
#                 }
#             ] 
            
#         }
#       ]
#     }
#   ]
# }
#     STRICT RULE FOR LINKS: 
# - URLs must be in a SINGLE STRING separated by commas
# - NO individual quotes around URLs
# - Example: "links":"url1, url2, url3"
#         - When reciting copy-righted materials please just understand the content and then give the interpreted content of the data. Paraphrase the key points in your own words, ensuring that the original meaning is preserved. Include proper attribution to the original source.
#         """
#     )

#     chat = model.start_chat(history=[])
#     # gemini_model = GeminiWrapper(model)
#     # gemini_agent = create_json_chat_agent(llm=gemini_model,)
    
#     # response = gemini_agent.run(query)
#     # print(response,type(response),sep="\n\n")
    
    
    
    
    
    
#     response = chat.send_message("Provide detailed explanation for the information given here such that it resembles a course with proper content that students can read and be knowledgable : " + query, stream=True)
#     response_text = ""
#     for chunk in response:
#         response_text += chunk.text
        
#     cleaned_text = response_text.strip()
#     # print("Cleaned Response : ",cleaned_text)
    
#     # Remove any markdown formatting if present
#     if "```" in cleaned_text:
#         # Remove everything before the first {
#         cleaned_text = cleaned_text[cleaned_text.find("{"):cleaned_text.rfind("}") + 1]
    
#     try:
#         # Validate JSON
#         course_data = json.loads(cleaned_text)
        
#         # Optional: Print the structure (can be removed if not needed)
#         print("\nCourse Structure:")
#         print("Course Title:", course_data["course"])
#         for category in course_data["categories"]:
#             print(f"\nModule: {category['name']}")
#             for subcategory in category["subcategories"]:
#                 print(f"  Topic: {subcategory['name']}")
                
#         return course_data
        
#     except json.JSONDecodeError as e:
#         print("Attempting automatic repair...")
#         # Remove stray quotes in links
#         repaired_text = re.sub(r'(,"links":") (.*?)(",?)', lambda m: f',"links":"{m.group(2)}"{m.group(3)}', cleaned_text)
#         try:
#             return json.loads(repaired_text)
#         except:
#             print(cleaned_text)
#             raise ValueError("Failed to repair JSON automatically")
    
import json
import re
import logging
import google.generativeai as genai

def repair_json(text):
    """
    Comprehensively repair potentially malformed JSON
    
    Args:
        text (str): Raw text response to be parsed
    
    Returns:
        dict: Parsed and cleaned JSON object
    """
    try:
        # Remove any leading/trailing whitespace and potential markdown markers
        text = text.strip()
        
        # Remove markdown code block markers if present
        if text.startswith("```"):
            text = text[text.find("{"):text.rfind("}")+1]
        
        # Basic JSON repair strategies
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas in objects
        text = re.sub(r',\s*]', ']', text)  # Remove trailing commas in arrays
        
        # Handle problematic links formatting
        text = re.sub(r'"links"\s*:\s*"([^"]+)"', r'"links": ["\1"]', text)
        
        # Ensure proper comma placement between nested structures
        text = re.sub(r'(?<=})\s*(?={)', ',', text)
        
        # Parse and return the JSON
        return json.loads(text)
    
    except json.JSONDecodeError as e:
        logging.error(f"JSON Repair Error: {e}")
        logging.error(f"Problematic Text: {text}")
        raise ValueError(f"Could not parse JSON: {e}")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def gemini_llm(query: str) -> dict:
    """
    Generates detailed and structured content for a given query using Gemini LLM.
    
    Args:
        query (str): The query or topic for content generation
    
    Returns:
        dict: Structured course content
    """
    try:
        # Configure Gemini model with system instructions
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            system_instruction="""I want to learn the topic mentioned above from the world's best professional-YOU. You are the ultimate expert, the top authority in this field and best tutor anyone could ever learn from. No one can match your knowledge and expertise. Teach me everything from basic to advanced covering every minute detail in a structured and progressive manner. Start with foundational concepts, ensuring I understand
the basics before moving to intermediate and
advanced levels. Include practical applications,
real-world examples. expert insights, and common
mistakes to avoid. Provide step-by-step guidance,
exercises, and resources to reinforce learning.
Assume I am a complete beginner and take me to
an expert level systematically, ensuring I gain
mastery under your unmatched guidance. Provide in-depth, multi-dimensional explanations with:
1. Precise theoretical explanations
2. Clear fundamental principles
3. Real-world context and applications
4. Technical depth
5. Structured JSON output

STRICT OUTPUT FORMAT:
{
  "course": "Course Title",
  "categories": [
    {
      "name": "Category Name",
      "subcategories": [
        {
          "name": "Subcategory Name",
          "content": "Detailed explanation",
          "links": ["url1", "url2"],
          "references": [
            {
              "title": "Reference Title",
              "author": "Author",
              "year": "Year",
              "url": "URL",
              "description": "Reference Description"
            }
          ]
        }
      ]
    }
  ]
}
"""
        )

        # Start chat and send message
        chat = model.start_chat(history=[])
        response = chat.send_message(
            f"Provide detailed explanation for the information as a structured course: {query}", 
            stream=False
        )
        
        # Collect full response text
        response_text = "".join(chunk.text for chunk in response)
        
        # Repair and parse JSON
        course_data = repair_json(response_text)
        
        # Optional: Print course structure for debugging
        # print("\nCourse Structure:")
        # print("Course Title:", course_data.get("course", "Untitled"))
        # for category in course_data.get("categories", []):
        #     print(f"\nModule: {category['name']}")
        #     for subcategory in category.get("subcategories", []):
        #         print(f"  Topic: {subcategory['name']}")
        print(course_data)
        
        return course_data
    
    except Exception as e:
        logging.error(f"Error in Gemini LLM processing: {e}")
        raise
    
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

For Course generation or learning path generation specifically:
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
Generate me a detailed course on Blockchain from beginner to advanced. Divide all the relevant concepts difficulty-wise and further categorize them into digestible modules with references for each topic and relevant YouTube videos so that I get a bit more detailed explanation on the respective topic.
"""
else:
    prompt_text = get_multiline_input()

# Execute the agent
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIError)  # Retry only if APIError is raised
)
def execute_groq_agent(prompt_text):
    answer= agent_executor.invoke({"input": prompt_text})
    course_dictionary = gemini_llm(answer['output'])
    return course_dictionary

# try:
answer = execute_groq_agent(prompt_text)
# except APIError as e:
#     print(f"Failed to execute agent after retries: {e}")

# print("Tools in the tool belt : ",agent_executor.tools)


