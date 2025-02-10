import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from typing import Any
from crawl_test import crawl_website
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_AGENT_KEY")
genai.configure(api_key=api_key)

# Custom Gemini LLM wrapper for LangChain
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class GeminiChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        # Convert LangChain messages to Gemini API format
        gemini_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                gemini_messages.append({
                    'role': 'user',
                    'parts': [{'text': msg.content}]
                })
            elif isinstance(msg, SystemMessage):
                gemini_messages.append({
                    'role': 'user',
                    'parts': [{'text': f"System instructions: {msg.content}"}]
                })
            else:  # AI/Assistant messages
                gemini_messages.append({
                    'role': 'model',
                    'parts': [{'text': msg.content}]
                })

        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(gemini_messages)
        
        # Create a full ChatResult with AIMessage
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response.text),
                    text=response.text
                )
            ]
        )

    def bind_tools(self, tools):
        return self

    def _llm_type(self):
        return "gemini"

# Initialize the Gemini LLM
llm = GeminiChatModel()

# Define the tools
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
        system_instruction="""You are a professional course creator and educational content developer. I want to learn the topic mentioned above from the world's best professional-YOU. You are the ultimate expert, the top authority in this field and best tutor anyone could ever learn from. No one can match your knowledge and expertise. Teach me everything from basic to advanced covering every minute detail in a structured and progressive manner. Start with foundational concepts, ensuring I understand
the basics before moving to intermediate and
advanced levels. Include practical applications,
real-world examples. expert insights, and common
mistakes to avoid. Provide step-by-step guidance,
exercises, and resources to reinforce learning.
Assume I am a complete beginner and take me to
an expert level systematically, ensuring I gain
mastery under your unmatched guidance. Your task is to:
1. Provide an in-depth, multi-dimensional explanation for each topic that:
- Delivers a precise, detailed theoretical explanation
- Articulates the fundamental "why" behind the concept
- Deconstructs complex ideas into digestible, sequential explanations
- Embeds real-world context and practical applications
- Reveals underlying mechanisms and core principles
- Illuminates the broader significance and potential impact
- Transcends surface-level descriptions
- Incorporates technical depth
- Connects theoretical knowledge with practical implementation
- Uses clear, accessible language
- Provides rich contextual understanding
- Donot just include sentences like "this section focuses on...." Instead explain the topic in detail please
2. Organize the content into modules and submodules.
3. Include examples, references, and practical applications.
4. STrict instruction : Format the output with the following structure (an object but dont include backticks in the beginning or in the end because its causing errors here."):

{
  "course": "Course Title",
  "categories": [
    {
      "name": "Category Name",
      "subcategories": [
        {
          "name": "Subcategory Name",
          "content": "Detailed content for this subcategory."
          "links":"URL Links to relevant resources including websites, journals, Youtube Videos. STRICT RULE : Before using the links first check to make sure that they are valid or not. Becuase most of the links you gave earlier were either removed or were invalid"
        }
      ]
    }
  ]
}
        - When reciting copy-righted materials please just understand the content and then give the interpreted content of the data. Paraphrase the key points in your own words, ensuring that the original meaning is preserved. Include proper attribution to the original source.
        """
    )

    chat = model.start_chat(history=[])
    # gemini_model = GeminiWrapper(model)
    # gemini_agent = create_json_chat_agent(llm=gemini_model,)
    
    # response = gemini_agent.run(query)
    # print(response,type(response),sep="\n\n")
    
    
    
    
    
    
    response = chat.send_message("Provide detailed explanation for the information given here such that it resembles a course with proper content that students can read and be knowledgable : " + query, stream=True)
    response_text = ""
    for chunk in response:
        response_text += chunk.text
        
    cleaned_text = response_text.strip()
    print("Cleaned Response : ",cleaned_text)
    
    # Remove any markdown formatting if present
    if "```" in cleaned_text:
        # Remove everything before the first {
        cleaned_text = cleaned_text[cleaned_text.find("{"):cleaned_text.rfind("}") + 1]
    
    try:
        # Validate JSON
        course_data = json.loads(cleaned_text)
        
        # Optional: Print the structure (can be removed if not needed)
        print("\nCourse Structure:")
        print("Course Title:", course_data["course"])
        for category in course_data["categories"]:
            print(f"\nModule: {category['name']}")
            for subcategory in category["subcategories"]:
                print(f"  Topic: {subcategory['name']}")
                print(f"        Topic: {subcategory['content']}")
                
        return course_data
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Response Text that failed to parse:", cleaned_text)
        raise

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
    )
]

# Create a simple agent
agent = create_tool_calling_agent(
    llm=llm,  # Use the custom Gemini LLM
    tools=tools,
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", """You are a thorough research assistant and an expert professional course creator and educational content developer that MUST follow this exact workflow for EVERY query:

1. REQUIRED FIRST STEP: Use the DuckDuckGo Search tool to find at least 3-5 relevant URLs about the topic. Also search for relevant youtube videos for user's better understanding. URL Links to relevant resources including websites, journals, Youtube Videos. STRICT RULE : Before using the links first check to make sure that they are valid or not. Becuase most of the links you gave earlier were either removed or were invalid.

2. REQUIRED SECOND STEP: For each useful URL found, use the Crawl Website tool to extract detailed information. Analyze at least 2-3 different sources. URL Links to relevant resources including websites, journals, Youtube Videos. STRICT RULE : Before using the links first check to make sure that they are valid or not. Becuase most of the links you gave earlier were either removed or were invalid.

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

Remember: Incomplete tool usage will result in incomplete information. Always complete all three steps."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),  # Required for tool calling
        ]
    ),
)

# Initialize the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Retry logic for agent execution
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(json.JSONDecodeError)
)
def execute_agent(prompt_text):
    print(prompt_text)
    response = agent_executor.invoke({"input": prompt_text})
    course = gemini_llm(response['output'])
    return course
    # Validate the output
    # json.loads(response["output"])  # Raises JSONDecodeError if invalid
    # return response

# Example usage
# try:
prompt_text = """
Generate me a detailed course on Blockchain from beginner to advanced. Divide all the relevant concepts difficulty-wise and further categorize them into digestible modules with references for each topic and relevant YouTube videos so that I get a bit more detailed explanation on the respective topic.
"""
execute_agent(prompt_text)


    # Save the output to a JSON file
    # output_file = "course_output.json"
    # with open(output_file, "w") as f:
    #     json.dump(json.loads(response["output"]), f, indent=2)
    # print(f"Course output saved to {output_file}")
# except json.JSONDecodeError as e:
#     print(f"Failed to generate valid JSON output after retries: {e}")
# except Exception as e:
#     print(f"An error occurred: {e}")