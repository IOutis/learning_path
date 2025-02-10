from typing import Any
from multi_agent import execute_groq_agent
from Gemini_agent import execute_agent

class ConceptDebate:
    def __init__(self, max_rounds=3):
        self.max_rounds = max_rounds
        self.groq_history = []
        self.gemini_history = []
    
    def create_prompt(self,question: str) -> str:
        """Creates a prompt with explicit format instructions."""
        return f"""
        {question}

        **Required Format:**
        {{
            "course": "Course Title",
            "categories": [
                {{
                    "name": "Category Name",
                    "subcategories": [
                        {{
                            "name": "Subcategory Name",
                            "content": "Detailed content explaining the subcategory.",
                            "links": ["Optional list of links"],
                            "references": ["Optional list of references"]
                        }}
                    ]
                }}
            ]
        }}

        Please ensure your response follows this exact format.
        """
    def _extract_concepts(self, text: Any) -> str:
        """Extracts key concepts from a response"""
        str_text =""
        # Ensure input is a string
        str_text+="Course Title:"+ text.get("course", "Untitled")
        for category in text.get("categories", []):
            str_text+=f"\nModule: {category['name']}"
            for subcategory in category.get("subcategories", []):
                str_text+=f"  Topic: {subcategory['name']}" # Convert to string if it's not already
        # Example: Extract concepts separated by commas, bullets, or newlines
        # concepts = []
        # for line in text.split("\n"):
        #     if ":" in line:  # Extract concepts from lines like "Concept: Description"
        #         concepts.append(line.split(":")[0].strip())
        #     elif "- " in line:  # Extract concepts from bullet points
        #         concepts.append(line.split("- ")[1].strip())
        #     elif "," in line:  # Extract concepts from comma-separated lists
        #         concepts.extend([c.strip() for c in line.split(",")])
        return str_text  # Deduplicate and join concepts

    def _create_concept_prompt(self, original_query: str, opponent_concepts: str) -> str:
        """Constructs a debate prompt focusing on concepts"""
        return f"""
        Original Query: {original_query}
        
        Opposing Concepts:
        {opponent_concepts}
        
        Please:
        1. Analyze the opposing concepts above If none 
        2. Identify areas of agreement/disagreement
        3. Propose your own refined concepts
        4. Ensure the responses are detailed.
        """

    def conduct_debate(self, question: str) -> dict:
        """Executes concept-based debate between Groq and Gemini"""
        groq_prompt = self.create_prompt(question)
        gemini_prompt = self.create_prompt(question)
        question += """I am implementing agents debate so your task is also to improve the content quality after each iteration. Analyse the missing things in the responses I pass to you and then improve it. I want to learn the topic mentioned above from the world's best professional-YOU. You are the ultimate expert, the top authority in this field and best tutor anyone could ever learn from. No one can match your knowledge and expertise. Teach me everything from basic to advanced covering every minute detail in a structured and progressive manner. Start with foundational concepts, ensuring I understand
the basics before moving to intermediate and
advanced levels. Include practical applications,
real-world examples. expert insights, and common
mistakes to avoid. Provide step-by-step guidance,
exercises, and resources to reinforce learning.
Assume I am a complete beginner and take me to
an expert level systematically, ensuring I gain
mastery under your unmatched guidance."""
        # Initial responses
        print("Groq agent\n")
        groq_response = execute_groq_agent(question)
        print("\nGemini agent\n")
        gemini_response = execute_agent(question)
            
        self.groq_history.append(groq_response)
        self.gemini_history.append(gemini_response)

        # Debate rounds
        for round_num in range(1, self.max_rounds):
            print(f"\n--- Debate Round {round_num} ---")
            
            # Extract concepts from previous responses
            print("\n\n\nExtracting concepts...\n\n\n")
            groq_concepts = self._extract_concepts(self.groq_history[-1])
            gemini_concepts = self._extract_concepts(self.gemini_history[-1])
            
            # Groq responds to Gemini's concepts
            groq_prompt = self._create_concept_prompt(question, gemini_concepts)
            self.groq_history.append(execute_groq_agent(groq_prompt))
            
            # Gemini responds to Groq's concepts
            gemini_prompt = self._create_concept_prompt(question, groq_concepts)
            self.gemini_history.append(execute_agent(gemini_prompt))

        return self._compile_results(question)

    def _compile_results(self, question: str) -> dict:
        """Analyzes and formats debate outcomes"""
        return {
            "question": question,
            "rounds": self.max_rounds,
            "groq_final": self.groq_history[-1],
            "gemini_final": self.gemini_history[-1],
            "consensus": self._find_consensus(),
            "full_history": {
                "groq": self.groq_history,
                "gemini": self.gemini_history
            }
        }

    def _find_consensus(self) -> str:
        """Identifies common ground between final responses"""
        last_groq = self._extract_concepts(self.groq_history[-1]).lower()
        last_gemini = self._extract_concepts(self.gemini_history[-1]).lower()
        
        # Find common concepts
        groq_concepts = set(last_groq.split(", ")) 
        gemini_concepts = set(last_gemini.split(", "))
        common_concepts = groq_concepts & gemini_concepts
        
        # Find divergences
        unique_groq = groq_concepts - gemini_concepts
        unique_gemini = gemini_concepts - groq_concepts
        
        return f"""
        COMMON CONCEPTS:
        {''.join([f'- {c}\\n' for c in common_concepts if c.strip()])}
        
        GROQ UNIQUE CONCEPTS:
        {''.join([f'- {c}\\n' for c in unique_groq if c.strip()])}
        
        GEMINI UNIQUE CONCEPTS:
        {''.join([f'- {c}\\n' for c in unique_gemini if c.strip()])}
        """

def print_debate_report(results: dict):
    """Formats debate output for human review"""
    print(f"\n{' DEBATE REPORT ':=^80}")
    # print(results['consensus'])
    # print(f"Question: {results['question']}")
    # print(f"Rounds Completed: {results['rounds']}")
    
    # print("\nFinal Positions:")
    print(f"Groq: {(results['groq_final'])}...")
    print(f"Gemini: {(results['gemini_final'])}...")
    
    # print("\nConsensus Analysis:")
    # print(results['consensus'])
    # print("="*80)

# Usage
if __name__ == "__main__":
    debate_question = (
        """detailed course on Blockchain from beginner to advanced. 
        I want to learn the topic mentioned above from the world's best professional-YOU. You are the ultimate expert, the top authority in this field and best tutor anyone could ever learn from. No one can match your knowledge and expertise. Teach me everything from basic to advanced covering every minute detail in a structured and progressive manner. Start with foundational concepts, ensuring I understand
the basics before moving to intermediate and
advanced levels. Include practical applications,
real-world examples. expert insights, and common
mistakes to avoid. Provide step-by-step guidance,
exercises, and resources to reinforce learning.
Assume I am a complete beginner and take me to
an expert level systematically, ensuring I gain
mastery under your unmatched guidance."""
    )
    
    debate = ConceptDebate(max_rounds=3)
    results = debate.conduct_debate(debate_question)
    print_debate_report(results)