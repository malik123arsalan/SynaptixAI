import os
import json
from config import Config
from schemas import CompleteStudyPlan
from memory import SynaptixMemory
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser

class SynaptixAgentEngineDirect:
    """Core Orchestration Engine - Bypassing File System completely and hardening JSON quote constraints"""

    def __init__(self):
        os.environ["GROQ_API_KEY"] = Config.GROQ_API_KEY
        self.model = ChatGroq(model_name=Config.GROQ_MODEL, temperature=0.1) # Temperature kam kiya taaki model strict rahe
        self.parser = JsonOutputParser(pydantic_object=CompleteStudyPlan)
        self.memory = SynaptixMemory(token_limit=4)

    def process_chat_turn(self, user_input: str) -> dict:
        # 1. Add current user message to active history
        self.memory.add_message("user", user_input)
        full_context_history = self.memory.get_full_context()

        # 2. Extract schema instructions
        format_instructions = self.parser.get_format_instructions()

        # 3. Create raw system message text string with strict double-quote warning
        system_text = (
            "You are Synaptix-AI, an expert engineering mentor. Create a highly accurate study target based on the conversation history.\n"
            "CRITICAL: You must strictly output valid raw JSON using DOUBLE QUOTES (\") for all keys and string values. Never use single quotes (') for JSON properties.\n"
            "Do not include any markdown backticks or extra text outside the JSON block.\n\n"
            f"Format Instructions:\n{format_instructions}"
        )

        # 4. Pack raw message objects directly without using any Prompt Templates
        messages_payload = [SystemMessage(content=system_text)] + full_context_history

        print(f"⚡ [ENGINE LOG]: Invoking Groq model directly for input: '{user_input}'...")

        # 5. Direct chain execution
        chain = self.model | self.parser
        response_json = chain.invoke(messages_payload)

        # 6. CRITICAL FIX: Save back to memory using json.dumps() to force double quotes format string!
        self.memory.add_message("ai", json.dumps(response_json))

        return response_json



import json

# Initialize our direct memory-resident engine
agent_direct = SynaptixAgentEngineDirect()

print("🏁 Starting Live Day 7 Multi-Turn Integration Test (Direct Mode)\n")

# Turn 1: Initial Requirement
print("--- 🔄 TURN 1 ---")
turn_1_output = agent_direct.process_chat_turn("I want to learn Binary Search Tree (BST) operations today.")
print(json.dumps(turn_1_output, indent=4))

# Turn 2: Refining the request
print("\n--- 🔄 TURN 2 ---")
turn_2_output = agent_direct.process_chat_turn("Actually, also add AVL Tree rotations into the mix since I have an exam tomorrow.")
print(json.dumps(turn_2_output, indent=4))

# Turn 3: Pushing memory thresholds
print("\n--- 🔄 TURN 3 ---")
turn_3_output = agent_direct.process_chat_turn("Perfect. Focus heavily on Left-Right (LR) rotations and resource links.")
print(json.dumps(turn_3_output, indent=4))
