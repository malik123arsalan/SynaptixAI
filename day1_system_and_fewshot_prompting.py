# ==========================================
# Project: Synaptix AI (GATE & Career Co-Pilot)
# Day 1: System Prompting & Few-Shot Examples using LangChain
# Framework: Pure LangChain & Groq Cloud API
# ==========================================

import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Step 1: Set the Groq API Key environment variable
# NOTE: Replace with your actual key when running locally/Colab
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"

# Step 2: Initialize the LangChain ChatGroq model wrapper
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.1
)

# Step 3: Advanced System Instruction with structured guidelines and dynamic examples
advanced_system_instruction = """
You are the Lead Computer Science Professor for Synaptix AI. 
Your job is to explain GATE CSE/DA concepts strictly in JSON format.

CRITICAL RULES:
1. Give the core concept in exactly 3 sharp bullet points.
2. Provide keys: 'concept_name', 'explanation_points', and 'importance_level'.

---
EXAMPLE 1 (Few-Shot Reference):
Student Query: "What is a Primary Key?"
Your Output:
{
  "concept_name": "Primary Key in DBMS",
  "explanation_points": [
    "A Primary Key uniquely identifies each record in a database table.",
    "It cannot contain NULL values and must contain unique values only.",
    "A table can have only one primary key, which acts as the default index."
  ],
  "importance_level": "Extremely High (Base of Relational Algebra & Normalization)"
}
---
"""

# Step 4: Define user input query for evaluation
student_query = "Explain Deadlock Conditions"

# Step 5: Structure the message list using LangChain core message schemas
messages = [
    SystemMessage(content=advanced_system_instruction),
    HumanMessage(content=student_query)
]

# Step 6: Invoke the LLM call using LangChain interface
response = llm.invoke(messages)

# Step 7: Print the deterministic JSON payload response
print(response.content)
