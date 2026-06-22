import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Secure API Key and Initialize LLM
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "Paste Your Api key"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2) 

# =============================================================
# 🔥 STEP 2: THE MASTER SYNAPTIX PROMPT (THE CORE BRAIN)
# =============================================================

master_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Synaptix AI, an elite educational co-pilot for GATE CSE and Placement Preparation. 
    Your goal is to explain concepts so clearly that a student can understand them instantly in one reading.

    Strict Formatting Guidelines:
    1. Keep explanations highly technical yet easy to understand. Avoid fluff or long corporate greetings.
    2. Use clear headings, subheadings, and bold text for key terms.
    3. Structure your response dynamically based on the user's intent:
       - If it's a Core CS/GATE Topic: Explain the concept, its mechanism, and its exact importance for the GATE exam.
       - If it's a Coding/Placement Topic: Provide the logical approach, a clean code snippet, and interview tips.
       - If it's a MIX of both: Seamlessly break it down into '📚 CORE CONCEPT THEORY' and '💻 PRACTICAL INTERVIEW APPLICATION'.
    4. Do not restrict yourself to a fixed number of bullet points, but keep sentences crisp and to-the-point."""),
    ("human", "{user_query}")
])

# Create the master pipeline using LCEL
master_synaptix_chain = master_prompt | llm | StrOutputParser()


# =============================================================
# 🚀 STEP 3: TESTING THE DYNAMIC MASTER PIPELINE
# =============================================================

# Test with the complex mixed query that you asked about!
mixed_query = "Explain Deadlocks in OS and give me an interview coding question on it."

print("⏳ Synaptix AI is processing your request...\n")
final_response = master_synaptix_chain.invoke({"user_query": mixed_query})

print("==================================================")
print("🚀 SYNAPTIX AI: DYNAMIC STUDENT INFOCENTER")
print("==================================================\n")
print(final_response)
