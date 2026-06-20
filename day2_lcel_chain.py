import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Step 1: Secure the API Key (In case you are running a new session)
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "paste your key"

# Step 2: Initialize our Brain (The LLM)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

# Step 3: Define a Dynamic Prompt Template using LangChain
# Notice how we use {topic} instead of hardcoding "Deadlock Conditions"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert GATE CSE Professor. 
    Explain the user's topic strictly in JSON format.
    Provide exactly 3 sharp bullet points.
    Keys required: 'concept_name', 'explanation_points', 'importance_level'."""),
    ("human", "Explain this topic: {topic}")
])

# Step 4: Initialize LangChain's In-built JSON Parser
# This replaces our manual 'json.loads()' and 'try-except' safety logic automatically!
json_parser = JsonOutputParser()

# ==========================================
# 🔥 THE MAGIC OF LCEL: CREATING THE CHAIN 🔥
# ==========================================
# We connect Prompt -> LLM -> Parser using the pipe '|' symbol
synaptix_chain = prompt_template | llm | json_parser

# Step 5: Execute the entire chain with just ONE command!
# Pass the topic dynamically inside a dictionary
print("Running the LangChain LCEL Pipeline...\n")
final_output = synaptix_chain.invoke({"topic": "Process Synchronization"})

# =============================================================
# 🔥 BEAUTIFIED OUTPUT PRINTING (REPLACED SINGLE LINE PRINT) 🔥
# =============================================================

print("\n" + "="*50)
print("🚀 SYNAPTIX AI: STRUCTURED USER-FACING OUTPUT")
print("="*50 + "\n")

# Extract and print individual values using standard Python dictionary keys
print(f"📚 CONCEPT    : {final_output['concept_name']}")
print(f"📊 IMPORTANCE : {final_output['importance_level']}")
print("\n📝 EXPLANATION POINTS:")

# Iterate through the parsed list to display each point on a new line with indexing
for index, point in enumerate(final_output['explanation_points']):
    print(f"  {index + 1}. {point}")

print("\n" + "="*50)
