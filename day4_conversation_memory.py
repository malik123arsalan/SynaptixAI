# Install the community package required for ChatMessageHistory
!pip install langchain-community
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Step 1: Securely fetch the API key and initialize the LLM
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "Paste Your Api key"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# =============================================================
# 🔥 STEP 2: DEFINE THE PROMPT WITH A MEMORY PLACEHOLDER
# =============================================================
# The MessagesPlaceholder dynamically injects past conversation history into the prompt context.
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Synaptix AI, a precise GATE CSE tutor. Give short, conceptual answers."),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{user_query}")
])

# Define the base execution chain using LangChain Expression Language (LCEL)
base_chain = memory_prompt | llm | StrOutputParser()


# =============================================================
# 🧠 STEP 3: CONFIGURE IN-MEMORY SESSION STORAGE
# =============================================================
# A dictionary to store independent chat histories for different user sessions.
sessions_storage = {}

def get_session_history(session_id: str):
    """
    Retrieves or creates a new ChatMessageHistory instance for a given session ID.
    """
    if session_id not in sessions_storage:
        sessions_storage[session_id] = ChatMessageHistory()
    return sessions_storage[session_id]


# Wrap the base chain with message history tracking capabilities
synaptix_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="user_query",
    history_messages_key="chat_history"
)


# =============================================================
# 🚀 STEP 4: EXECUTE AND VALIDATE MULTI-TURN CONVERSATION
# =============================================================
# Unique configuration dictionary to identify the specific student session
config = {"configurable": {"session_id": "student_1"}}

# First Turn: Asking a fresh conceptual question
print("🤖 Student: What is Semaphores in OS?")
response_1 = synaptix_with_memory.invoke({"user_query": "What is Semaphores in OS?"}, config=config)
print(f"✨ Synaptix: {response_1}\n")
print("-" * 50 + "\n")

# Second Turn: Asking a follow-up question relying on past context
print("🤖 Student: What are the two main operations on it?")
response_2 = synaptix_with_memory.invoke({"user_query": "What are the two main operations on it?"}, config=config)
print(f"✨ Synaptix: {response_2}")
