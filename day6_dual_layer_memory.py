%%writefile memory.py
import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class SynaptixMemory:
    """
    Advanced Dual-Layer Memory System for Synaptix-AI.
    Layer 1: Live Sliding Window Context to optimize LLM token limits (RAM).
    Layer 2: Persistent Long-Term Archive Logs for historical retrieval (Hard Disk).
    """

    def __init__(self, token_limit=4):
        # Rolling window threshold for active conversation memory
        self.token_limit = token_limit
        self.chat_history = []
        self.rolling_summary = ""

        # Mocking the permanent cold-storage database for long-term recovery
        self.permanent_database_archive = []

    def add_message(self, role: str, content: str):
        """
        Ingests new messages into both the live operational memory
        and the permanent structural archive logs.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Archive permanently into Layer 2 (Simulated Cold Storage Database)
        # This ensures data is never lost, even after a year or more.
        self.permanent_database_archive.append({
            "timestamp": timestamp,
            "role": role,
            "content": content
        })

        # 2. Append into Layer 1 (Active Chat History for LLM Context)
        if role == "user":
            self.chat_history.append(HumanMessage(content=content))
        elif role == "ai":
            self.chat_history.append(AIMessage(content=content))

        # Core optimization trigger: Prune history if it violates context thresholds
        if len(self.chat_history) > self.token_limit:
            self._trigger_selective_summarization()

    def _trigger_selective_summarization(self):
        """
        Condenses older messages in the live frame into a rolling semantic summary.
        Clears out buffer space while preserving conversation trajectory.
        """
        print("\n🚨 [SYSTEM ALERT]: Live Context Limit Exceeded! Triggering Memory Optimization...")

        # Extract the oldest 2 messages to compile into the rolling summary
        messages_to_summarize = self.chat_history[:2]

        # Create a raw structural snapshot for semantic compression
        summary_chunk = " | ".join([f"{type(m).__name__}: {m.content}" for m in messages_to_summarize])

        # Append to historical rolling summary block
        self.rolling_summary += f" [Past Context Snapshot: {summary_chunk}] "

        # Prune and drop the processed messages from active RAM allocation
        self.chat_history = self.chat_history[2:]

        print("✅ [SYSTEM INFO]: Live token window cleared. Long-term logs remain fully archived.")

    def get_full_context(self):
        """
        Compiles the processed system summary and current active messages
        into a finalized payload ready for LLM processing.
        """
        context_messages = []
        if self.rolling_summary:
            context_messages.append(SystemMessage(content=f"Summary of past conversation steps: {self.rolling_summary}"))

        return context_messages + self.chat_history

    def fetch_historical_archive(self):
        """
        Simulates data retrieval from cold-storage for 1-year old logs.
        """
        return self.permanent_database_archive
