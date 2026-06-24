import os
from google.colab import userdata
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# 1. Securely fetch the Groq API key from Colab Secrets
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

# 2. Re-defining our Day 5 strict schema structure
class DailyTask(BaseModel):
    topic_name: str = Field(
        description="The name of the GATE CSE or Placement topic to study today."
    )
    time_required: str = Field(
        description="Estimated time needed, e.g., '2 hours'."
    )
    source_suggested: str = Field(
        description="Strictly mention the website name like GeeksforGeeks or YouTube."
    )


# 3. Initialize the parser and the Groq LLM model
parser = JsonOutputParser(pydantic_object=DailyTask)
model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# 4. Crafting the prompt with dynamic format instructions
prompt_template = """
You are Synaptix AI, a strict personal mentor. Generate a highly targeted 1-day study target for a student preparing for OS Semaphores.
{format_instructions}
"""

final_prompt = prompt_template.format(
    format_instructions=parser.get_format_instructions()
)

# 5. Invoking the model and parsing the raw response into clean JSON
print("🚀 Launching Groq Engine... Fetching Structured Response...\n")
raw_response = model.invoke(final_prompt)
parsed_output = parser.parse(raw_response.content)

# 6. Printing the final clean output
import json

print(json.dumps(parsed_output, indent=4))
