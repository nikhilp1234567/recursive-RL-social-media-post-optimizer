import os
import glob
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

# Correctly load context files, excluding the main brand report
script_dir = os.path.dirname(os.path.abspath(__file__))
context_path = os.path.join(script_dir, "context")
brand_file = os.path.join(context_path, "Brand Strategy Report File.md")

all_files = glob.glob(os.path.join(context_path, "*.txt")) + glob.glob(os.path.join(context_path, "*.md"))
context_files = [f for f in all_files if f != brand_file]

context = "\n\n".join([open(f).read() for f in context_files])

prompt = open(brand_file).read() + f"\n\n--- CONTEXT FILES ---\n{context}"

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt,
    config=config,
)
report_content = response.text
output_path = os.path.join(script_dir, "specific_brand_report.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(report_content)