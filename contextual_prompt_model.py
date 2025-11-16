import os
import json
import uuid
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini API: {str(e)}")

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

report_path = os.path.join(os.path.dirname(__file__), "specific_brand_report.txt")
try:
    with open(report_path, "r", encoding="utf-8") as f:
        brand_report = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Brand report file not found: {report_path}")
except IOError as e:
    raise IOError(f"Failed to read brand report file: {str(e)}")

# Step 1: Generate trends
try:
    trends_response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=f"{brand_report}\n\nGenerate 5-7 current trends/topics/events relevant to this brand that would make good social media posts for the twitter account of a . List them with no other text. ensure you use the above brand report to accurately target the industry and landscape of the brand.",
        config=config,
    )
except Exception as e:
    raise RuntimeError(f"Failed to generate trends from model: {str(e)}")

try:
    trends = trends_response.text
except AttributeError:
    raise ValueError("Model response does not contain text attribute. Response may have been blocked or failed.")

# Save trends to topics.jsonl
topics_output_path = os.path.join(os.path.dirname(__file__), "data", "topics.jsonl")
os.makedirs(os.path.dirname(topics_output_path), exist_ok=True)

try:
    with open(topics_output_path, "a", encoding="utf-8") as f:
        json.dump({
            "id": str(uuid.uuid4()),
            "date": datetime.now().isoformat(),
            "topics": trends
        }, f)
        f.write("\n")
except IOError as e:
    raise IOError(f"Failed to write topics to output file: {str(e)}")
except Exception as e:
    raise RuntimeError(f"Unexpected error while writing topics: {str(e)}")

# Step 2: Pick best trend, create prompt and save
try:
    final_prompt_response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=f"Brand Report:\n{brand_report}\n\nTrends:\n{trends}\n\nPick the best current trend/topic/event and create a detailed prompt for generating a social media post that aligns with the brand voice. output only the prompt with no other text.",
        config=config,
    )
except Exception as e:
    raise RuntimeError(f"Failed to generate final prompt from model: {str(e)}")

try:
    final_prompt = final_prompt_response.text
except AttributeError:
    raise ValueError("Model response does not contain text attribute. Response may have been blocked or failed.")

output_path = os.path.join(os.path.dirname(__file__), "data", "prompts.jsonl")
id = str(uuid.uuid4())

# Ensure data directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump({
            "id": id,
            "date": datetime.now().isoformat(),
            "prompt": final_prompt
        }, f)
        f.write("\n")
except IOError as e:
    raise IOError(f"Failed to write prompt to output file: {str(e)}")
except Exception as e:
    raise RuntimeError(f"Unexpected error while writing output: {str(e)}")