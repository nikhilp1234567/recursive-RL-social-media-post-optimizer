import os
import json
import uuid
from datetime import datetime
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
report_path = os.path.join(os.path.dirname(__file__), "specific_brand_report.txt")
brand_report = open(report_path).read()

model = genai.GenerativeModel('gemini-2.5-pro')

# Step 1: Generate trends
trends_response = model.generate_content(
    f"{brand_report}\n\nGenerate 5-7 current trends/topics/events relevant to this brand that would make good social media posts. List them numbered with no other text."
)
trends = trends_response.text

# Step 2: Pick best trend, create prompt and save
final_prompt_response = model.generate_content(
    f"Brand Report:\n{brand_report}\n\nTrends:\n{trends}\n\nPick the best current trend/topic/event and create a detailed prompt for generating a social media post that aligns with the brand voice. output only the prompt with no other text."
)
final_prompt = final_prompt_response.text

output_path = os.path.join(os.path.dirname(__file__), "data", "prompts.jsonl")
id = str(uuid.uuid4())
with open(output_path, "a") as f:
    json.dump({
        "id": id,
        "date": datetime.now().isoformat(),
        "prompt": final_prompt
    }, f)
    f.write("\n")




