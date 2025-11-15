import os
from openai import OpenAI
import json
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
report_path = os.path.join(os.path.dirname(__file__), "specific_brand_report.txt")
brand_report = open(report_path).read()

# Step 1: Generate trends
trends = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"{brand_report}\n\nGenerate 5-7 current trends/topics/events relevant to this brand that would make good social media posts. List them numbered with no other text."}]
).choices[0].message.content

# Step 2: Pick best trend, create prompt and save
final_prompt = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Brand Report:\n{brand_report}\n\nTrends:\n{trends}\n\nPick the best current trend/topic/event and create a detailed prompt for generating a social media post that aligns with the brand voice. output only the prompt with no other text."}]
).choices[0].message.content

output_path = os.path.join(os.path.dirname(__file__), "data", "prompts.jsonl")
id = str(uuid.uuid4())
with open(output_path, "a") as f:
    import uuid
    json.dump({
        "id": id,
        "date": datetime.now().isoformat(),
        "prompt": final_prompt
    }, f)
    f.write("\n")

# step 3: train model and run with generated prompt 



