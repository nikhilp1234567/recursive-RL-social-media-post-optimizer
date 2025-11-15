import os, glob
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
context = "\n\n".join([open(f).read() for f in glob.glob("context/*.{md,txt}")])
brand_file = os.path.join(os.path.dirname(__file__), "..", "Brand Strategy Report File.md")
prompt = open(brand_file).read() + f"\n\n--- CONTEXT FILES ---\n{context}"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

report_content = response.choices[0].message.content
output_path = os.path.join(os.path.dirname(__file__), "specific_brand_report.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(report_content)
