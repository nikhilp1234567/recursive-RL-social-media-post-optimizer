import os
import glob
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Exclude the brand file from context since it's loaded separately
brand_file = os.path.join(os.path.dirname(__file__), "context", "Brand Strategy Report File.md")
context_files = [f for f in glob.glob("context/*.{md,txt}") if not f.endswith("Brand Strategy Report File.md")]
context = "\n\n".join([open(f).read() for f in context_files])

prompt = open(brand_file).read() + f"\n\n--- CONTEXT FILES ---\n{context}"

model = genai.GenerativeModel('gemini-2.5-pro')
response = model.generate_content(prompt)

report_content = response.text
output_path = os.path.join(os.path.dirname(__file__), "specific_brand_report.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(report_content)
