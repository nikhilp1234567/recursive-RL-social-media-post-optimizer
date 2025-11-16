from helpers.reinforcement_learning_helpers import DATASET_FILE
import json
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

def check_tweet_length(post:str) -> int:
    if len(post) > 280:
        return 0
    return 1

def check_duplication(current_post:str) -> int:
    dataset = []
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            if line.strip(): 
                dataset.append(json.loads(line.strip()))
    for post in dataset:
        if post['post'] == current_post:
            return 0
    return 1

def validate_tweet_content():
    return

def remove_instructions_from_post(post:str) -> str:
    # instructions are between [INST] and [/INST]
    start_instr = post.find("[INST]")
    end_instr = post.find("[/INST]")
    if start_instr != -1 and end_instr != -1:
        cleaned_post = post[:start_instr] + post[end_instr + len("[/INST]"):]
        return cleaned_post.strip()
    return post

def clean_with_gemini(post:str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="""This is a post which needs to be within 280 characters to be posted on X. 
        While maintaining the key ideas in the post, fix the formatting issues and make it suitable for posting on X. Provide a single post and noting else. 
        Current Post: """ + str(post) ,
    )
    return response.text

def diversify_with_gemini(post:str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="""This is a post which needs to be posted on X (within 280 characters).
        While maintaining the key ideas in the post, paraphrase and make it suitable for posting on X. Provide a single post and noting else. 
        Current Post: """ + str(post) ,
    )
    return response.text

def clean_generated_post(post: str) -> str:
    post = remove_instructions_from_post(post)
    while True:
        if check_tweet_length(post) and check_duplication(post):
            break
        post = clean_with_gemini(post)
        if check_tweet_length(post) and check_duplication(post):
            break
        post = diversify_with_gemini(post)
    return post

# add in link cleaning function