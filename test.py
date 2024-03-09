import csv
import google.generativeai as genai
import re
import random

# Configure the generative AI model
genai.configure(api_key="AIzaSyBSW7znT-UMBm2wfD5qHXEmA23gpkkYamo")

# Set up the model generation configuration and safety settings
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 5000,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings
)

# Function to clean tweet text
def clean_tweet(tweet):
    return re.sub(r'[^\w\s]', '', tweet)

# Function to clean response text
def clean_response(response):
    return re.sub(r'[*]', '', response).strip()

# List of prompts
prompts = [
    "Here is the tweet: '{tweet}'. Annotate the following tweet with one of the following labels: Positive, Extremely positive, Negative, Extremely Negative. Also tell me the reason of your answer",
    "Read the given tweet: '{tweet}'. Read this carefully and annotate it with one of the following labels: Positive, Extremely positive, Negative, Extremely Negative. Also give explanation of your choice",
    "Consider this tweet: '{tweet}'. Analyze it carefully and annotate this with one of the following labels: Positive, Extremely positive, Negative, Extremely Negative. Also give the good reason for this."
]

# Read the CSV file and write annotated data to separate CSV files for each prompt
with open("Corona_NLP_test.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for index, prompt in enumerate(prompts, start=1):
        with open(f"corona_NLP_test_annotated_prompt{index}.csv", "w", newline="", encoding="utf-8") as outfile:
            fieldnames = ["No.", "tweet", "Prompt", "generated annotations", "explanation"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, row in enumerate(reader, start=1):
                tweet = clean_tweet(row["OriginalTweet"])

                try:
                    convo = model.start_chat(history=[])
                    convo.send_message(prompt.format(tweet=tweet))

                    if convo and convo.last:
                        response = convo.last.text

                        if "HARM_CATEGORY" in response:
                            label = "Safety setting triggered"
                            reason = response.strip()
                        else:
                            if "." in response:
                                parts = response.split(".")
                                if len(parts) > 1:
                                    label = parts[0].strip()
                                    reason = ".".join(parts[1:]).strip()
                                    reason = clean_response(reason)

                                    allowed_labels = {"Positive", "Extremely positive", "Negative", "Extremely Negative"}
                                    if label not in allowed_labels:
                                        label = "Didn't Understand"
                                else:
                                    label = "Neutral"
                                    reason = "No specific reason given"
                            else:
                                label = "Neutral"
                                reason = "No specific reason given"
                    else:
                        label = "Error: Unable to get response from the model"
                        reason = "Error: Unable to get response from the model"
                except Exception as e:
                    print("Error occurred during conversation:", e)
                    label = "Error: Unable to analyze tweet"
                    reason = "Error: Unable to analyze tweet"

                writer.writerow({
                    "No.": i,
                    "tweet": tweet,
                    "Prompt": prompt,
                    "generated annotations": label,
                    "explanation": reason
                })
