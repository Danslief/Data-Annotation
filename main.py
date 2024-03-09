import csv
import google.generativeai as genai
import re
import random

genai.configure(api_key="AIzaSyBSW7znT-UMBm2wfD5qHXEmA23gpkkYamo")

# Set up the model
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

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings
)


# Function to clean tweet text
def clean_tweet(tweet):
    # Remove special characters like '#'
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet


# Function to clean response text
def clean_response(response):
    # Remove asterisks and other special characters
    response = re.sub(r'[*]', '', response)
    return response.strip()


# List of prompts
prompts = [
    "Here is the tweet: '{tweet}'.Annotate the following tweet carefully with one of the following labels: Positive, Extremely positive, Negative, Extremely Negative. Also tell me the reason of your answer",
    "Read the given tweet: '{tweet}'. Read this carefully and annotate it with one of the following labels: Positive, Extremely positive, Negative, Extremely Negative. Also give explanation of your choice",
    "Consider this tweet: '{tweet}'. Analyze it carefully and annotate this with one of the following labels: Positive, Extremely positive, Negative, Extremely Negative. Also give the good reason for this."
]

# Read the CSV file and write annotated data directly to the output CSV file
with open("Corona_NLP_test.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = ["No.", "tweet", "Prompt", "generated annotations", "explanation"]

    with open("corona_NLP_test_annotated.csv", "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader, start=1):
            tweet = row["OriginalTweet"]
            # Clean tweet text
            tweet = clean_tweet(tweet)

            # Randomly select a prompt
            prompt = random.choice(prompts).format(tweet=tweet)

            try:
                convo = model.start_chat(history=[])
                convo.send_message(prompt)

                if convo and convo.last:
                    response = convo.last.text
                    print("Response:", response)  # Print response for debugging

                    # Check if the response contains safety settings message
                    if "HARM_CATEGORY" in response:
                        label = "Safety setting triggered"
                        reason = response.strip()
                    else:
                        # Check if the response contains annotations and reasons
                        if "." in response:
                            # Split response into annotations and reasons
                            parts = response.split(".")
                            if len(parts) > 1:  # Ensure both annotation and reason are present
                                # The first part contains the annotation
                                label = parts[0].strip()

                                # The second part contains the reason
                                reason = ".".join(parts[1:]).strip()
                                reason = clean_response(reason)  # Clean response text

                                allowed_labels = {"Positive", "Extremely positive", "Negative", "Extremely Negative"}
                                if label not in allowed_labels:
                                    label = "Didn't Understands"
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
                print("Error occurred during conversation:", e)  # Print exception for debugging
                label = "Error: Unable to analyze tweet"
                reason = "Error: Unable to analyze tweet"

            writer.writerow({
                "No.": i,
                "tweet": tweet,
                "Prompt": prompt,
                "generated annotations": label,
                "explanation": reason
            })