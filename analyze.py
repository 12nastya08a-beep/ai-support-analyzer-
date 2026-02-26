"""
Script to analyze customer support chats from a JSON dataset.
Outputs a new JSON file with intent, satisfaction, quality_score, and agent_mistakes.
"""

import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv


def setup_ai():
    """
    Loads environment variables and initializes the GenAI client.
    Returns the configured client or None if the API key is missing.
    """
    load_dotenv(override=True)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is missing in the .env file!")
        return None
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def analyze_chat(client, chat_text: str) -> dict:
    """
    Analyzes a single customer support chat using Gemini 2.0 Flash.
    Safety filters are explicitly disabled to process angry or sarcastic customer messages.
    Returns a dictionary containing intent, satisfaction, quality score, and mistakes.
    """
    system_prompt = """
    Analyze the chat and return EXACTLY a JSON object with these keys:
    - "intent": (payment_issue, technical_error, account_access, tariff_question, refund_request, or other)
    - "satisfaction": (satisfied, neutral, or unsatisfied). Pay attention to sarcasm or hidden dissatisfaction!
    - "quality_score": (integer 1 to 5)
    - "agent_mistakes": (list: ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation. Leave empty [] if none)
    """

    # Disable safety filters so the AI doesn't block rude/sarcastic test chats
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=chat_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
            safety_settings=safety_settings
        )
    )

    return json.loads(response.text)


if __name__ == "__main__":
    # Prevent overwriting the mocked presentation data
    if os.path.exists("analyzed_dataset.json"):
        print("Analyzed dataset already exists. Skipping analysis to protect presentation data.")
        exit()

    # Initialize the AI client
    ai_client = setup_ai()
    if not ai_client:
        exit()

    # Load the generated dataset
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: 'dataset.json' not found! Please ensure the file exists.")
        exit()

    analyzed_results = []
    print(f"Starting analysis of {len(dataset)} chats...\n")

    # Loop through each chat and analyze it
    for item in dataset:
        print(f"Analyzing chat #{item['id']}...")

        try:
            # Get the analysis from Gemini
            analysis = analyze_chat(ai_client, item['dialogue'])

            # Add the analysis to our data
            item['analysis'] = analysis
            analyzed_results.append(item)

            # Save progress incrementally to a new file
            with open("analyzed_dataset.json", "w", encoding="utf-8") as f:
                json.dump(analyzed_results, f, ensure_ascii=False, indent=4)

            # Pause to respect API rate limits
            time.sleep(61)

        except Exception as error:
            print(f"Error processing chat #{item['id']}: {error}")

    print("\nAnalysis complete! Results saved to 'analyzed_dataset.json'.")
