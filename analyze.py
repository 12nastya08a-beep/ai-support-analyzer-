"""
Script to analyze customer support chats from a JSON dataset.
Uses a fallback strategy: Cohere -> Google Gemini -> Groq.
Outputs a new JSON file with intent, satisfaction, quality_score, and agent_mistakes.
"""

import json
import os
import time

import cohere
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq


def setup_environment():
    """
    Loads environment variables from the .env file.
    Must be called before initializing any API clients.
    """
    load_dotenv(override=True)


def analyze_chat_with_fallback(chat_text: str) -> dict:
    """
    Analyzes a single customer support chat using multiple API providers.
    Uses Cohere V2 API for the final fallback.
    Returns a dictionary containing intent, satisfaction, quality score, and mistakes.
    """
    system_prompt = """
    Analyze the chat and return EXACTLY a JSON object with these keys:
    - "intent": (payment_issue, technical_error, account_access, tariff_question, refund_request, or other)
    - "satisfaction": (satisfied, neutral, or unsatisfied). Pay attention to sarcasm!
    - "quality_score": (integer 1 to 5)
    - "agent_mistakes": (list: ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation. Leave empty [] if none)
    """

     # Step 1: Try to Cohere (V2 API)
    try:
        print("  -> Attempting with Cohere (V2)...")
        co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

        # Combine system prompt and user chat for Cohere V2
        full_prompt = system_prompt + "\n\nChat to analyze:\n" + chat_text

        # Using the new active model
        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0
        )

        # Extract the raw text
        result_text = response.message.content[0].text.strip()

        # Robust JSON extraction: find the first '{' and last '}' to strip markdown/text
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            # Slice the string to keep ONLY the valid JSON structure
            clean_json_str = result_text[start_idx:end_idx + 1]
        else:
            clean_json_str = result_text

        return json.loads(clean_json_str)
    except Exception as e:
        print(f"  [!] Cohere failed. Switching to Google Gemini...")

    # Step 2: Try Google Gemini
    try:
        print("  -> Attempting with Google Gemini...")
        client_google = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # Disable safety filters for testing angry customer scenarios
        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        response = client_google.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=chat_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=safety_settings
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [!] Google Gemini failed. Switching to Groq...")
        # Step 3: Try Groq (Llama 3)
    try:
        print("  -> Attempting with Groq...")
        client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"  [!] All APIs failed. Final error: {e}")
        return {"intent": "error", "satisfaction": "neutral", "quality_score": 0, "agent_mistakes": []}


if __name__ == "__main__":
    setup_environment()

    # Check if the analyzed file exists, and remove it to start fresh
    if os.path.exists("analyzed_dataset.json"):
        print("Analyzed dataset already exists. Deleting old file to start fresh...")
        os.remove("analyzed_dataset.json")

    # Load the generated dataset
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: 'dataset.json' not found! Run generate.py first.")
        exit()

    analyzed_results = []
    print(f"Starting analysis of {len(dataset)} chats with fallback strategy...\n")

    for item in dataset:
        print(f"Analyzing chat #{item['id']}...")

        analysis = analyze_chat_with_fallback(item['dialogue'])
        item['analysis'] = analysis
        analyzed_results.append(item)

        # Save progress incrementally
        with open("analyzed_dataset.json", "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, ensure_ascii=False, indent=4)

        # Small pause to be polite to the APIs
        time.sleep(2)

    print("\nAnalysis complete! Results saved to 'analyzed_dataset.json'.")