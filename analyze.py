"""
Script to analyze customer support chats.
Outputs JSON with intent, satisfaction, quality_score, and agent_mistakes.
"""

import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv


def setup_ai():
    """
    Loads environment variables from .env file.
    """
    load_dotenv(override=True)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is missing in the .env file!")


def analyze_chat(chat_text: str) -> dict:
    """
    Analyzes a customer support chat using the NEW Gemini SDK.
    Returns a dictionary containing intent, satisfaction, quality score, and mistakes.
    """
    system_prompt = """
    Analyze the chat and return EXACTLY a JSON object with these keys:
    - "intent": (payment_issue, technical_error, account_access, tariff_question, refund_request, or other)
    - "satisfaction": (satisfied, neutral, or unsatisfied). Pay attention to sarcasm or hidden dissatisfaction!
    - "quality_score": (integer 1 to 5)
    - "agent_mistakes": (list: ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation. Leave empty [] if none)
    """

    # 1. Initialize the client using the API key
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # 2. Send the request using the new SDK rules
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=chat_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
        )
    )

    # 3. Convert the AI's text response into a Python dictionary
    return json.loads(response.text)


# ----- Test Block -----
if __name__ == "__main__":
    setup_ai()

    test_chat = """
    Client: У мене списало гроші двічі за один і той самий тариф!
    Agent: Вітаю. Наші тарифи можна переглянути за посиланням на сайті.
    Client: Ну дуже вам "дякую" за допомогу. Ви зовсім не читали моє питання, піду в іншу компанію.
    Agent: Гарного дня! Завжди раді допомогти.
    """

    print("Analyzing test chat using the NEW Google GenAI SDK...\n")

    try:
        result = analyze_chat(test_chat)
        print(json.dumps(result, indent=4, ensure_ascii=False))
    except Exception as error:
        print(f"Oops, something went wrong: \n {error}")