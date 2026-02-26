"""
Script to generate synthetic customer support chats using LLMs.
Uses a fallback strategy: Google Gemini -> Groq -> Cohere.
Covers various scenarios and outputs English dialogues.
"""

import os
import json
import time

from dotenv import load_dotenv
from google import genai
from groq import Groq
import cohere


def setup_environment():
    """
    Loads environment variables from the .env file.
    Must be called before initializing any API clients.
    """
    load_dotenv(override=True)


def generate_single_chat_with_fallback(scenario: str) -> str:
    """
    Generates a dialogue using multiple API providers.
    Uses Cohere V2 API for the final fallback.
    Returns the generated chat string.
    """
    prompt_instruction = f"""
    Write a realistic customer support dialogue in ENGLISH.
    Scenario: {scenario} 
    Format exactly like this:
    Client: [text]
    Agent: [text]
    """

    # Step 1: Try Google Gemini
    try:
        print("  -> Generating with Google Gemini...")
        client_google = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client_google.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_instruction,
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        print(f"  [!] Google Gemini failed. Switching to Groq...")

    # Step 2: Try Groq (Llama 3)
    try:
        print("  -> Generating with Groq...")
        client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client_groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": prompt_instruction}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [!] Groq failed. Switching to Cohere...")

    # Step 3: Try to Cohere (V2 API)
    try:
        print("  -> Generating with Cohere (V2)...")
        co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "user", "content": prompt_instruction}
            ],
            temperature=0.7
        )
        # Extract the text from the V2 response structure
        return response.message.content[0].text.strip()
    except Exception as e:
        print(f"  [!] All APIs failed. Final error: {e}")
        return "Client: Error generating chat.\nAgent: Please check API keys."


def main():
    """
    Main pipeline for generating and saving the dataset.
    Deletes the old dataset to start fresh.
    """
    setup_environment()

    # Check if the file exists, and remove it to start fresh
    if os.path.exists("dataset.json"):
        print("Dataset already exists. Deleting old file and generating new data...")
        os.remove("dataset.json")

    scenarios = [
        "1. Refund request. Agent makes a mistake. Client is angry.",
        "2. Technical error. Agent gives useless links. Sarcasm.",
        "3. Tariff question. Successful case. Satisfied client.",
        "4. Payment issue. Double charge. Perfect resolution.",
        "5. Account access. Password reset fails. Agent ignores issue.",
        "6. Other. Physical address request. Quick response.",
        "7. Technical error. App crashes. Rude agent.",
        "8. Refund request. Wrong item. Exception made.",
        "9. Tariff question. Discount request. Free month offered.",
        "10. Payment issue. Card declined. Incorrect info given.",
        "11. Account access. Hacked account. Escalation.",
        "12. Technical error. Slow site. Agent is dismissive.",
        "13. Other. Account deletion. Manipulation attempt.",
        "14. Refund request. Partial refund. Calculation error.",
        "15. Tariff question. Hidden fees. Sarcasm.",
        "16. Payment issue. Promo code fail. Manual fix.",
        "17. Account access. 2FA issue. Unnecessary personal info.",
        "18. Technical error. Video issue. Workaround found.",
        "19. Refund request. Policy denial. Polite explanation.",
        "20. Tariff question. Upgrade plan. Quick fuss-free fix."
    ]

    generated_dataset = []
    print(f"Starting generation of {len(scenarios)} chats with fallback strategy...\n")

    for i, scenario in enumerate(scenarios):
        print(f"[{i + 1}/{len(scenarios)}] Processing scenario...")

        chat_text = generate_single_chat_with_fallback(scenario)

        generated_dataset.append({
            "id": i + 1,
            "scenario_type": scenario,
            "dialogue": chat_text
        })

        with open("dataset.json", "w", encoding="utf-8") as file:
            json.dump(generated_dataset, file, ensure_ascii=False, indent=4)

        time.sleep(2)

    print("\nGeneration complete. Data saved to 'dataset.json'.")


if __name__ == "__main__":
    main()
