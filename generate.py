"""
Script to generate synthetic customer support chats using LLM.
Covers various scenarios: payment, tech issues, refunds.
"""

import os
import json
import time
from google import genai
from dotenv import load_dotenv


def setup_client():
    """Initializes the GenAI client with API key validation."""
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is missing!")
        return None
    return genai.Client(api_key=api_key)


def get_working_model(client):
    """Detects the best available Flash model version."""
    try:
        available_models = [m.name for m in client.models.list()]
        preferred_models = [
            'gemini-2.0-flash',
            'gemini-1.5-flash-002',
            'gemini-1.5-flash'
        ]
        for preferred in preferred_models:
            if (preferred in available_models or f"models/{preferred}" in
                    available_models):
                return preferred
        for m in available_models:
            if "flash" in m.lower():
                return m
    except Exception as e:
        print(f"Failed to retrieve model list: {e}")
    return "gemini-1.5-flash"


def generate_single_chat(client, model_name, scenario: str) -> str:
    """
    Generates dialogue. Raises an Exception if the API fails
    or returns an empty response.
    """
    prompt_instruction = f"""
    Write a realistic customer support dialogue in Ukrainian.
    Scenario: {scenario} 
    Format:
    Client: [text]
    Agent: [text]
    """
    response = client.models.generate_content(
        model=model_name,
        contents=prompt_instruction,
    )

    if response and response.text:
        return response.text.strip()

    raise Exception("Empty response or blocked by safety filters")


def main():
    """Main pipeline for generating and saving the dataset."""
    ai_client = setup_client()
    if not ai_client:
        return

    target_model = get_working_model(ai_client)
    print(f"Using model: {target_model}")

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

    for i, scenario in enumerate(scenarios):
        print(f"[{i + 1}/{len(scenarios)}] Generating dialogue...")

        try:
            # Тепер ми просто викликаємо функцію
            chat_text = generate_single_chat(ai_client, target_model, scenario)

            generated_dataset.append({
                "id": i + 1,
                "scenario_type": scenario,
                "dialogue": chat_text
            })

            # Save progress incrementally
            with open("dataset.json", "w", encoding="utf-8") as file:
                json.dump(generated_dataset, file, ensure_ascii=False,
                          indent=4)

            time.sleep(4)

        except Exception as e:
            error_msg = str(e)
            print(f"Skipping scenario {i + 1} due to error: {error_msg}")

            if "429" in error_msg:
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
            continue

    print("\nGeneration complete. Data saved to 'dataset.json'.")


if __name__ == "__main__":
    main()
