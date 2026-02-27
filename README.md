# AI Support Analyzer 🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LLM: Multi-Model](https://img.shields.io/badge/LLM-Cohere%20%7C%20Gemini%20%7C%20Groq-orange.svg)](https://cohere.com/)
[![Docker: Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/dariatrukhan/ai-support-analyzer)

> **AI Support Analyzer** - is an intelligent system designed for automated generation and analysis of customer support dialogues.

_The project leverages state-of-the-art Large Language Models (LLMs) to evaluate agent performance, identify user intent, and detect hidden customer dissatisfaction._

## 👥 Team

- **Anastasiia Mykytiuk** - Architecture & Repo Setup
- **Darya Miroshnichenko** - Prompt Engineering & Data Generation
- **Daryna Bogaevska** - Backend Logic, AI Analysis, Multi-LLM Fallback & Docker Support
- **Daria Trukhan** - QA & Documentation

---

## 🚀 Key Features

- **Multi-Model Strategy (Fallback)**: Ensures high reliability using a cascading API system: **Cohere → Google Gemini → Groq**. If one service is unavailable, the system automatically switches to the next one.

- **Synthetic Data Generation**: Creates realistic support dialogues across 20 different scenarios (e.g., payment issues, technical bugs, refund requests).
- **Deep NLU Analysis**: Automatically extracts key metrics from every conversation:
  - **Intent**: Precise categorization of the user's request.
  - **Satisfaction**: Sentiment analysis (detecting frustration and even sarcasm).
  - **Quality Score**: Evaluation of agent performance on a 1-5 scale.
  - **Error Detection**: Identifying specific mistakes (e.g., ignoring questions, lack of empathy).
- **Containerization**: Fully Docker-ready for fast, isolated, and consistent deployment.

## 🏗️ Architecture

The project consists of two core modules:

1.  **`generate.py`**: Utilizes LLMs to create a high-quality synthetic dataset of support interactions.
2.  **`analyze.py`**: Processes the dialogues and transforms raw text into structured JSON data with deep analytical insights.

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   generate.py   │ ───> │  dataset.json   │ ───> │   analyze.py    │
│ (LLM Generator) │      │ (Raw Dialogues) │      │ (LLM Analyzer)  │
└─────────────────┘      └─────────────────┘      └────────┬────────┘
                                                           │
                                                ┌──────────▼──────────┐
                                                │analyzed_dataset.json│
                                                │  (JSON Insights)    │
                                                └─────────────────────┘

```

---

## 📦 Installation & Setup

### 1. Environment Preparation

Create a `.env` file in the root directory and add your API keys:

```env
COHERE_API_KEY=your_cohere_key
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

---

## 🚀 How to Run

You can run this project either using Docker (recommended for a quick start) or by setting up a Local Python Environment.

### Option 1: Running with Docker

1.  #### Configure Environment:

    Create a .env file and add your keys
    _(written above)_

2.  #### Generate Data:

    ```Bash
    docker run -it --env-file .env -v $(pwd):/app dariatrukhan/ai-support-analyzer python generate.py
    ```

3.  #### Analyze Data:

    ```Bash
    docker run -it --env-file .env -v $(pwd):/app dariatrukhan/ai-support-analyzer python analyze.py
    ```

    _The results (`dataset.json` and `analyzed_dataset.json`) will appear in your current folder._

### Option 2: Local Python Setup

Use this method if you want to modify the code or run it natively.

1. #### Clone the Repository:

   ```Bash
   git clone https://github.com/12nastya08a-beep/ai-support-analyzer-.git

   cd ai-support-analyzer-
   ```

2. #### Install Dependencies (it is recommended to use a virtual environment):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. #### Configure Environment:
   Create a .env file and add your keys
   _(written above)_

## Execution:

To generate dialogues:

```bash
python generate.py #python33 generate.py for Mac
```

To analyze dialogues:

```bash
python analyze.py #python3 analyze.py for Mac
```

## 🛠 Troubleshooting

- **File not found**: Ensure you run `generate.py` before `analyze.py`.

- **API Errors**: Check your internet connection and verify that your API keys are valid and have active quotas.

- **Docker Permissions**: On Linux/Mac, ensure Docker has permission to write to the current directory (used for saving JSON results).

## 📋 Output Format (JSON)

The analyzer provides a detailed report for every chat:

```JSON
{
     "id": 1,
     "analysis": {
        "intent": "refund_request",
        "satisfaction": "unsatisfied",
        "quality_score": 2,
        "agent_mistakes": ["rude_tone", "no_resolution"]
    }
}
```

## 📚 Requirements

- **Python**: 3.10+

- **Key Libraries**: cohere, google-genai, groq, python-dotenv

- **Goal**: Compliance with AI Test Task requirements.
