# smart-expense-categorizer
A smart transaction analyzer built with Python and Streamlit to automatically sort and visualize spending.

# 💰 Smart Expense Categorizer (Streamlit + LLM)

This project is a Streamlit application designed to automate the manual process of categorizing personal financial transactions. It uses a **dual-layer approach**—combining fast, reliable **rule-based pattern matching** with **AI (LLM)** for context-aware classification—to deliver accurate and insightful spending summaries.

## ✨ Features

* **Multi-File Support**: Accepts transaction data from **CSV**, **Excel**, **PDF**, and **TXT** files.
* **Intelligent Column Detection**: Automatically detects 'Description' and 'Amount' columns using common keywords.
* **Dual-Layer Categorization**:
    1.  **Rule-based**: Uses regular expressions (`categories.py`) for high-speed, predictable classification.
    2.  **AI Fallback**: Transactions with low rule confidence are sent to an LLM via **OpenRouter** for context-aware categorization.
* **Customizable LLM**: Utilizes the **OpenRouter API** to allow easy swapping of models for optimization (cost or performance).
* **Visualization Suite**: Generates Bar Charts, Pie Charts, and Comparative Plots to summarize spending.

---

## 🚀 Setup and Installation

Follow these steps to get the app running locally on your machine.

### Prerequisites

* Python 3.8+
* All project files (app.py, categories.py, etc.) saved in one local folder.

### Step 1: Install Dependencies (`requirements.txt`)

The requirements.txt file lists all necessary Python libraries.

1.  **Create and Activate Virtual Environment (Recommended):** This keeps your project clean.
    
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\activate
 
2.  **Install Packages:** Run this command from your project folder:
       
      pip install -r requirements.txt
   

3. Configure API Key (pass.env)

You need an API key from **OpenRouter** for the AI features.

1.  Rename the Template File: In your project folder, rename the template file pass.env.example to pass.env.

2.  Insert Your Key: Open the pass.env file and paste your secret key:
    pass.env
    OPENROUTER_API_KEY="sk-or-v1-YOUR_SECRET_KEY_GOES_HERE"

    important note > API Key Check: To confirm your key is working and your account has credits, run the app and check the terminal. If you see a 402 Payment Required error, you must add credits to your OpenRouter account or switch to a free model (e.g., deepseek/deepseek-r1:free) in app.py and categories.py.

4: Run the Streamlit App

Start the application from your terminal by typing 'streamlit run app.py'
