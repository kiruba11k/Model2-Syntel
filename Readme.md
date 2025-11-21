Automated Company Research & Intent Analysis System

A Streamlit + Groq + Tavily powered research automation tool

1. Overview

This application automates company-level research and generates structured insights using:

Groq LLM (LLaMA 3.1 8B) for fast and accurate text extraction

Tavily Search API for web research

Streamlit for the user interface

The system extracts factual information, validates it, removes hallucinations, and produces:

Company intelligence

Buying signals

Strategic intent analysis

Syntel relevance scoring (High / Medium / Low)

Clean, traceable outputs with source links

This application is designed so any non-technical user can easily operate it, provided the required API keys are available.

2. Features
Automated Web Research

Dynamically generates queries based on the company name.

Gathers data from multiple reliable sources using Tavily.

LLM-Powered Information Extraction

Extracts factual content only.

Removes assumptions, hallucinations, and unsupported claims.

Core Intent Analysis

Evaluates company news or articles to determine strategic direction.

Connects this intent to infrastructure and technology requirements.

Syntel GTM Relevance Evaluation

Uses structured criteria to determine if the company is a good target.

Returns:

3-point summary of relevance

Intent Score: High / Medium / Low

TSV formatted output

Source Traceability

Every extracted field includes original URLs for verification.

Non-Technical Friendly UI

Streamlit-based interface

Simple input fields

One-click analysis

3. Technology Stack
Component	Purpose
Python 3.10+	Core language
Streamlit	Web-based UI
Groq LLM (LLaMA 3.1 8B)	High-speed text extraction
Tavily API	Search and research
LangChain	LLM orchestration
Pandas	Data handling
4. System Requirements

Python 3.10 or higher

Internet connection

Valid API keys:

GROQ_API_KEY

TAVILY_API_KEY

5. Installation (Step-by-Step)
Step 1: Clone the Repository
git clone <your-repo-url>
cd <your-repo-folder>

Step 2: Create a Virtual Environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Add API Keys to Streamlit Secrets

Create the file:

.streamlit/secrets.toml


Add the keys:

GROQ_API_KEY = "your_groq_key"
TAVILY_API_KEY = "your_tavily_key"

Step 5: Run the Application
streamlit run app.py


The application will open automatically in your browser.

6. Application Workflow (Non-Technical Explanation)

Enter the Company Name
The system uses this to generate research queries.

Optionally Provide a News Article URL
This helps extract the company's “core intent”.

Click “Run Analysis”
The system will:

Search the internet

Extract data

Validate facts

Remove hallucinations

Generate clean structured insights

View the Output
You will receive:

Branch network details

Expansion plans

Digital transformation initiatives

IT leadership updates

Infrastructure signals

Cloud/GCC setup

Tender information

IoT/automation activity

Full core intent analysis

Syntel relevance score

TSV-formatted result

You can download or copy the output for reporting or CRM updates.

7. Code Architecture
Key Components

Dynamic Search Generation
Produces context-aware search queries.

Search + Extraction Layer
Collects relevant text from multiple sources.

LLM Extraction Engine
Ensures:

Factual accuracy

No assumptions

Clean formatting

Core Intent Analyzer
Reads the article and extracts strategic direction.

Relevance Engine (Version 2)
Applies strict scoring rules to determine intent level.

8. Important Design Principles
Hallucination Prevention

Only extracts data explicitly found in search results

Rejects:

Assumptions

Generic statements

Inferred information

Source Traceability

Every extracted field includes:

[Sources: url1, url2, ...]

Strict Formatting Rules

TSV outputs

Numbered summaries

Clean text normalization

9. File Structure
project/
│
├── app.py                      # Main Streamlit application
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml            # API keys

10. Troubleshooting
Missing API Keys

If the app shows:

ERROR: Both GROQ_API_KEY and TAVILY_API_KEY must be set


Ensure secrets are added correctly.

LLM Initialization Error

Check:

Internet connection

Valid Groq API key

No firewall restriction

No Search Results

Some companies may have limited online visibility.
You may try:

Adding “Pvt Ltd” to the name

Adding industry keywords

11. Security Notes

API keys are stored securely using Streamlit Secrets.

No data is stored permanently.

Sensitive fields should never be logged or shared.

12. Intended Users

This tool is designed for:

Sales teams

Pre-sales engineers

GTM strategists

Business analysts

Research teams

Non-technical staff needing automated intelligence
