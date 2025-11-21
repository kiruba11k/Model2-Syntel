# **Automated Company Research & Intent Analysis System**

A fully automated research and intent-analysis workflow powered by **Streamlit**, **Groq**, and **Tavily**.
Designed for both technical and non-technical users.
---
Live Link:[https://lsm2sy.streamlit.app/]


## **1. Overview**

This application automates company-level research and produces structured, factual insights using:

* **Groq LLM (LLaMA 3.1 8B)** for high-speed and accurate text extraction
* **Tavily Search API** for real-time web research
* **Streamlit** for an interactive and user-friendly interface

The system is capable of generating:

* Company intelligence
* Buying signals
* Strategic intent analysis
* **Syntel relevance scoring** (High / Medium / Low)

All outputs include traceable sources, validated details, and hallucination-free insights.
Non-technical users can operate the tool with ease as long as the required API keys are provided.

---

## **2. Features**

### **Automated Web Research**

* Dynamically generates keyword-rich search queries.
* Collects information from multiple trusted online sources via Tavily.

### **LLM-Powered Information Extraction**

* Extracts only factual, source-verified content.
* Removes assumptions, hallucinated insights, and unsupported claims.

### **Core Intent Analysis**

* Evaluates company news or articles to determine strategic direction.
* Maps intent to technology, infrastructure, and expansion signals.

### **Syntel GTM Relevance Evaluation**

Outputs include:

* Three-point relevance summary
* **Intent Score:** High / Medium / Low
* TSV-formatted structured data

### **Source Traceability**

* Every field includes links to the original sources for verification.

### **Non-Technical Friendly UI**

* Clean Streamlit interface
* Simple input fields
* One-click analysis workflow

---

## **3. Technology Stack**

| Component               | Purpose                           |
| ----------------------- | --------------------------------- |
| Python 3.10+            | Core development language         |
| Streamlit               | Web-based UI                      |
| Groq LLM (LLaMA 3.1 8B) | Fast and accurate text extraction |
| Tavily API              | Web search and data gathering     |
| LangChain               | LLM orchestration                 |
| Pandas                  | Data handling and formatting      |

---

## **4. System Requirements**

* Python **3.10 or higher**
* Stable internet connection
* Required API keys:

  * `GROQ_API_KEY`
  * `TAVILY_API_KEY`

---

## **5. Installation (Step-by-Step)**

### **Step 1: Clone the Repository**

```bash
git clone "https://github.com/kiruba11k/Model2-Syntel.git"
cd Model2-Syntel
```

### **Step 2: Create a Virtual Environment**

```bash
python -m venv venv
```

**Activate the environment**

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Add API Keys to Streamlit Secrets**

Create file:

```
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_key"
TAVILY_API_KEY = "your_tavily_key"
```

### **Step 5: Run the Application**

```bash
streamlit run app.py
```

The app will open in your default browser.

---

## **6. Application Workflow (Non-Technical Explanation)**

1. **Enter the Company Name**
   Used to generate intelligent research queries.

2. **Optionally Add a News Article URL**
   Enables deeper strategic intent analysis.

3. **Click “Run Analysis”**
   The system will:

   * Search the internet
   * Extract and validate data
   * Remove hallucinations
   * Generate structured insights

4. **View Results**
   You will receive detailed insights, including:

   * Branch network
   * Expansion and hiring signals
   * Digital transformation initiatives
   * IT leadership updates
   * Cloud / Infrastructure / GCC plans
   * IoT / Automation activity
   * Tender and procurement details
   * Full strategic intent analysis
   * **Syntel relevance score**
   * TSV-ready structured output

You can copy or download the results for CRM updates, reports, or GTM planning.

---

## **7. Code Architecture**

### **Key Components**

#### **Dynamic Search Generator**

Creates optimized search queries based on the company context.

#### **Search + Extraction Layer**

Fetches and aggregates relevant text from multiple sources.

#### **LLM Extraction Engine**

Ensures:

* High factual accuracy
* Clean normalized outputs
* Zero assumptions

#### **Core Intent Analyzer**

Interprets news articles to identify strategic direction and upcoming initiatives.

#### **Relevance Engine (v2)**

Scores the company based on strict GTM rules.

---

## **8. Important Design Principles**

### **Hallucination Prevention**

The system extracts only information explicitly present in search results.
It rejects:

* Assumptions
* Generic statements
* Inferred or speculative insights

### **Source Traceability**

Each extracted field contains:

```
[Sources: url1, url2, ...]
```

### **Strict Formatting**

* TSV structured output
* Numbered summaries
* Clean text normalization

---

## **9. File Structure**

```
project/
│
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml        # API keys
```

---

## **10. Troubleshooting**

### **Missing API Keys**

If you see:

```
ERROR: Both GROQ_API_KEY and TAVILY_API_KEY must be set
```

Ensure the keys are correctly placed in `secrets.toml`.

### **LLM Initialization Errors**

Check:

* Internet connectivity
* Valid Groq API key
* Firewall restrictions

### **No Search Results**

Try:

* Adding “Pvt Ltd” to the name
* Adding industry keywords

---

## **11. Security Notes**

* API keys are securely managed via Streamlit Secrets.
* No data is stored permanently.
* Sensitive data is not logged or shared.

---

## **12. Intended Users**

This tool is ideal for:

* Sales teams
* Pre-sales engineers
* GTM strategists
* Business analysts
* Research teams
* Non-technical users needing automated intelligence

---
