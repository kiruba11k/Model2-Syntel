import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any
from io import BytesIO
from datetime import datetime
import time

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration & Environment Setup ---
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("ERROR: Both GROQ_API_KEY and TAVILY_API_KEY must be set in Streamlit secrets.")
    st.stop()

# --- LLM and Tool Initialization ---
try:
    llm_groq = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Required Fields Definition ---
REQUIRED_FIELDS = [
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex", "core_intent_analysis",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- Core Functions ---

def clean_and_format_url(url: str) -> str:
    if not url or url == "N/A":
        return "N/A"
    if url.startswith('//'):
        url = 'https:' + url
    elif not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.replace(' ', '').strip()

def generate_dynamic_search_queries(company_name: str, field_name: str) -> List[str]:
    field_queries = {
        "branch_network_count": [
            f'"{company_name}" branch network facilities locations count',
            f'"{company_name}" warehouse facility count pallet capacity 2024 2025'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion news 2024 2025 new facilities',
            f'"{company_name}" new warehouse construction Q3 Q4 2024 2025'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT initiatives 2024'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT infrastructure leadership 2024'
        ],
        "existing_network_vendors": [
            f'"{company_name}" network infrastructure vendors Cisco HPE Aruba',
            f'"{company_name}" IT technology stack network equipment'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN tender network upgrade 2024'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation robotics implementation'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption strategy AWS Azure Google',
            f'"{company_name}" using AWS services EC2 S3',
            f'"{company_name}" global capability center GCC setup location',
            f'"{company_name}" cloud migration update'
        
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility expansion'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget capex investment technology spending'
        ]
    }
    return field_queries.get(field_name, [f'"{company_name}" {field_name}'])

def dynamic_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    queries = generate_dynamic_search_queries(company_name, field_name)
    all_results = []
    
    for query in queries[:3]:
        try:
            time.sleep(0.5)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        if len(content) > 50:
                            all_results.append({
                                "title": result.get('title', ''),
                                "content": content[:800],
                                "url": result.get('url', ''),
                                "field": field_name,
                                "query": query
                            })
        except Exception:
            continue
    return all_results

def get_detailed_extraction_prompt(company_name: str, field_name: str, research_context: str) -> str:
    
    prompts = {
        "branch_network_count": f"""
        Extract ONLY the factual information about branch network or facilities count for {company_name} from the provided research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY numbers and facts explicitly mentioned in the research data
        - DO NOT invent, estimate, or calculate any numbers
        - If no specific count is found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED NETWORK COUNT:
        """,
        
        "expansion_news_12mo": f"""
        Extract ONLY the factual, recent expansion news for {company_name} from the last 12-18 months (or with dates 2024/2025/2026) mentioned in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific expansion announcements, such as **new facility acquisitions, greenfield projects, brownfield capacity additions, or major investment commitments.**
        - Include dates, locations, and financial figures (e.g., capex) ONLY if explicitly stated.
        - **CRITICAL:** DO NOT infer or assume any expansions. Ignore generic statements about 'growth' or 'market share increase.'
        - If no recent, concrete expansion news is found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED EXPANSION NEWS:
        """,
        
        "digital_transformation_initiatives": f"""
        Extract ONLY the specific digital transformation initiatives mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY initiatives explicitly mentioned
        - Include specific technologies ONLY if named
        - DO NOT infer or assume any initiatives
        - If no initiatives found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED DIGITAL INITIATIVES:
        """,
        
        "it_leadership_change": f"""
        Extract ONLY the specific IT leadership changes (CIO/CTO/Head Infra) mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY changes explicitly mentioned with names and dates
        - DO NOT infer leadership changes
        - If no changes found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED LEADERSHIP CHANGE:
        """,
        
        "existing_network_vendors": f"""
        Extract ONLY the specific network vendors and technology stack mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY vendors and technologies explicitly mentioned
        - DO NOT assume or infer vendors based on industry
        - If no vendors found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED VENDORS:
        """,
        
        "wifi_lan_tender_found": f"""
        Extract ONLY specific information about WiFi or LAN tenders/projects mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific tenders or upgrades explicitly mentioned
        - Include details ONLY if provided in research
        - DO NOT assume network upgrades
        - If no tenders found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED TENDER INFORMATION:
        """,
        
        "iot_automation_edge_integration": f"""
        Extract ONLY the specific IoT, automation, or edge integration projects mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific projects and technologies explicitly mentioned
        - DO NOT infer IoT usage based on industry
        - If no IoT projects found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
       "cloud_adoption_gcc_setup": f"""
        Extract ONLY the specific cloud adoption or GCC setup details **ATTRIBUTED DIRECTLY TO {company_name}** mentioned in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific cloud providers (AWS, Azure, GCP) or GCC plans **EXPLICITLY MENTIONED** in relation to {company_name}.
        - **CRITICAL:** IGNORE all generic descriptions of cloud services (like 'EC2 is a virtual server'), general cloud best practices, or generic third-party service offerings not tied to {company_name}.
        - DO NOT assume cloud adoption based on industry or partnerships.
        - If no specific, attributable cloud/GCC details found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED CLOUD/GCC DETAILS:
        """,
        
        "physical_infrastructure_signals": f"""
        Extract ONLY the specific physical infrastructure developments mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific construction projects explicitly mentioned
        - Include locations and details ONLY if provided
        - DO NOT infer infrastructure projects
        - If no developments found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        
        "it_infra_budget_capex": f"""
        Extract ONLY the specific IT infrastructure budget or capex information mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific budget figures explicitly mentioned
        - DO NOT estimate or calculate budgets
        - If no budget information found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED IT BUDGET INFORMATION:
        """
    }
    
    return prompts.get(field_name, f"""
    Extract ONLY the comprehensive, short, and correct information about {field_name} for {company_name}.
    
    RESEARCH DATA: {research_context}
    
    REQUIREMENTS: 
    - Output must be short, factual, and extremely concise. 
    - Extract ONLY information explicitly mentioned in the research data
    - DO NOT invent, estimate, or infer any information
    - Start directly with the extracted data.
    EXTRACTED INFORMATION:
    """)

def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    
    if not search_results:
        return "N/A"
    
    research_context = f"Research data for {company_name} - {field_name}:\n\n"
    for i, result in enumerate(search_results[:4]):
        research_context += f"SOURCE {i+1} - {result.get('title', 'No Title')}:\n"
        research_context += f"CONTENT: {result['content']}\n\n" 
    
    # MODIFICATION 1: Collect ALL unique URLs for maximum source inclusion
    unique_urls = list(set([clean_and_format_url(result['url']) for result in search_results if result.get('url')]))
    
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context)
    
    try:
        # Assuming llm_groq is initialized correctly
        response = llm_groq.invoke([
            SystemMessage(content=f"""You are an expert research analyst. Extract FACTUAL DATA ONLY from the provided research context for {company_name}.
            **CRITICAL INSTRUCTIONS:**
            - Extract ONLY information explicitly mentioned in the provided research data
            - DO NOT use any prior knowledge or make assumptions
            - DO NOT invent, estimate, or calculate any numbers
            - If information is not found in the research data, output "N/A"
            - Start your response directly with the factual data or "N/A"
            - Be concise and factual"""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Enhanced validation for hallucination prevention
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', 'information not available', ''] or 
            'not mentioned' in response.lower() or
            'no specific' in response.lower()):
            return "N/A"
        
        # Clean up any introductory phrases
        clean_up_phrases = [
            r'^\s*Based on the provided research data,.*:', 
            r'^\s*Here is the extracted information:',
            r'^\s*Extracted information:', 
            r'^\s*The.*for.*is:',
            r'^\s*\*\s*', 
            r'^\s*-\s*', 
            r'^\s*\d+\.\s*', 
        ]
        
        for phrase in clean_up_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE | re.DOTALL).strip()

        # Final cleaning
        response = re.sub(r'\n+', ' ', response).strip() 
        response = re.sub(r'\s+', ' ', response) 
        response = response.replace("**", "").replace("*", "") 
        
        # Add sources for traceability, ensuring all URLs are included in a single cell
        if unique_urls and response != "N/A":
            # MODIFICATION 2: Consolidate all unique URLs, separated by comma and space, with no character limit.
            source_text = f" [Sources: {', '.join(unique_urls)}]" 
            response += source_text
        
        # MODIFICATION 3: Remove the 500-character truncation (was: return response[:500])
        return response
        
    except Exception as e:
        # print(f"Error during LLM extraction for {field_name}: {e}") # Debugging line
        return "N/A"
        
def analyze_core_intent_article(article_url: str, company_name: str) -> str:
    """
    Analyze the core intent article provided by the user
    """
    if not article_url or article_url == "N/A":
        return "N/A - No article URL provided"
    
    try:
        # Use Tavily to get content from the specific URL
        search_results = search_tool.invoke({
            "query": f"site:{article_url}",
            "max_results": 1,
            "include_raw_content": True
        })
        
        article_content = ""
        if search_results and isinstance(search_results, list):
            for result in search_results:
                if isinstance(result, dict):
                    content = result.get('content', '') or result.get('snippet', '')
                    if content:
                        article_content = content[:1500]  # Limit content length
                        break
        
        if not article_content:
            # If Tavily can't fetch, try direct search about the article topic
            search_results = search_tool.invoke({
                "query": f'"{company_name}" recent news article',
                "max_results": 3
            })
            
            article_content = "Research context from recent news:\n"
            for i, result in enumerate(search_results[:2]):
                if isinstance(result, dict):
                    content = result.get('content', '') or result.get('snippet', '')
                    article_content += f"Source {i+1}: {content}\n\n"
        
        prompt = f"""
        Analyze the following article/content about {company_name} and extract the core business intent or strategic initiative:
        
        ARTICLE/CONTENT: {article_content}
        
        CORE INTENT ANALYSIS:
        - What is the main business objective or strategic move described?
        - What technology or infrastructure needs does this imply?
        - How does this relate to network/infrastructure requirements?
        
        Provide a concise analysis focusing on the strategic intent and implied technology needs.
        """
        
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business analyst. Analyze the core intent from the provided article/content."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        return f"{response} [Article: {article_url}]" if response else f"N/A - Could not analyze article [URL: {article_url}]"
        
    except Exception as e:
        return f"N/A - Error analyzing article: {str(e)} [URL: {article_url}]"

# --- DEDICATED RELEVANCE FUNCTION WITH CORE INTENT INTEGRATION ---

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str, core_intent_analysis: str) -> tuple:
    """
    Generates relevance analysis and intent score with core intent integration
    """
    
    # Prepare data context for the LLM (No change here)
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level", "core_intent_analysis"]:
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    data_context = "\n".join(context_lines)

    # Relevance Prompt (No change, as the prompt is already structured to ask for TSV)
    relevance_prompt = f"""
    You are evaluating whether the company below is relevant to Syntel's Go-To-Market for Wi-Fi & Network Integration.
    
    ---
    **SYNTEL GTM FOCUS**
    **Geography:** India
    **Industries:** Ports, Stadiums, Education, Manufacturing, Healthcare, Hospitality, **Warehouses**, BFSI, IT/ITES, GCCs
    **ICP:** 150-500+ employees, ₹100 Cr+ revenue
    
    **SCORING CRITERIA (CRITICAL):**
    - **HIGH Intent:** Company has **1 or more concrete expansion/capex plans** (new facility, greenfield) AND **1 or more technical needs** (IoT/Automation, Cloud/S4HANA migration, or stated Wi-Fi upgrade). **The Core Intent analysis is a major driver of infrastructure spending.**
    - **MEDIUM Intent:** Company is in a target industry with **general digital transformation initiatives** OR **leadership change** OR **unconfirmed expansion news/budget** OR **Cloud/GCC setup but no immediate infra signals.** The Core Intent suggests future, but not immediate, infrastructure need.
    - **LOW Intent:** Company is only a target industry with **no specific buying signals, no leadership change, and all key fields are "N/A."**
    
    **CORE INTENT ANALYSIS:**
    {core_intent_analysis}
    
    **Offerings:** Wi-Fi deployments, Network integration & managed services, Multi-vendor implementation (Altai + others), Full implementation support.
    ---
    
    **COMPANY DETAILS TO ANALYZE ({company_name}):**
    {data_context}
    
    **TASK:**
    1. Determine the Intent Score (**High / Medium / Low**) based **STRICTLY** on the SCORING CRITERIA and the data provided.
    2. Generate a **3-point summary** for "Why Relevant to Syntel."
    3. **INTEGRATE THE CORE INTENT** into your analysis where relevant
    4. Output the final result in the exact TSV format specified below.
    
    **OUTPUT FORMAT (TSV):**
    Company Name\tWhy Relevant to Syntel\tIntent (High / Medium / Low)
    
    **RULES:**
    - "Why Relevant" must contain the 3 bullet points, separated by a newline.
    - **CRITICAL:** Format the points as a numbered list: **1)**, **2)**, **3)**.
    - Do not include headers in the output
    - Ensure the points are short and professional
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a meticulous GTM analyst. Generate the output *only* in the requested TSV format, following all rules for specificity, the **1), 2), 3) numbered list format**, and **STRICTLY adhere to the provided SCORING CRITERIA**. Integrate core intent insights where relevant. **Crucially, ensure the third column contains ONLY 'High', 'Medium', or 'Low'.**"),
            HumanMessage(content=relevance_prompt)
        ]).content.strip()
        
        # Robust parsing of the TSV output
        parts = response.split('\t')
        
        # CRITICAL FIX 1: Normalize the raw TSV response to handle extra newlines/spaces
        if len(parts) >= 3:
            company = parts[0].strip()
            relevance_text_raw = parts[1].strip()
            score_raw = parts[2].strip()

            # --- Score Extraction (Fix for Intent Scoring column) ---
            # Search for the final 'High', 'Medium', or 'Low' and discard any extra text/bullets the LLM put there
            score_match = re.search(r'(High|Medium|Low)', score_raw, re.IGNORECASE)
            
            if score_match:
                score = score_match.group(0)
            else:
                score = "Medium" # Fallback if score is not found (though LLM is prompted strictly)

            # --- Relevance Text Extraction (Fix for Why Relevant column) ---
            
            # The LLM often repeats the bullets in the score column, so we rely on the first column for bullets.
            
            # 1. Aggressively split the text by common separators (newline, period, hyphen, existing numbers)
            raw_bullets = re.split(r'[\n\r]|(\d+\)|\d+\.)|[\.\-•]', relevance_text_raw)
            
            cleaned_bullets = []
            for bullet in raw_bullets:
                if bullet is None: continue 
                clean_bullet = bullet.strip()
                # Remove boilerplate introductions/cleanup markdown
                clean_bullet = re.sub(r'\*\*|\*|__|_|^\w+ Relevant to Syntel|^\s*The reasons are\s*:?', '', clean_bullet, flags=re.IGNORECASE)
                
                # Filter out short, empty, or boilerplate phrases
                if len(clean_bullet) > 10 and not clean_bullet.lower().startswith(company_name.lower()):
                    cleaned_bullets.append(clean_bullet.capitalize())

            # 2. Ensure exactly 3 bullets and apply the final 1), 2), 3) format
            final_bullets = cleaned_bullets[:3]
            
            # Add fallback bullets if we don't have enough specific ones
            while len(final_bullets) < 3:
                 final_bullets.append("Strategic relevance due to operating in a target industry.")
            
            # Apply the desired numbered format 1), 2), 3)
            formatted_bullets = "\n".join([f"{i+1}) {point}" for i, point in enumerate(final_bullets)])
            
            return formatted_bullets, score.strip()
        
        raise ValueError("LLM response not in expected TSV format.")

    except Exception:
        # Fallback block (score remains "Medium" in case of failure)
        fallback_bullets_list = []
        
        # 1. Core Intent Signal
        if "N/A" not in core_intent_analysis:
            fallback_bullets_list.append("Core intent analysis indicates strategic initiatives requiring network infrastructure support.")
        else:
            fallback_bullets_list.append("Company operates in target sectors requiring robust network infrastructure.")
        
        # 2. Expansion Signal
        if company_data.get('expansion_news_12mo') not in ["N/A", ""]:
             fallback_bullets_list.append(f"Recent expansion signals immediate need for network planning and deployment.")
        else:
             fallback_bullets_list.append("Operations in target industry sector align with Syntel's network GTM focus.")

        # 3. Technology Signal
        if company_data.get('iot_automation_edge_integration') not in ["N/A", ""]:
             fallback_bullets_list.append(f"IoT/Automation initiatives require high-performance Wi-Fi coverage across facilities.")
        else:
             fallback_bullets_list.append("Scale of operations indicates need for reliable, wide-area network coverage.")

        formatted_bullets = "\n".join([f"{i+1}) {point}" for i, point in enumerate(fallback_bullets_list)])
        return formatted_bullets, "Medium"# --- Main Research Function ---

def dynamic_research_company_intelligence(company_name: str, article_url: str = None) -> Dict[str, Any]:
    """Main function to conduct comprehensive company research"""
    
    company_data = {}
    all_search_results = []
    
    total_fields = len(REQUIRED_FIELDS) - 3  # Exclude core_intent_analysis and last two fields
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # First, analyze core intent article if provided
    if article_url:
        status_text.info(f" Analyzing core intent article for {company_name}...")
        core_intent = analyze_core_intent_article(article_url, company_name)
        company_data["core_intent_analysis"] = core_intent
        progress_bar.progress(10)
    else:
        company_data["core_intent_analysis"] = "N/A - No article URL provided"
    
    # Research other fields
    research_fields = [f for f in REQUIRED_FIELDS if f not in ["core_intent_analysis", "why_relevant_to_syntel_bullets", "intent_scoring_level"]]
    
    for i, field in enumerate(research_fields):
        progress = 10 + (i / len(research_fields)) * 70
        progress_bar.progress(int(progress))
        status_text.info(f" Researching **{field.replace('_', ' ').title()}** for {company_name}...")
        
        try:
            search_results = dynamic_search_for_field(company_name, field)
            all_search_results.extend(search_results)
            
            field_data = dynamic_extract_field_with_sources(company_name, field, search_results)
            company_data[field] = field_data
            
            time.sleep(1.0) 
            
        except Exception:
            company_data[field] = "N/A"
            continue
    
    status_text.info(" Conducting strategic relevance analysis...")
    progress_bar.progress(90)
    
    try:
        relevance_bullets, intent_score = syntel_relevance_analysis_v2(
            company_data, company_name, company_data.get("core_intent_analysis", "N/A")
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception:
        company_data["why_relevant_to_syntel_bullets"] = "• Analysis failure: Check core data points for LLM processing."
        company_data["intent_scoring_level"] = "Medium"
    
    progress_bar.progress(100)
    status_text.success(" Comprehensive research complete!")
    
    return company_data

def format_horizontal_display_with_sources(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean, professional HORIZONTAL display format"""
    
    mapping = {
        "Company Name": "company_name", 
        "Branch Network / Facilities Count": "branch_network_count", 
        "Expansion News (Last 12 Months)": "expansion_news_12mo",
        "Digital Transformation Initiatives": "digital_transformation_initiatives", 
        "IT Infrastructure Leadership Change": "it_leadership_change",
        "Existing Network Vendors / Tech Stack": "existing_network_vendors", 
        "Recent Wi-Fi Upgrade or LAN Tender Found": "wifi_lan_tender_found",
        "IoT / Automation / Edge Integration Mentioned": "iot_automation_edge_integration", 
        "Cloud Adoption / GCC Setup": "cloud_adoption_gcc_setup",
        "Physical Infrastructure Signals": "physical_infrastructure_signals", 
        "IT Infra Budget / Capex Allocation": "it_infra_budget_capex",
        "Core Intent Analysis": "core_intent_analysis",
        "Why Relevant to Syntel": "why_relevant_to_syntel_bullets",
        "Intent Scoring": "intent_scoring_level"
    }
    
    # Create a dictionary for the single row
    row_data = {}
    for display_col, data_field in mapping.items():
        if display_col == "Company Name":
            row_data[display_col] = company_input
        else:
            row_data[display_col] = data_dict.get(data_field, "N/A")
    
    # Create DataFrame with one row and field names as columns
    df = pd.DataFrame([row_data])
    return df

# --- Streamlit UI ---
if __name__ == "__main__":
    st.title(" Dynamic Company Intelligence Generator")
    st.sidebar.header("Configuration")
    
    company_name = st.sidebar.text_input(
        "Enter Company Name to Research:", 
        value="",
        key="company_input_box",
        placeholder="e.g., Snowman Logistics"
    )
    
    article_url = st.sidebar.text_input(
        "Core Intent Article URL:",
        value="",
        key="article_url_input",
        placeholder="Paste the article link that prompted this research"
    )
    
    # Initialize session state variables if they don't exist
    if 'company_name_to_search' not in st.session_state:
        st.session_state['company_name_to_search'] = None
    if 'company_data' not in st.session_state:
        st.session_state['company_data'] = None
    if 'article_url' not in st.session_state:
        st.session_state['article_url'] = None

    # This button explicitly triggers the search
    trigger_search = st.sidebar.button("Run Comprehensive Research")
    
    search_triggered = False
    
    if trigger_search and company_name:
        st.session_state['company_name_to_search'] = company_name
        st.session_state['article_url'] = article_url
        st.session_state['company_data'] = None # Clear old data
        search_triggered = True
    
    # Execution Block
    if st.session_state['company_name_to_search'] and st.session_state['company_data'] is None:
        
        with st.spinner(f"Starting comprehensive research for **{st.session_state['company_name_to_search']}**..."):
            company_data = dynamic_research_company_intelligence(
                st.session_state['company_name_to_search'], 
                st.session_state['article_url']
            )  
            st.session_state['company_data'] = company_data
            
        st.success(f"Research for **{st.session_state['company_name_to_search']}** completed successfully.")

    # Display Block
    if 'company_data' in st.session_state and st.session_state['company_data']:
        current_company = st.session_state['company_name_to_search']
        st.header(f" Extracted Intelligence: {current_company}")
        
        # Display the horizontal dataframe
        df_display = format_horizontal_display_with_sources(
            current_company, 
            st.session_state['company_data']
        )
        
        st.dataframe(df_display, use_container_width=True)

        # Download button functions
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Company_Intel')
            writer.close()
            processed_data = output.getvalue()
            return processed_data

        excel_data = to_excel(df_display)
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{current_company}_Intelligence_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
