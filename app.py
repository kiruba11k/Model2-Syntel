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
            f'"{company_name}" IoT automation robotics implementation 2024',
            f'"{company_name}" edge computing automation technology'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption AWS Azure GCC setup 2024',
            f'"{company_name}" global capability center cloud migration'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility expansion 2024',
            f'"{company_name}" warehouse construction infrastructure development'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget capex investment technology spending 2024',
            f'"{company_name}" capital expenditure IT infrastructure'
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
        - Include relevant source URLs
        - If no specific count is found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED NETWORK COUNT:
        """,
        
        "expansion_news_12mo": f"""
        Extract ONLY the factual expansion news for {company_name} from the last 12 months mentioned in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY specific expansion announcements mentioned in the research
        - Include dates and locations ONLY if explicitly stated
        - Include relevant source URLs
        - DO NOT infer or assume any expansions
        - If no expansion news found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED EXPANSION NEWS:
        """,
        
        "digital_transformation_initiatives": f"""
        Extract ONLY the specific digital transformation initiatives mentioned for {company_name} in the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ONLY initiatives explicitly mentioned
        - Include specific technologies ONLY if named
        - Include relevant source URLs
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
        - Include relevant source URLs
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
        - Include relevant source URLs
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
        - Include relevant source URLs
        - DO NOT assume network upgrades
        - If no tenders found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED TENDER INFORMATION:
        """,
        
        "iot_automation_edge_integration": f"""
        Extract SPECIFIC IoT, Automation, and Edge computing implementations for {company_name} from the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract EXACT technologies, projects, and use cases mentioned
        - Include specific vendor names if mentioned (e.g., "Siemens IoT", "Rockwell Automation")
        - **YOU MUST include the relevant source URL at the end**
        - If no specific IoT/Automation found, state "N/A"
        - DO NOT cut any words in the middle
        - Start directly with the extracted data
        
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
        
        "cloud_adoption_gcc_setup": f"""
        Extract SPECIFIC Cloud Adoption or GCC Setup information for {company_name} from the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract EXACT cloud providers, migration projects, or GCC plans mentioned
        - Include specific details like "moving to AWS", "Azure migration", "GCC setup in Bangalore"
        - **YOU MUST include the relevant source URL at the end**
        - If no cloud/GCC information found, state "N/A"
        - DO NOT cut any words in the middle
        - Start directly with the extracted data
        
        EXTRACTED CLOUD/GCC DETAILS:
        """,
        
        "physical_infrastructure_signals": f"""
        Extract COMPLETE physical infrastructure developments for {company_name} from the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract ALL details about new construction, expansions, facilities
        - Include locations, capacities, timelines mentioned
        - **YOU MUST include the relevant source URL at the end**
        - **DO NOT truncate or cut words - provide complete information**
        - If no infrastructure developments found, state "N/A"
        - Start directly with the extracted data
        
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        
        "it_infra_budget_capex": f"""
        Extract SPECIFIC IT infrastructure budget or capex information for {company_name} from the research data.
        
        RESEARCH DATA: {research_context}
        
        REQUIREMENTS:
        - Extract EXACT budget figures, percentages, or investment plans mentioned
        - Include timeframes and focus areas if specified
        - **YOU MUST include the relevant source URL at the end**
        - If no specific budget information found, state "N/A - No specific IT budget figures found in public sources"
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
    - **YOU MUST include the relevant source URL at the end**
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
    
    unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:3]
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context)
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content=f"""You are an expert research analyst. Extract FACTUAL DATA ONLY from the provided research context for {company_name}.
            **CRITICAL INSTRUCTIONS:**
            - Extract ONLY information explicitly mentioned in the provided research data
            - DO NOT use any prior knowledge or make assumptions
            - DO NOT invent, estimate, or calculate any numbers
            - **YOU MUST INCLUDE RELEVANT SOURCE URLS IN YOUR RESPONSE**
            - If information is not found in the research data, output "N/A"
            - Start your response directly with the factual data or "N/A"
            - Be concise and factual"""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Enhanced validation for hallucination prevention
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', 'information not available', ''] or 
            len(response) < 5 or
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
        
        # For fields that require source URLs but might not have them embedded, add them
        needs_url = field_name in ["iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
                                 "physical_infrastructure_signals", "it_infra_budget_capex"]
        
        if needs_url and unique_urls and response != "N/A" and "http" not in response.lower():
            source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
        
        return response[:800]  # Increased limit for physical infrastructure field
            
    except Exception as e:
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

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str, core_intent_analysis: str) -> tuple:
    """
    Generates relevance analysis and intent score with core intent integration
    """
    
    # Prepare data context for the LLM
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level", "core_intent_analysis"]:
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    data_context = "\n".join(context_lines)

    # Enhanced prompt with STRONG core intent integration
    relevance_prompt = f"""
    You are evaluating whether the company below is relevant to Syntel's Go-To-Market for Wi-Fi & Network Integration.
    
    ---
    **SYNTEL GTM FOCUS**
    **Geography:** India
    **Industries:** Ports, Stadiums, Education, Manufacturing, Healthcare, Hospitality, **Warehouses**, BFSI, IT/ITES, GCCs
    **ICP:** 150-500+ employees, ₹100 Cr+ revenue
    **Key Buying Signals (HIGH INTENT):**
    - Opening or expanding factories, offices, campuses, **warehouses** (especially 2024/2025/2026 dates).
    - Digital transformation / cloud modernization
    - Wi-Fi or LAN upgrade signals
    - **IoT/automation (AGVs, robots, sensors)**
    - Leadership changes (CIO/CTO/Infra head)
    - Large physical spaces needing wireless coverage
    
    **Offerings:** Wi-Fi deployments, Network integration & managed services, Multi-vendor implementation (Altai + others), Full implementation support.
    ---
    
    **COMPANY DETAILS TO ANALYZE ({company_name}):**
    {data_context}
    
    **CORE INTENT ANALYSIS (CRITICAL - MUST INTEGRATE):**
    {core_intent_analysis}
    
    **TASK:**
    1. **MUST INTEGRATE CORE INTENT ANALYSIS** into your relevance assessment
    2. Determine the Intent Score (High / Medium / Low) based on Buying Signals AND the Core Intent analysis
    3. Generate a 3-bullet point summary for "Why Relevant to Syntel." 
    4. **AT LEAST ONE BULLET POINT MUST DIRECTLY REFERENCE THE CORE INTENT ANALYSIS**
    5. Connect the company's strategic moves from the core intent to Syntel's network/Wi-Fi offerings
    6. Output the final result in the exact TSV format specified below.
    
    **OUTPUT FORMAT (TSV):**
    Company Name\tWhy Relevant to Syntel\tIntent (High / Medium / Low)
    
    **RULES:**
    - "Why Relevant" must contain the 3 bullet points, separated by a newline
    - **FIRST BULLET: Must reference core intent and connect to network needs**
    - Second/Third Bullets: Can use other buying signals from company details
    - Do not include headers in the output
    - Ensure the bullets are short and professional
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="""You are a meticulous GTM analyst. Generate the output *only* in the requested TSV format.
            **CRITICAL: You MUST integrate the Core Intent Analysis into your relevance assessment and ensure at least one bullet point directly references it.**"""),
            HumanMessage(content=relevance_prompt)
        ]).content.strip()
        
        # Robust parsing of the TSV output
        parts = response.split('\t')
        if len(parts) == 3:
            company, relevance_text, score = parts
            
            bullets = relevance_text.split('\n')
            
            cleaned_bullets = []
            for bullet in bullets:
                clean_bullet = re.sub(r'^[•\-\s]*', '• ', bullet.strip())
                clean_bullet = re.sub(r'\*\*|\*|__|_', '', clean_bullet)
                if len(clean_bullet) > 5:
                    cleaned_bullets.append(clean_bullet)

            # Ensure we have exactly 3 bullets
            while len(cleaned_bullets) < 3:
                if len(cleaned_bullets) == 0:
                    cleaned_bullets.append("• Core strategic initiatives require robust network infrastructure support.")
                elif len(cleaned_bullets) == 1:
                    cleaned_bullets.append("• Operations in target sectors align with Syntel's network GTM focus.")
                else:
                    cleaned_bullets.append("• Company scale indicates budget availability for IT infrastructure projects.")
            
            formatted_bullets = "\n".join(cleaned_bullets[:3])
            return formatted_bullets, score.strip()
        
        raise ValueError("LLM response not in expected TSV format.")

    except Exception as e:
        # Enhanced fallback with core intent consideration
        fallback_bullets_list = []
        
        # 1. Core Intent Signal (FIRST BULLET MUST REFERENCE CORE INTENT)
        if "N/A" not in core_intent_analysis and "No article" not in core_intent_analysis:
            # Extract key phrases from core intent for the bullet
            if "expansion" in core_intent_analysis.lower():
                fallback_bullets_list.append("• Core strategic expansion plans create immediate need for network infrastructure deployment.")
            elif "digital" in core_intent_analysis.lower() or "transformation" in core_intent_analysis.lower():
                fallback_bullets_list.append("• Digital transformation initiatives from core intent require modern network solutions.")
            elif "iot" in core_intent_analysis.lower() or "automation" in core_intent_analysis.lower():
                fallback_bullets_list.append("• Automation/IoT focus from core intent demands high-performance wireless coverage.")
            else:
                fallback_bullets_list.append("• Strategic initiatives identified in core intent analysis align with Syntel's network offerings.")
        else:
            fallback_bullets_list.append("• Core business strategy indicates need for reliable network infrastructure across operations.")
        
        # 2. Expansion Signal
        if company_data.get('expansion_news_12mo') not in ["N/A", ""]:
             fallback_bullets_list.append("• Recent expansion announcements signal immediate need for network planning and deployment.")
        elif company_data.get('branch_network_count') not in ["N/A", ""]:
             fallback_bullets_list.append("• Extensive facility network requires comprehensive wireless coverage solutions.")
        else:
             fallback_bullets_list.append("• Operations in logistics/warehousing sector align with Syntel's network GTM focus.")

        # 3. Technology Signal
        if company_data.get('iot_automation_edge_integration') not in ["N/A", ""]:
             fallback_bullets_list.append("• IoT/Automation initiatives require high-performance Wi-Fi coverage across large facilities.")
        elif company_data.get('digital_transformation_initiatives') not in ["N/A", ""]:
             fallback_bullets_list.append("• Digital transformation projects indicate budget allocation for IT infrastructure upgrades.")
        else:
             fallback_bullets_list.append("• Scale of operations indicates need for reliable, wide-area network coverage.")

        return "\n".join(fallback_bullets_list), "Medium"

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
        "IoT / Automation / Edge Integration": "iot_automation_edge_integration", 
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

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Company_Intel')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def create_tsv_data(df: pd.DataFrame) -> str:
    """Convert DataFrame to TSV string"""
    return df.to_csv(sep='\t', index=False)

# --- Streamlit UI ---
if __name__ == "__main__":
    st.title("🚀 Dynamic Company Intelligence Generator")
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
        st.header(f"📊 Extracted Intelligence: {current_company}")
        
        # Display the horizontal dataframe
        df_display = format_horizontal_display_with_sources(
            current_company, 
            st.session_state['company_data']
        )
        
        # Responsive dataframe with adjustable columns
        st.dataframe(df_display, use_container_width=True, height=400)

        # Download buttons in columns
        st.subheader("📥 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Excel download
            excel_data = to_excel(df_display)
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"{current_company}_Intelligence_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # TSV download
            tsv_data = create_tsv_data(df_display)
            st.download_button(
                label="📄 Download TSV",
                data=tsv_data,
                file_name=f"{current_company}_Intelligence_{datetime.now().strftime('%Y%m%d')}.tsv",
                mime="text/tab-separated-values",
                use_container_width=True
            )
        
        with col3:
            # Copy to clipboard (using st.code for easy copying)
            st.download_button(
                label="📋 Copy TSV Data",
                data=create_tsv_data(df_display),
                file_name=f"{current_company}_data.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Display TSV data for easy copying
        st.subheader("📋 TSV Data (Select and Copy)")
        tsv_display = create_tsv_data(df_display)
        st.code(tsv_display, language="text")
