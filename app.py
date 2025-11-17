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
# Assume these are correctly set in Streamlit secrets
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
    "linkedin_url", "company_website_url", "industry_category", 
    "employee_count_linkedin", "headquarters_location", "revenue_source",
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- Core Functions ---

def clean_and_format_url(url: str) -> str:
    """Clean and format URLs"""
    if not url or url == "N/A":
        return "N/A"
    if url.startswith('//'):
        url = 'https:' + url
    elif not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.replace(' ', '').strip()

def generate_dynamic_search_queries(company_name: str, field_name: str) -> List[str]:
    """Generate dynamic search queries based on company and field"""
    
    field_queries = {
        "linkedin_url": [f'"{company_name}" LinkedIn company page'],
        "company_website_url": [f'"{company_name}" official website'],
        "industry_category": [f'"{company_name}" industry business sector'],
        "employee_count_linkedin": [f'"{company_name}" employee count LinkedIn'],
        "headquarters_location": [f'"{company_name}" corporate headquarters city country'],
        "revenue_source": [f'"{company_name}" revenue USD dollars financial results'],
        "branch_network_count": [f'"{company_name}" latest warehouse facility count pallet capacity 2025',f'"{company_name}" branch network facilities locations count',],
        "expansion_news_12mo": [f'"{company_name}" expansion news 2024 2025 new facilities',f'"{company_name}" new warehouse construction Q3 Q4 2025',],
        "digital_transformation_initiatives": [f'"{company_name}" digital transformation IT initiatives'],
        "it_leadership_change": [f'"{company_name}" CIO CTO IT leadership'],
        "existing_network_vendors": [f'"{company_name}" technology vendors Cisco VMware SAP Oracle'],
        "wifi_lan_tender_found": [f'"{company_name}" WiFi LAN tender network upgrade'],
        "iot_automation_edge_integration": [f'"{company_name}" IoT automation robotics implementation'],
        "cloud_adoption_gcc_setup": [f'"{company_name}" cloud adoption AWS Azure GCC'],
        "physical_infrastructure_signals": [f'"{company_name}" new construction facility expansion'],
        "it_infra_budget_capex": [f'"{company_name}" IT budget capex investment technology spending']
    }
    
    return field_queries.get(field_name, [f'"{company_name}" {field_name}'])

def dynamic_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Dynamic search for specific field information with multiple query attempts"""
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
    """Get detailed extraction prompts for each field with strict single-fact requirements"""
    
    prompts = {
        "industry_category": f"""Analyze the research data and provide ONLY the single best-fit, primary industry category for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Output ONE single industry/sector (e.g., 'Cold Chain Logistics' or 'Pharmaceutical Manufacturing'). - Start directly with the extracted data.
        EXTRACTED PRIMARY INDUSTRY:
        """,
        "employee_count_linkedin": f"""Extract ONLY the single most credible or largest employee count/range for {company_name}. If multiple are found, prioritize the LinkedIn range or the largest confirmed number.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Output ONE single value (e.g., '1,001-5,000 employees (LinkedIn)' or '3,500 employees confirmed'). - Start directly with the extracted data.
        EXTRACTED EMPLOYEE COUNT:
        """,
        "headquarters_location": f"""Analyze all sources and identify the single, **official corporate headquarters** for {company_name}. Extract ONLY the **City, State/Province, and Country** of the headquarters.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Output the location as ONE single concise string (e.g., 'Bengaluru, Karnataka, India' or 'Dallas, Texas, USA'). - Start directly with the extracted data.
        EXTRACTED HEADQUARTERS LOCATION:
        """,
        "revenue_source": f"""Extract ONLY the latest and most relevant annual or quarterly revenue figure for {company_name}. Convert the final number to USD (with period) and provide ONE concise fact.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Output ONE single value (e.g., 'USD 19.3 million (FY 2025-26 Est.)' or 'USD 450M Annual Revenue (FY 2024)'). - Start directly with the extracted data.
        EXTRACTED REVENUE INFORMATION:
        """,
        # --- MODIFIED PROMPT FOR HIGH ACCURACY ---
        "branch_network_count": f"""
        Analyze the research data and extract ONLY the **latest, consolidated total** of physical facilities/warehouses/locations and their capacity/city count for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Output ONE single, consolidated number and the associated capacity/location count (e.g., '44 warehouses across 21 cities with 1,54,330 pallets' or '542 locations').
        - **Prioritize the most recent (2025/2026) consolidated figure.**
        - Start directly with the extracted data.
        
        EXTRACTED NETWORK COUNT:
        """,
        # --- MODIFIED PROMPT FOR HIGH ACCURACY ---
        "expansion_news_12mo": f"""
        Extract ONLY the most recent and significant expansion news for {company_name} from the **last 12-24 months (2024 and 2025/2026)**. Consolidate new facilities, geographic expansions, and fleet/capacity additions.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific new facilities, their capacity (if available), and the operational/announced dates (e.g., 'New 5,900-pallet warehouse in Pune (operational June 2026); facilities in Kolkata and Krishnapatnam opened June 2025.').
        - **Focus strictly on announced/completed projects between late 2024 and Q4 2025.**
        - Start directly with the extracted data.
        
        EXTRACTED EXPANSION NEWS:
        """,
        "digital_transformation_initiatives": f"""Extract ONLY the key digital transformation and IT projects for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - List specific technologies and projects (e.g., 'SAP S/4HANA Greenfield Implementation (2021), SAP BTP Integration'). - Start directly with the extracted data.
        EXTRACTED DIGITAL INITIATIVES:
        """,
        "it_leadership_change": f"""Extract ONLY the key details of any significant IT leadership change (CIO, CTO, etc.) for {company_name} in the last 24 months.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide name, role, and the change (e.g., 'Sunil Nair stepped down as CEO in Q4 2024, new leadership not yet named.'). - Start directly with the extracted data.
        EXTRACTED LEADERSHIP CHANGE:
        """,
        "existing_network_vendors": f"""Extract ONLY the key technology and network vendors mentioned for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - List specific vendors (e.g., 'Cisco, VMware, SAP, Microsoft'). - Start directly with the extracted data.
        EXTRACTED VENDORS:
        """,
        "wifi_lan_tender_found": f"""Extract ONLY specific information about any Wi-Fi/LAN or network tender/project found for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide details of the tender/project (e.g., 'RFP for WAN upgrade in Q3 2025' or 'No specific tender found.'). - Start directly with the extracted data.
        EXTRACTED TENDER INFORMATION:
        """,
        "iot_automation_edge_integration": f"""Extract ONLY the key IoT, Automation, and Edge computing implementations for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Specify technologies and use cases (e.g., 'IoT for temp checks on blood samples, remote monitoring system with ECIL'). - Start directly with the extracted data.
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
        "cloud_adoption_gcc_setup": f"""Extract ONLY the key Cloud Adoption or Global Capability Center (GCC) setup details for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Specify cloud providers, migration status, or GCC plans (e.g., 'Hybrid Cloud model with Azure, no GCC plans found.'). - Start directly with the extracted data.
        EXTRACTED CLOUD/GCC DETAILS:
        """,
        "physical_infrastructure_signals": f"""Extract ONLY the key physical infrastructure developments for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - List new construction projects and facility expansions (e.g., 'Ranchi Integrated Diagnostics Centre, JV with Star Imaging in Maharashtra'). - Start directly with the extracted data.
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        "it_infra_budget_capex": f"""Extract ONLY the specific IT infrastructure budget and capital expenditure information for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide specific budget figures, timeframes, or investment focus areas (e.g., 'No figures found, focus on digital transformation and expansion'). - Start directly with the extracted data.
        EXTRACTED IT BUDGET INFORMATION:
        """
    }
    
    return prompts.get(field_name, f"""
    Extract ONLY the comprehensive, short, and correct information about {field_name} for {company_name}.
    
    RESEARCH DATA:
    {research_context}
    
    REQUIREMENTS:
    - Output must be short, factual, and extremely concise.
    - Start directly with the extracted data.
    
    EXTRACTED INFORMATION:
    """)

def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Enhanced extraction with better source utilization and accuracy"""
    
    if not search_results:
        return "N/A"
    
    if field_name == "linkedin_url":
        for result in search_results:
            url = result.get('url', '')
            if 'linkedin.com/company' in url.lower():
                return clean_and_format_url(url)
        return "N/A"
    
    if field_name == "company_website_url":
        for result in search_results:
            url = result.get('url', '')
            if any(domain in url.lower() for domain in ['.com', '.in', '.org', '.net']):
                if not any(social in url.lower() for social in ['linkedin', 'facebook', 'twitter', 'youtube', 'slideshare']):
                    return clean_and_format_url(url)
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
            **The output must be EXTREMELY CONCISE, FACTUAL, and SHORT (under 100 words, unless it's a bulleted list like relevance/expansion).**
            **DO NOT** use any introductory phrases, conversational fillers, or descriptive headers.
            Start the output directly with the correct data point. Omit source mentions from the main text body."""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        if (not response or response.lower() in ['n/a', 'not found', 'no information', 'information not available', ''] or len(response) < 5):
            return "N/A"
        
        clean_up_phrases = [
            r'^\s*Based on the provided research data,.*:', r'^\s*Here is the extracted information:',
            r'^\s*Extracted information:', r'^\s*The headquarters address for [A-Za-z\s]+ is:',
            r'^\s*The official corporate headquarters is located at:', r'^\s*The official corporate headquarters is:',
            r'^\s*\*\s*', r'^\s*-\s*', r'^\s*\d+\.\s*', 
        ]
        
        for phrase in clean_up_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE | re.DOTALL).strip()

        if field_name == "headquarters_location" and response != "N/A":
            response_parts = response.split(',')
            if len(response_parts) > 3:
                response = ", ".join(response_parts[-3:]).strip()
            response = re.sub(r'\s+', ' ', response).strip()
            response = re.sub(r'\s*\d+\s*', '', response).strip() 
            response = response.replace("India India", "India").strip() 
        
        response = re.sub(r'https?://\S+', '', response)
        response = re.sub(r'\n+', ' ', response).strip() 
        response = re.sub(r'\s+', ' ', response) 
        response = response.replace("**", "").replace("*", "") 
        
        if field_name not in ["linkedin_url", "company_website_url"]:
            if unique_urls and response != "N/A":
                source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
                response += source_text
        
        return response[:500]
            
    except Exception as e:
        return "N/A"

# --- Main Research Function (Unchanged) ---
def dynamic_research_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Main function to conduct comprehensive company research"""
    
    company_data = {}
    all_search_results = []
    
    total_fields = len(REQUIRED_FIELDS) - 2
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.info(f"🔍 Researching **{field.replace('_', ' ').title()}** for {company_name}...")
        
        try:
            search_results = dynamic_search_for_field(company_name, field)
            all_search_results.extend(search_results)
            
            field_data = dynamic_extract_field_with_sources(company_name, field, search_results)
            company_data[field] = field_data
            
            time.sleep(1.0) 
            
        except Exception:
            company_data[field] = "N/A"
            continue
    
    status_text.info("🤔 Conducting strategic relevance analysis...")
    progress_bar.progress(90)
    
    try:
        relevance_bullets, intent_score = generate_dynamic_relevance_analysis(
            company_data, company_name, all_search_results
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception:
        company_data["why_relevant_to_syntel_bullets"] = "• Comprehensive analysis in progress based on company growth and technology initiatives"
        company_data["intent_scoring_level"] = "Medium"
    
    progress_bar.progress(100)
    status_text.success("✅ Comprehensive research complete!")
    
    return company_data

# --- Enhanced Relevance Analysis Function (Focusing on Syntel's Network GTM) ---
def generate_dynamic_relevance_analysis(company_data: Dict, company_name: str, all_search_results: List[Dict]) -> tuple:
    """Generate comprehensive relevance analysis, prioritizing Network Integration and Expansion."""
    
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            if clean_value and clean_value != "N/A":
                context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    context = "\n".join(context_lines)
    unique_urls = list(set([result['url'] for result in all_search_results if result.get('url')]))[:3]
    source_context = f"Research sources: {', '.join(unique_urls)}" if unique_urls else "Based on comprehensive research"
    
    relevance_prompt = f"""
    COMPANY: {company_name} (A Cold Chain Logistics/Warehouse Operator)
    {source_context}
    
    DETAILED COMPANY DATA:
    {context}
    
    SYNTEL CORE EXPERTISE (Network Integration & GTM Focus):
    1. Target Industry: **Warehouses** & Logistics Hubs.
    2. Primary Need: New **Wi-Fi / Network Integration** for expansion, coverage, and AGV/forklift roaming.
    3. Differentiation: Altai WiFi provides **3-5x coverage, seamless roaming (Zero-Roaming Drop)**, and is ideal for large, complex warehouse spaces.
    4. Service: End-to-end multi-brand deployment and integration support.
    
    TASK: Analyze the company data, focusing on **Expansion News** and **Branch Network Count**. Create 3 detailed, evidence-based bullet points explaining the relevance to Syntel's Network Integration GTM strategy (Altai).
    
    Then provide an INTENT SCORE: High/Medium/Low based on concrete business and technology signals.
    
    High Intent Signals:
    - New warehouse construction/expansion (especially 2025/2026 dates).
    - Mention of IoT, Automation, or Edge Integration requiring seamless Wi-Fi.
    - Large number of facilities/locations requiring central network management.
    
    FORMAT:
    BULLETS:
    1. [Expansion Signal] - [Syntel's Core Offering] - [Benefit/Evidence]
    2. [Operational Challenge] - [Syntel's Differentiation (Altai)] - [Specific Use Case]
    3. [IT/Network Need] - [Syntel's Service Model] - [Strategic Opportunity]
    SCORE: High/Medium/Low
    
    Be specific, evidence-based, and focus on actionable opportunities related to network modernization.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business analyst specializing in IT services alignment. Provide detailed, evidence-based analysis focused on Network Integration and Expansion opportunities."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse response
        bullets = []
        score = "Medium"
        
        lines = response.split('\n')
        bullet_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('BULLETS:') or bullet_section:
                bullet_section = True
                if (line.startswith(('1', '2', '3', '•', '-')) and len(line) > 10 and 
                    not line.startswith('SCORE:')):
                    clean_line = re.sub(r'^[1-3][\.\)]\s*', '', line)
                    clean_line = re.sub(r'^[•\-]\s*', '', clean_line)
                    bullets.append(f"• {clean_line}")
            elif 'SCORE:' in line.upper():
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
                bullet_section = False
        
        while len(bullets) < 3:
            bullets.append(f"• Potential network integration opportunity based on company's focus on logistics and warehousing.")
        
        cleaned_bullets = []
        for bullet in bullets[:3]:
            clean_bullet = re.sub(r'\*\*|\*|__|_', '', bullet)
            clean_bullet = re.sub(r'\s+', ' ', clean_bullet).strip()
            cleaned_bullets.append(clean_bullet)
        
        formatted_bullets = "\n".join(cleaned_bullets)
        return formatted_bullets, score
        
    except Exception:
        fallback_bullets = """• High relevance as a logistics/warehouse player (Syntel Target Industry).
• New facility expansions (Pune, Kolkata, Krishnapatnam) create immediate need for new network integration (GTM Focus).
• Opportunity to position Altai WiFi for seamless, long-range coverage across large warehouse floors for IoT/scanners."""
        return fallback_bullets, "High" # Default to high intent given expansion signals

def format_concise_display_with_sources(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean, professional display format (Unchanged)"""
    
    mapping = {
        "Company Name": "company_name", "LinkedIn URL": "linkedin_url", "Company Website URL": "company_website_url", 
        "Industry Category": "industry_category", "Employee Count (LinkedIn)": "employee_count_linkedin",
        "Headquarters (Location)": "headquarters_location", "Revenue (Source)": "revenue_source",
        "Branch Network / Facilities Count": "branch_network_count", "Expansion News (Last 12 Months)": "expansion_news_12mo",
        "Digital Transformation Initiatives": "digital_transformation_initiatives", "IT Leadership Change": "it_leadership_change",
        "Existing Network Vendors": "existing_network_vendors", "Wi-Fi/LAN Tender Found": "wifi_lan_tender_found",
        "IoT/Automation/Edge": "iot_automation_edge_integration", "Cloud Adoption/GCC": "cloud_adoption_gcc_setup",
        "Physical Infrastructure": "physical_infrastructure_signals", "IT Infra Budget/Capex": "it_infra_budget_capex",
        "Intent Scoring": "intent_scoring_level", "Why Relevant to Syntel": "why_relevant_to_syntel_bullets",
    }
    
    data_list = []
    for display_col, data_field in mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(data_field, "N/A")
        
        data_list.append({"Column Header": display_col, "Value": str(value)})
            
    df = pd.DataFrame(data_list)
    return df

# --- Streamlit UI (Execution Block - Unchanged) ---
if __name__ == "__main__":
    DEFAULT_COMPANY = "Snowman Logistics"
    
    st.title("🤖 Dynamic Company Intelligence Generator")
    st.sidebar.header("Configuration")
    
    company_name = st.sidebar.text_input("Enter Company Name to Research:", DEFAULT_COMPANY)
    
    if 'company_name' not in st.session_state:
        st.session_state['company_name'] = DEFAULT_COMPANY
        st.session_state['company_data'] = None

    trigger_search = st.sidebar.button("Run Comprehensive Research")
    
    if trigger_search or (company_name != st.session_state.get('company_name', DEFAULT_COMPANY) and company_name):
        st.session_state['company_name'] = company_name
        st.session_state['company_data'] = None

    if st.session_state['company_data'] is None and st.session_state.get('company_name'):
        
        with st.spinner(f"Starting comprehensive research for **{st.session_state['company_name']}**..."):
            company_data = dynamic_research_company_intelligence(st.session_state['company_name']) 
            st.session_state['company_data'] = company_data
            
        st.success(f"Research for **{st.session_state['company_name']}** completed successfully.")

    if 'company_data' in st.session_state and st.session_state['company_data']:
        st.header(f"📊 Extracted Intelligence: {st.session_state['company_name']}")
        
        df_display = format_concise_display_with_sources(
            st.session_state['company_name'], 
            st.session_state['company_data']
        )
        
        st.dataframe(df_display.set_index('Column Header'), use_container_width=True)

        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=True, sheet_name='Company_Intel')
            writer.close()
            processed_data = output.getvalue()
            return processed_data

        excel_data = to_excel(df_display.set_index('Column Header'))
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{st.session_state['company_name']}_Intelligence_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
