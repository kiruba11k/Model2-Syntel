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
    "linkedin_url", "company_website_url", "industry_category", 
    "employee_count_linkedin", "headquarters_location", "revenue_source",
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- Core Functions (Extraction, Search, and Relevance) ---

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
        "existing_network_vendors": [f'"{company_name}" network infrastructure vendors Cisco HPE', f'"{company_name}" technology stack'],
        "wifi_lan_tender_found": [f'"{company_name}" WiFi LAN tender network upgrade'],
        "iot_automation_edge_integration": [f'"{company_name}" IoT automation robotics implementation'],
        "cloud_adoption_gcc_setup": [f'"{company_name}" cloud adoption AWS Azure GCC'],
        "physical_infrastructure_signals": [f'"{company_name}" new construction facility expansion'],
        "it_infra_budget_capex": [f'"{company_name}" IT budget capex investment technology spending']
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
        "existing_network_vendors": f"""
        Analyze the research data and extract the key **Network Hardware/Infrastructure Vendors** (e.g., Cisco, HPE, Ruckus, Dell) for {company_name}.
        If network hardware vendors are not found, list the primary cloud/software/monitoring tools found instead, and note the distinction.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific vendors/technologies (e.g., 'Cisco (Network), VMware (Virtualization), SAP (ERP)' or 'AWS, PostgreSQL, Grafana (Software/Cloud Stack)').
        - **DO NOT CUT ANY WORDS OR SOURCE LINKS IN THE MIDDLE OF THE OUTPUT.**
        - Start directly with the extracted data.
        
        EXTRACTED VENDORS:
        """,
        "wifi_lan_tender_found": f"""Extract ONLY specific information about any Wi-Fi/LAN or network tender/project found for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide details of the tender/project (e.g., 'RFP for WAN upgrade in Q3 2025' or 'No specific tender found.'). - Start directly with the extracted data.
        EXTRACTED TENDER INFORMATION:
        """,
        "iot_automation_edge_integration": f"""
        Extract ONLY the key IoT, Automation, and Edge computing implementations for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Specify technologies and use cases (e.g., 'IoT for temp checks on blood samples, remote monitoring system with ECIL').
        - **DO NOT CUT ANY WORDS OR SOURCE LINKS IN THE MIDDLE OF THE OUTPUT.**
        - Start directly with the extracted data.
        
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
        "cloud_adoption_gcc_setup": f"""
        Extract ONLY the key Cloud Adoption or Global Capability Center (GCC) setup details for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Specify cloud providers, migration status, or GCC plans (e.g., 'Hybrid Cloud model with Azure, no GCC plans found.').
        - **DO NOT CUT ANY WORDS OR SOURCE LINKS IN THE MIDDLE OF THE OUTPUT.**
        - Start directly with the extracted data.
        
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
    
    RESEARCH DATA: {research_context}
    
    REQUIREMENTS: - Output must be short, factual, and extremely concise. - Start directly with the extracted data.
    EXTRACTED INFORMATION:
    """)

def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    
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
            **The output must be FACTUAL and concise, adhering strictly to the prompt's instructions (especially regarding NO TRUNCATION).**
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
        
        response = re.sub(r'\n+', ' ', response).strip() 
        response = re.sub(r'\s+', ' ', response) 
        response = response.replace("**", "").replace("*", "") 
        
        if field_name not in ["linkedin_url", "company_website_url", "existing_network_vendors", "iot_automation_edge_integration", "cloud_adoption_gcc_setup"]:
            if unique_urls and response != "N/A":
                source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
                response += source_text
        
        return response[:500] 
            
    except Exception as e:
        return "N/A"

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str) -> tuple:
    
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    data_context = "\n".join(context_lines)

    relevance_prompt = f"""
    You are evaluating whether the company below is relevant to Syntel’s Go-To-Market for Wi-Fi & Network Integration.
    
    ---
    **SYNTEL GTM FOCUS**
    **Geography:** India
    **Industries:** Ports, Stadiums, Education, Manufacturing, Healthcare, Hospitality, **Warehouses**, BFSI, IT/ITES, GCCs
    **ICP:** 150–500+ employees, ₹100 Cr+ revenue
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
    
    **TASK:**
    1. Determine the Intent Score (High / Medium / Low) based *only* on the Buying Signals present in the Company Details.
    2. Generate a 3-bullet point summary for "Why Relevant to Syntel." The bullets **MUST be specific and directly reference the company's data (e.g., 'New facility construction in Pune signals immediate need for network.')**.
    3. Output the final result in the exact TSV format specified below.
    
    **OUTPUT FORMAT (TSV):**
    Company Name\tWhy Relevant to Syntel\tIntent (High / Medium / Low)
    
    **RULES:**
    - "Why Relevant" must contain the 3 bullet points, separated by a newline.
    - Do not include headers in the output.
    - Ensure the bullets are short and professional.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a meticulous GTM analyst. Generate the output *only* in the requested TSV format, following all rules for specificity and score assignment."),
            HumanMessage(content=relevance_prompt)
        ]).content.strip()
        
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

            while len(cleaned_bullets) < 3:
                 cleaned_bullets.append("• Strategic relevance due to being a target industry.")
            
            formatted_bullets = "\n".join(cleaned_bullets[:3])
            return formatted_bullets, score.strip()
        
        raise ValueError("LLM response not in expected TSV format.")

    except Exception:
        fallback_bullets_list = []
        
        if company_data.get('expansion_news_12mo') not in ["N/A", ""]:
             fallback_bullets_list.append(f"• Recent expansion ({company_data['expansion_news_12mo'][:60]}...) signals immediate need for network planning and deployment.")
        else:
             fallback_bullets_list.append("• Company operates in a large facility sector, a primary target industry for Syntel's network GTM.")

        if company_data.get('iot_automation_edge_integration') not in ["N/A", ""]:
             fallback_bullets_list.append(f"• IoT/Automation initiatives ({company_data['iot_automation_edge_integration'][:60]}...) require high-performance, seamless Wi-Fi coverage across large facilities.")
        else:
             fallback_bullets_list.append("• Large physical spaces demand stable, wide-area network coverage, aligning with the Altai differentiation.")

        if company_data.get('revenue_source') not in ["N/A", ""]:
             fallback_bullets_list.append(f"• Financial scale ({company_data['revenue_source'][:30]}...) confirms the ICP revenue size, indicating budget availability for infra projects.")
        else:
             fallback_bullets_list.append("• Company's scale and growth trajectory point to high future capex for IT infrastructure modernization.")

        return "\n".join(fallback_bullets_list), "Medium"

# --- Main Research Function ---

def dynamic_research_company_intelligence(company_name: str) -> Dict[str, Any]:
    
    company_data = {}
    all_search_results = []
    
    total_fields = len(REQUIRED_FIELDS) - 2
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
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
            company_data, company_name
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception:
        company_data["why_relevant_to_syntel_bullets"] = "• Analysis failure: Check core data points for LLM processing."
        company_data["intent_scoring_level"] = "Medium"
    
    progress_bar.progress(100)
    status_text.success(" Comprehensive research complete!")
    
    return company_data

def format_concise_display_with_sources(company_input: str, data_dict: dict) -> pd.DataFrame:
    
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

# --- Streamlit UI (Execution Block - MODIFIED) ---
if __name__ == "__main__":
    
    # Initialize session state for company name and data
    if 'company_name' not in st.session_state:
        st.session_state['company_name'] = ""
    if 'company_data' not in st.session_state:
        st.session_state['company_data'] = None
    if 'search_triggered' not in st.session_state:
        st.session_state['search_triggered'] = False

    st.title("🤖 Dynamic Company Intelligence Generator")
    st.sidebar.header("Configuration")
    
    # 1. Input field for company name (empty by default)
    company_name_input = st.sidebar.text_input("Enter Company Name to Research:", value=st.session_state['company_name'])
    
    # 2. Search button
    trigger_search = st.sidebar.button("Run Comprehensive Research")
    
    # Logic to trigger search
    if trigger_search and company_name_input.strip():
        # Update session state and flag search
        st.session_state['company_name'] = company_name_input.strip()
        st.session_state['company_data'] = None  # Clear old data
        st.session_state['search_triggered'] = True
    elif trigger_search and not company_name_input.strip():
        st.sidebar.warning("Please enter a company name to begin the search.")
        st.session_state['search_triggered'] = False
        st.session_state['company_data'] = None

    # Check if a search was triggered and we need to run the research function
    if st.session_state['search_triggered'] and st.session_state['company_data'] is None:
        
        company_to_search = st.session_state['company_name']
        
        with st.spinner(f"Starting comprehensive research for **{company_to_search}**..."):
            company_data = dynamic_research_company_intelligence(company_to_search) 
            st.session_state['company_data'] = company_data
            
        st.success(f"Research for **{company_to_search}** completed successfully.")
        st.session_state['search_triggered'] = False # Reset flag

    # Display results if data exists
    if st.session_state['company_data']:
        company_display_name = st.session_state['company_name']
        st.header(f" Extracted Intelligence: {company_display_name}")
        
        df_display = format_concise_display_with_sources(
            company_display_name, 
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
            file_name=f"{company_display_name}_Intelligence_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif not st.session_state['search_triggered']:
        st.info(" Enter a company name in the sidebar and click 'Run Comprehensive Research' to begin.")
