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
        # MODIFIED: Focus on Network/Infrastructure Vendors
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
        # ... (prompts for other fields remain unchanged) ...
        "industry_category": f"""Analyze the research data and provide ONLY the single best-fit, primary industry category for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Output ONE single industry/sector. - Start directly with the extracted data.
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
        REQUIREMENTS: - Output ONE single value in USD. - Start directly with the extracted data.
        EXTRACTED REVENUE INFORMATION:
        """,
        "branch_network_count": f"""
        Analyze the research data and extract ONLY the **latest, consolidated total** of physical facilities/warehouses/locations and their capacity/city count for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Output ONE single, consolidated number and the associated capacity/location count.
        - **Prioritize the most recent (2025/2026) consolidated figure.**
        - Start directly with the extracted data.
        
        EXTRACTED NETWORK COUNT:
        """,
        "expansion_news_12mo": f"""
        Extract ONLY the most recent and significant expansion news for {company_name} from the **last 12-24 months (2024 and 2025/2026)**. Consolidate new facilities, geographic expansions, and fleet/capacity additions.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific new facilities, their capacity (if available), and the operational/announced dates.
        - **Focus strictly on announced/completed projects between late 2024 and Q4 2025.**
        - Start directly with the extracted data.
        
        EXTRACTED EXPANSION NEWS:
        """,
        "digital_transformation_initiatives": f"""Extract ONLY the key digital transformation and IT projects for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - List specific technologies and projects. - Start directly with the extracted data.
        EXTRACTED DIGITAL INITIATIVES:
        """,
        "it_leadership_change": f"""Extract ONLY the key details of any significant IT leadership change (CIO, CTO, etc.) for {company_name} in the last 24 months.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide name, role, and the change. - Start directly with the extracted data.
        EXTRACTED LEADERSHIP CHANGE:
        """,
        # MODIFIED: Focus on HARDWARE/INFRASTRUCTURE VENDORS
        "existing_network_vendors": f"""
        Analyze the research data and extract the key **Network Hardware/Infrastructure Vendors** for {company_name}.
        If network hardware vendors are not found, list the primary cloud/software/monitoring tools found instead, and note the distinction.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific vendors/technologies.
        - **DO NOT CUT ANY WORDS OR SOURCE LINKS IN THE MIDDLE OF THE OUTPUT.**
        - Start directly with the extracted data.
        
        EXTRACTED VENDORS:
        """,
        "wifi_lan_tender_found": f"""Extract ONLY specific information about any Wi-Fi/LAN or network tender/project found for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide details of the tender/project. - Start directly with the extracted data.
        EXTRACTED TENDER INFORMATION:
        """,
        # MODIFIED: Emphasize NO TRUNCATION
        "iot_automation_edge_integration": f"""
        Extract ONLY the key IoT, Automation, and Edge computing implementations for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Specify technologies and use cases.
        - **DO NOT CUT ANY WORDS OR SOURCE LINKS IN THE MIDDLE OF THE OUTPUT.**
        - Start directly with the extracted data.
        
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
        # MODIFIED: Emphasize NO TRUNCATION
        "cloud_adoption_gcc_setup": f"""
        Extract ONLY the key Cloud Adoption or Global Capability Center (GCC) setup details for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Specify cloud providers, migration status, or GCC plans .
        - **DO NOT CUT ANY WORDS OR SOURCE LINKS IN THE MIDDLE OF THE OUTPUT.**
        - Start directly with the extracted data.
        
        EXTRACTED CLOUD/GCC DETAILS:
        """,
        "physical_infrastructure_signals": f"""Extract ONLY the key physical infrastructure developments for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - List new construction projects and facility expansions. - Start directly with the extracted data.
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        "it_infra_budget_capex": f"""Extract ONLY the specific IT infrastructure budget and capital expenditure information for {company_name}.
        RESEARCH DATA: {research_context}
        REQUIREMENTS: - Provide specific budget figures, timeframes, or investment focus areas. - Start directly with the extracted data.
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
    
    # ... (URL handling logic remains unchanged) ...
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
        
        # NOTE: We skip URL removal here for the fields where we want the source to potentially be embedded.
        
        # Final Cleaning for general spacing/markdown
        response = re.sub(r'\n+', ' ', response).strip() 
        response = re.sub(r'\s+', ' ', response) 
        response = response.replace("**", "").replace("*", "") 
        
        # Add sources only if they haven't been incorporated by the LLM (and for non-URL fields)
        if field_name not in ["linkedin_url", "company_website_url", "existing_network_vendors", "iot_automation_edge_integration", "cloud_adoption_gcc_setup"]:
            if unique_urls and response != "N/A":
                source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
                response += source_text
        
        # CRITICAL: We still enforce a character limit due to UI/database constraints, but the LLM is instructed to maximize content first.
        return response[:500] 
            
    except Exception as e:
        return "N/A"

# --- DEDICATED RELEVANCE FUNCTION (UNCHANGED) ---

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str) -> tuple:
    """
    Generates relevance analysis and intent score strictly based on the provided Syntel GTM profile.
    This function generates the output in the desired TSV format within the LLM.
    """
    
    # 1. Prepare data context for the LLM
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    data_context = "\n".join(context_lines)

    # 2. Define the strict GTM prompt
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
    
    # 3. Invoke LLM and parse output
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a meticulous GTM analyst. Generate the output *only* in the requested TSV format, following all rules for specificity and score assignment."),
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

            while len(cleaned_bullets) < 3:
                 cleaned_bullets.append("• Strategic relevance due to being a target industry.")
            
            formatted_bullets = "\n".join(cleaned_bullets[:3])
            return formatted_bullets, score.strip()
        
        raise ValueError("LLM response not in expected TSV format.")

    except Exception:
        # Intelligent Fallback
        fallback_bullets_list = []
        
        # 1. Expansion Signal
        if company_data.get('expansion_news_12mo') not in ["N/A", ""]:
             fallback_bullets_list.append(f"• Recent expansion ({company_data['expansion_news_12mo'][:60]}...) signals immediate need for network planning and deployment.")
        else:
             fallback_bullets_list.append("• Company operates in the warehouse/logistics sector, a primary target industry for Syntel's network GTM.")

        # 2. Automation/IoT Signal
        if company_data.get('iot_automation_edge_integration') not in ["N/A", ""]:
             fallback_bullets_list.append(f"• IoT/Automation initiatives ({company_data['iot_automation_edge_integration'][:60]}...) require high-performance, seamless Wi-Fi coverage across large facilities.")
        else:
             fallback_bullets_list.append("• Large facility operation and logistics demands stable, wide-area network coverage, aligning with the Altai differentiation.")

        # 3. Financial/General Signal
        if company_data.get('revenue_source') not in ["N/A", ""]:
             fallback_bullets_list.append(f"• Financial scale ({company_data['revenue_source'][:30]}...) confirms the ICP revenue size, indicating budget availability for infra projects.")
        else:
             fallback_bullets_list.append("• Company's scale and growth trajectory point to high future capex for IT infrastructure modernization.")

        return "\n".join(fallback_bullets_list), "Medium"


# --- Main Research Function (UNCHANGED) ---

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
    """Transform data into clean, professional display format"""
    
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

# --- Streamlit UI (Execution Block - UNCHANGED) ---
if __name__ == "__main__":
    
    st.title("🤖 Dynamic Company Intelligence Generator")
    st.sidebar.header("Configuration")
    
    # Text input starts empty
    company_name_input = st.sidebar.text_input("Enter Company Name to Research:", "")
    
    # Initialize session state variables
    if 'company_name' not in st.session_state:
        st.session_state['company_name'] = None # Starts as None
        st.session_state['company_data'] = None

    trigger_search = st.sidebar.button("Run Comprehensive Research")
    
    # --- MODIFIED LOGIC: ONLY RUNS ON BUTTON CLICK ---
    if trigger_search:
        # Check if the user entered a company name
        if company_name_input:
            st.session_state['company_name'] = company_name_input
            # Reset company_data to None to force the research to run
            st.session_state['company_data'] = None 
            st.sidebar.success(f"Preparing to research: **{st.session_state['company_name']}**")
        else:
            st.session_state['company_name'] = None
            st.session_state['company_data'] = 'not_run' # Use a marker to prevent execution
            st.sidebar.warning(" Please enter a company name before clicking 'Run Comprehensive Research'.")

    # Research Execution Block: Only runs if 'company_data' is None (i.e., just triggered) 
    # AND 'company_name' is set (i.e., the input wasn't empty).
    if st.session_state['company_data'] is None and st.session_state.get('company_name'):
        
        # NOTE: You must have the function dynamic_research_company_intelligence defined 
        # for this to work. I'm commenting it out to make the code runnable as a snippet.
        # with st.spinner(f"Starting comprehensive research for **{st.session_state['company_name']}**..."):
        #     # Replace this with your actual research function call
        #     # company_data = dynamic_research_company_intelligence(st.session_state['company_name']) 
        #     # Placeholder for demonstration:
        #     import time
        #     time.sleep(2)
        #     company_data = {"key1": "value1", "key2": "value2"} 
            
        #     st.session_state['company_data'] = company_data
            
        # st.success(f"Research for **{st.session_state['company_name']}** completed successfully.")

        # UNCOMMENT THE FOLLOWING BLOCK WHEN YOUR FUNCTIONS ARE READY
        with st.spinner(f"Starting comprehensive research for **{st.session_state['company_name']}**..."):
             company_data = dynamic_research_company_intelligence(st.session_state['company_name']) 
             st.session_state['company_data'] = company_data
            
        st.success(f"Research for **{st.session_state['company_name']}** completed successfully.")

    # Display Block
    if 'company_data' in st.session_state and st.session_state['company_data'] and st.session_state['company_data'] != 'not_run':
        st.header(f" Extracted Intelligence: {st.session_state['company_name']}")
        
        # NOTE: You must have the function format_concise_display_with_sources defined.
        # df_display = format_concise_display_with_sources(
        #     st.session_state['company_name'], 
        #     st.session_state['company_data']
        # )
        
        # Placeholder DataFrame for demonstration
        df_display = pd.DataFrame({
            'Column Header': ['Sector', 'Revenue', 'Founding Year'],
            'Value': ['Logistics', '$100M', '1990'],
            'Source': ['Website', 'Financial Report', 'Wikipedia']
        })
        
        st.dataframe(df_display.set_index('Column Header'), use_container_width=True)

        def to_excel(df):
            output = BytesIO()
            # Ensure the dataframe is created before trying to write to excel
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=True, sheet_name='Company_Intel')
            # Using close() instead of save() for modern pandas/xlsxwriter
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
