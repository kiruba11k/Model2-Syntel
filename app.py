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

# --- Syntel Core Offerings ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Enhanced Dynamic Search Queries ---
def generate_dynamic_search_queries(company_name: str, field_name: str) -> List[str]:
    """Generate dynamic search queries based on company and field"""
    
    field_queries = {
        "linkedin_url": [
            f'"{company_name}" LinkedIn company page',
            f'"{company_name}" LinkedIn official'
        ],
        "company_website_url": [
            f'"{company_name}" official website',
            f'"{company_name}" company website'
        ],
        "industry_category": [
            f'"{company_name}" industry business sector',
            f'"{company_name}" type of business'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employee count LinkedIn',
            f'"{company_name}" number of employees'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters location address',
            f'"{company_name}" corporate headquarters address'
        ],
        "revenue_source": [
            f'"{company_name}" revenue USD dollars financial results',
            f'"{company_name}" annual revenue income statement',
            f'"{company_name}" financial results revenue'
        ],
        "branch_network_count": [
            f'"{company_name}" branch network facilities locations count',
            f'"{company_name}" offices warehouses locations'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024 2025 new facilities',
            f'"{company_name}" growth expansion news latest'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT initiatives',
            f'"{company_name}" technology modernization projects'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT leadership',
            f'"{company_name}" chief information officer technology'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors Cisco VMware SAP Oracle',
            f'"{company_name}" IT infrastructure vendors partners'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN tender network upgrade',
            f'"{company_name}" network infrastructure project'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation robotics implementation',
            f'"{company_name}" smart technology implementation'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption AWS Azure GCC',
            f'"{company_name}" cloud migration strategy'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility expansion',
            f'"{company_name}" infrastructure development projects'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget capex investment technology spending',
            f'"{company_name}" technology investment budget'
        ]
    }
    
    return field_queries.get(field_name, [f'"{company_name}" {field_name}'])

# --- Enhanced Search with Multiple Queries ---
def dynamic_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Dynamic search for specific field information with multiple query attempts"""
    queries = generate_dynamic_search_queries(company_name, field_name)
    all_results = []
    
    for query in queries[:3]:
        try:
            time.sleep(1.2)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if isinstance(results, str):
                continue
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        if len(content) > 50:
                            all_results.append({
                                "title": result.get('title', ''),
                                "content": content[:800],  # Increased context
                                "url": result.get('url', ''),
                                "field": field_name,
                                "query": query
                            })
            elif isinstance(results, dict):
                content = results.get('content', '') or results.get('snippet', '')
                if len(content) > 50:
                    all_results.append({
                        "title": results.get('title', ''),
                        "content": content[:800],
                        "url": results.get('url', ''),
                        "field": field_name,
                        "query": query
                    })
                    
        except Exception as e:
            continue
    
    return all_results

# --- URL Cleaning Functions ---
def clean_and_format_url(url: str) -> str:
    """Clean and format URLs"""
    if not url or url == "N/A":
        return "N/A"
    
    if url.startswith('//'):
        url = 'https:' + url
    elif url.startswith('http://') and '//' in url[7:]:
        url = url.replace('http://', 'http://').replace('//', '/')
    elif url.startswith('https://') and '//' in url[8:]:
        url = url.replace('https://', 'https://').replace('//', '/')
    
    return url

# --- Enhanced Extraction Prompts ---
def get_detailed_extraction_prompt(company_name: str, field_name: str, research_context: str) -> str:
    """Get detailed extraction prompts for each field"""
    
    # Prompts are optimized to ask for FACTUAL DATA ONLY and nothing else.
    prompts = {
        "revenue_source": f"""
        Extract ONLY the annual revenue (in USD, if possible) and key financial facts for {company_name}.
        Include specific numbers, percentages, and time periods (FY 2024, Q1 2025, etc.).
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Output must be short, factual, and extremely concise.
        - Start directly with the extracted data.
        
        EXTRACTED FINANCIAL INFORMATION:
        """,
        
        "headquarters_location": f"""
        Extract ONLY the full, complete headquarters address for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Output must be the complete official address only.
        - Start directly with the extracted data.
        
        EXTRACTED HEADQUARTERS ADDRESS:
        """,
        
        "branch_network_count": f"""
        Extract ONLY the total number of facilities/branches/locations and key capacity facts for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific counts and key locations (e.g., '43 branches, 8 facilities, expansion in Chennai').
        - Start directly with the extracted data.
        
        EXTRACTED NETWORK INFORMATION:
        """,
        
        "employee_count_linkedin": f"""
        Extract ONLY the current employee count range from LinkedIn and any associated specific number.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific employee ranges (e.g., '501-1,000 employees (LinkedIn), 1087 associated members').
        - Start directly with the extracted data.
        
        EXTRACTED EMPLOYEE COUNT:
        """,
        
        "expansion_news_12mo": f"""
        Extract ONLY the most recent and significant expansion news for {company_name} from the last 12-24 months.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific new facilities, geographic expansions, and dates (e.g., 'New Processing Lab in Abhiramapuram, Chennai (Oct 2025)').
        - Start directly with the extracted data.
        
        EXTRACTED EXPANSION NEWS:
        """,
        
        "digital_transformation_initiatives": f"""
        Extract ONLY the key digital transformation and IT projects for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific technologies and projects (e.g., 'SAP S/4HANA Greenfield Implementation (2021), SAP BTP Integration').
        - Start directly with the extracted data.
        
        EXTRACTED DIGITAL INITIATIVES:
        """,
        
        "iot_automation_edge_integration": f"""
        Extract ONLY the key IoT, Automation, and Edge computing implementations for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Specify technologies and use cases (e.g., 'IoT for temp checks on blood samples, remote monitoring system with ECIL').
        - Start directly with the extracted data.
        
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
        
        "physical_infrastructure_signals": f"""
        Extract ONLY the key physical infrastructure developments for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List new construction projects and facility expansions (e.g., 'Ranchi Integrated Diagnostics Centre, JV with Star Imaging in Maharashtra').
        - Start directly with the extracted data.
        
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        
        "it_infra_budget_capex": f"""
        Extract ONLY the specific IT infrastructure budget and capital expenditure information for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific budget figures, timeframes, or investment focus areas (e.g., 'No figures found, focus on digital transformation and expansion').
        - Start directly with the extracted data.
        
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

# --- Enhanced Dynamic Extraction ---
def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Enhanced extraction with better source utilization and accuracy"""
    
    if not search_results:
        return "N/A"
    
    # SPECIAL HANDLING: For URLs
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
    
    # Format detailed research context
    research_context = f"Research data for {company_name} - {field_name}:\n\n"
    for i, result in enumerate(search_results[:4]):
        research_context += f"SOURCE {i+1} - {result.get('title', 'No Title')}:\n"
        research_context += f"CONTENT: {result['content']}\n\n" 
    
    # Get unique source URLs
    unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:3]
    
    # Use detailed extraction prompt
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context)
    
    try:
        # --- CRITICAL MODIFICATION: Finalized System Message for MAX Conciseness ---
        response = llm_groq.invoke([
            SystemMessage(content=f"""You are an expert research analyst. Extract FACTUAL DATA ONLY from the provided research context for {company_name}.
            **The output must be EXTREMELY CONCISE, FACTUAL, and SHORT (under 100 words).**
            **DO NOT** use any introductory phrases, conversational fillers, or descriptive headers.
            Start the output directly with the correct data point. Omit source mentions from the main text body."""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Validate and clean response
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', 'information not available', ''] or 
            len(response) < 5):
            return "N/A"
        
        # --- CRITICAL MODIFICATION: Expanded Aggressive Cleaning ---
        # Added more patterns to catch LLM conversational habits and clean up list markers
        clean_up_phrases = [
            r'^\s*Based on the provided research data,.*:',
            r'^\s*Here is the extracted information:',
            r'^\s*The key information is:',
            r'^\s*Extracted information:',
            r'^\s*The relevant data is:',
            r'^\s*Here\'s the comprehensive information about the industry categor', 
            r'^\s*The headquarters address for [A-Za-z\s]+ is:',
            r'^\s*Annual Revenue:',
            r'^\s*Components of the address:',
            r'^\s*Key Financial Facts:',
            r'^\s*New facilities:',
            r'^\s*Geographic expansions:',
            r'^\s*Snowman Logistics has:',
            r'^\s*[A-Za-z\s]+ has:',
            r'^\s*\*\s*', # Remove list marker at start
            r'^\s*-\s*', # Remove list marker at start
            r'^\s*\d+\.\s*', # Remove numbered list marker at start
        ]
        
        for phrase in clean_up_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE | re.DOTALL).strip()

        # Final Cleaning
        response = re.sub(r'https?://\S+', '', response)  # Remove stray URLs
        response = re.sub(r'\n+', ' ', response).strip() # Replace all newlines with a single space
        response = re.sub(r'\s+', ' ', response) # Consolidate multiple spaces
        response = response.replace("**", "").replace("*", "") # Remove bolding/emphasis
        
        # Special cleaning for industry category
        if field_name == "industry_category":
            return response.split('\n')[0].strip()[:100]
        
        # Add sources for all fields except the basic ones
        if field_name not in ["linkedin_url", "company_website_url", "industry_category"]:
            if unique_urls and response != "N/A":
                source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
                response += source_text
        
        return response[:500]
            
    except Exception as e:
        return "N/A"

# --- Enhanced Relevance Analysis ---
def generate_dynamic_relevance_analysis(company_data: Dict, company_name: str, all_search_results: List[Dict]) -> tuple:
    """Generate comprehensive relevance analysis"""
    
    # Create detailed context from collected data
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            if clean_value and clean_value != "N/A":
                context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    context = "\n".join(context_lines)
    
    # Get source URLs for credibility
    unique_urls = list(set([result['url'] for result in all_search_results if result.get('url')]))[:3]
    source_context = f"Research sources: {', '.join(unique_urls)}" if unique_urls else "Based on comprehensive research"
    
    relevance_prompt = f"""
    COMPANY: {company_name}
    {source_context}
    
    DETAILED COMPANY DATA:
    {context}
    
    SYNTEL CORE EXPERTISE:
    1. IT Automation/RPA: SyntBots platform for process automation
    2. Digital Transformation: Digital One suite for business transformation
    3. Cloud & Infrastructure: IT Infrastructure Management services
    4. KPO/BPO: Industry-specific knowledge process solutions
    
    TASK: Analyze the specific business needs and technology gaps where Syntel can provide solutions.
    Create 3 detailed, evidence-based bullet points explaining the relevance.
    
    Then provide an INTENT SCORE: High/Medium/Low based on concrete business and technology signals.
    
    High Intent Signals:
    - Active digital transformation projects
    - Recent IT leadership changes
    - Cloud migration initiatives
    - Large infrastructure investments
    - Expansion requiring IT scaling
    
    FORMAT:
    BULLETS:
    1. [Specific Business Need] - [Matching Syntel Solution] - [Evidence from Company Data]
    2. [Technology Gap] - [Syntel Capability] - [Supporting Data Point]
    3. [Growth Challenge] - [Syntel Service] - [Relevant Company Activity]
    SCORE: High/Medium/Low
    
    Be specific, evidence-based, and focus on actionable opportunities.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business analyst specializing in IT services alignment. Provide detailed, evidence-based analysis of business opportunities."),
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
        
        # Ensure we have 3 quality bullets
        while len(bullets) < 3:
            # Modified fallback to be less conversational
            bullets.append(f"• Potential IT service opportunity based on company growth and operational strategy.")
        
        # Clean bullets
        cleaned_bullets = []
        for bullet in bullets[:3]:
            clean_bullet = re.sub(r'\*\*|\*|__|_', '', bullet)
            clean_bullet = re.sub(r'\s+', ' ', clean_bullet).strip()
            cleaned_bullets.append(clean_bullet)
        
        formatted_bullets = "\n".join(cleaned_bullets)
        return formatted_bullets, score
        
    except Exception as e:
        # Finalized fallback for complete failure
        fallback_bullets = """• IT infrastructure modernization alignment due to expansion signals.
• Opportunity for Digital Transformation based on technology adoption.
• Process optimization and automation potential aligns with Syntel expertise."""
        return fallback_bullets, "Medium"

# --- Main Research Function ---
def dynamic_research_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Main function to conduct comprehensive company research"""
    
    company_data = {}
    all_search_results = []
    
    # Research each field with enhanced coverage
    total_fields = len(REQUIRED_FIELDS) - 2
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.info(f"🔍 Researching {field.replace('_', ' ').title()} for {company_name}...")
        
        try:
            # Enhanced search with multiple queries
            search_results = dynamic_search_for_field(company_name, field)
            all_search_results.extend(search_results)
            
            # Comprehensive extraction
            field_data = dynamic_extract_field_with_sources(company_name, field, search_results)
            company_data[field] = field_data
            
            time.sleep(1.2)
            
        except Exception as e:
            company_data[field] = "N/A"
            continue
    
    # Generate comprehensive relevance analysis
    status_text.info("🤔 Conducting strategic relevance analysis...")
    progress_bar.progress(90)
    
    try:
        relevance_bullets, intent_score = generate_dynamic_relevance_analysis(
            company_data, company_name, all_search_results
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception as e:
        company_data["why_relevant_to_syntel_bullets"] = "• Comprehensive analysis in progress based on company growth and technology initiatives"
        company_data["intent_scoring_level"] = "Medium"
    
    progress_bar.progress(100)
    status_text.success("✅ Comprehensive research complete!")
    
    return company_data

# --- Display Functions ---
def format_concise_display_with_sources(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean, professional display format"""
    
    mapping = {
        "Company Name": "company_name",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url", 
        "Industry Category": "industry_category",
        "Employee Count (LinkedIn)": "employee_count_linkedin",
        "Headquarters (Location)": "headquarters_location",
        "Revenue (Source)": "revenue_source",
        "Branch Network / Facilities Count": "branch_network_count",
        "Expansion News (Last 12 Months)": "expansion_news_12mo",
        "Digital Transformation Initiatives": "digital_transformation_initiatives",
        "IT Leadership Change": "it_leadership_change",
        "Existing Network Vendors": "existing_network_vendors",
        "Wi-Fi/LAN Tender Found": "wifi_lan_tender_found",
        "IoT/Automation/Edge": "iot_automation_edge_integration",
        "Cloud Adoption/GCC": "cloud_adoption_gcc_setup",
        "Physical Infrastructure": "physical_infrastructure_signals",
        "IT Infra Budget/Capex": "it_infra_budget_capex",
        "Intent Scoring": "intent_scoring_level",
        "Why Relevant to Syntel": "why_relevant_to_syntel_bullets",
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

# --- Streamlit UI (Standard Structure) ---
if __name__ == "__main__":
    st.title("🤖 Dynamic Company Intelligence Generator")
    st.sidebar.header("Configuration")
    
    # Input field
    company_name = st.sidebar.text_input("Enter Company Name to Research:", "Neuberg Diagnostics")
    
    # Research button
    if st.sidebar.button("Run Comprehensive Research"):
        if company_name:
            st.session_state['company_name'] = company_name
            st.session_state['company_data'] = None
            
            with st.spinner(f"Starting comprehensive research for **{company_name}**..."):
                company_data = dynamic_research_company_intelligence(company_name)
                st.session_state['company_data'] = company_data
                
            st.success(f"Research for **{company_name}** completed successfully.")

    # Display results
    if 'company_data' in st.session_state and st.session_state['company_data']:
        st.header(f"📊 Extracted Intelligence: {st.session_state['company_name']}")
        
        # Format the data into a clean DataFrame
        df_display = format_concise_display_with_sources(
            st.session_state['company_name'], 
            st.session_state['company_data']
        )
        
        # Display as a table, hiding the index
        st.dataframe(df_display.set_index('Column Header'), use_container_width=True)

        # Download button
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
