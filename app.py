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
    
    prompts = {
        "revenue_source": f"""
        Extract detailed revenue and financial information for {company_name}.
        FOCUS ON:
        - Annual revenue in USD ($)
        - Quarterly revenue growth
        - Revenue breakdown if available
        - Financial year results
        - Income statements
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Return revenue figures in USD ($) when possible
        - Include specific numbers and percentages
        - Include time periods (FY 2024, Q1 2025, etc.)
        - Be precise and comprehensive
        
        EXTRACTED FINANCIAL INFORMATION:
        """,
        
        "headquarters_location": f"""
        Extract the complete headquarters address for {company_name}.
        FOCUS ON:
        - Full street address
        - City, State, ZIP/Postal Code
        - Country
        - Official corporate address
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Return the complete official address
        - Include all address components
        - Be precise and accurate
        
        EXTRACTED HEADQUARTERS ADDRESS:
        """,
        
        "branch_network_count": f"""
        Extract detailed branch/network/facility information for {company_name}.
        FOCUS ON:
        - Total number of facilities/warehouses/branches
        - Specific locations and cities
        - Capacities (pallets, square footage, etc.)
        - Recent expansions
        - Geographic coverage
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific counts and locations
        - Include capacity details when available
        - Mention recent expansions
        
        EXTRACTED NETWORK INFORMATION:
        """,
        
        "employee_count_linkedin": f"""
        Extract accurate employee count information for {company_name}.
        FOCUS ON:
        - Current employee count range
        - LinkedIn employee data
        - Recent headcount changes
        - Employee growth trends
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific employee ranges
        - Include source context
        
        EXTRACTED EMPLOYEE COUNT:
        """,
        
        "expansion_news_12mo": f"""
        Extract recent expansion and growth news for {company_name} from last 12-24 months.
        FOCUS ON:
        - New facility openings
        - Geographic expansions
        - Capacity increases
        - Joint ventures/partnerships
        - Investment announcements
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Include specific dates and locations
        - Mention capacities and investments
        - Focus on recent developments (2023-2025)
        
        EXTRACTED EXPANSION NEWS:
        """,
        
        "digital_transformation_initiatives": f"""
        Extract digital transformation and IT initiatives for {company_name}.
        FOCUS ON:
        - ERP implementations (SAP, Oracle, etc.)
        - Automation projects
        - Digital platform developments
        - IT modernization programs
        - Specific technology projects
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific technologies and projects
        - Include implementation status when known
        
        EXTRACTED DIGITAL INITIATIVES:
        """,
        
        "iot_automation_edge_integration": f"""
        Extract IoT, Automation, and Edge computing implementations for {company_name}.
        FOCUS ON:
        - IoT sensor deployments
        - Warehouse automation
        - Robotics implementations
        - Smart technology adoption
        - Edge computing projects
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Specify technologies and use cases
        - Mention implementation scale
        
        EXTRACTED IOT/AUTOMATION DETAILS:
        """,
        
        "physical_infrastructure_signals": f"""
        Extract physical infrastructure developments for {company_name}.
        FOCUS ON:
        - New construction projects
        - Facility expansions
        - Infrastructure investments
        - Capacity enhancements
        - Real estate developments
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Include locations and capacities
        - Mention investment amounts when available
        
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        
        "it_infra_budget_capex": f"""
        Extract IT infrastructure budget and capital expenditure information for {company_name}.
        FOCUS ON:
        - IT budget allocations
        - Technology capex
        - Digital transformation investments
        - Infrastructure spending plans
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific budget figures
        - Include timeframes
        - Mention investment focus areas
        
        EXTRACTED IT BUDGET INFORMATION:
        """
    }
    
    return prompts.get(field_name, f"""
    Extract comprehensive information about {field_name} for {company_name}.
    
    RESEARCH DATA:
    {research_context}
    
    EXTRACTED INFORMATION:
    """)

# --- Enhanced Dynamic Extraction ---
def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Enhanced extraction with better source utilization and accuracy"""
    
    if not search_results:
        return "N/A"
    
    # SPECIAL HANDLING: For LinkedIn URL and Website URL
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
                if not any(social in url.lower() for social in ['linkedin', 'facebook', 'twitter', 'youtube']):
                    return clean_and_format_url(url)
        return "N/A"
    
    # Format detailed research context
    research_context = f"Research data for {company_name} - {field_name}:\n\n"
    for i, result in enumerate(search_results[:4]):  # Use more results
        research_context += f"SOURCE {i+1} - {result.get('title', 'No Title')}:\n"
        research_context += f"CONTENT: {result['content']}\n"
        research_context += f"URL: {result['url']}\n\n"
    
    # Get unique source URLs
    unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:3]
    
    # Use detailed extraction prompt
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context)
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="""You are an expert research analyst. Extract comprehensive, accurate information from the provided research data. 
            For financial data: Always convert to USD when possible and provide specific numbers.
            For locations: Provide complete addresses when available.
            For counts/numbers: Be precise and include units.
            Focus on extracting factual information found in the research sources."""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Validate and clean response
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', 'information not available', ''] or 
            len(response) < 10):
            return "N/A"
        
        # Clean the response
        response = re.sub(r'https?://\S+', '', response)  # Remove URLs
        response = re.sub(r'\n+', '\n', response).strip()
        response = re.sub(r'\s+', ' ', response)
        
        # Special cleaning for industry category
        if field_name == "industry_category":
            return response.split('\n')[0].strip()[:100]
        
        # Add sources for all fields except the basic ones
        if field_name not in ["linkedin_url", "company_website_url", "industry_category"]:
            if unique_urls and response != "N/A":
                source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
                response += source_text
        
        return response[:500]  # Reasonable length limit
            
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
            bullets.append(f"• Strategic IT service opportunity identified for {company_name} based on business growth patterns")
        
        # Clean bullets
        cleaned_bullets = []
        for bullet in bullets[:3]:
            clean_bullet = re.sub(r'\*\*|\*|__|_', '', bullet)
            clean_bullet = re.sub(r'\s+', ' ', clean_bullet).strip()
            cleaned_bullets.append(clean_bullet)
        
        formatted_bullets = "\n".join(cleaned_bullets)
        return formatted_bullets, score
        
    except Exception as e:
        fallback_bullets = f"""• Digital transformation opportunity based on business expansion
• IT infrastructure modernization potential from growth signals
• Automation and efficiency optimization alignment with Syntel expertise"""
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
        
        # For basic fields, return clean data only
        if display_col in ["LinkedIn URL", "Company Website URL", "Industry Category"]:
            data_list.append({"Column Header": display_col, "Value": str(value)})
        
        # Format relevance bullets
        elif data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str) and value != "N/A":
                cleaned_value = value.replace('1)', '•').replace('2)', '•').replace('3)', '•')
                cleaned_value = re.sub(r'^\d\.\s*', '• ', cleaned_value, flags=re.MULTILINE)
                cleaned_value = re.sub(r'\*\*|\*', '', cleaned_value)
                html_value = cleaned_value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left; line-height: 1.4;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # For fields with sources
            if isinstance(value, str) and "http" in value and "Source" in value:
                main_content = value.split(' [Source')[0] if ' [Source' in value else value
                sources_part = value.split(' [Source')[1] if ' [Source' in value else ""
                
                if sources_part:
                    urls = re.findall(r'https?://[^\s,\]]+', sources_part)
                    if urls:
                        source_links = []
                        for i, url in enumerate(urls[:2]):
                            clean_url = clean_and_format_url(url)
                            source_links.append(f'<a href="{clean_url}" target="_blank">Source {i+1}</a>')
                        
                        sources_html = f"<br><small>Sources: {', '.join(source_links)}</small>"
                        display_value = f'<div style="text-align: left; line-height: 1.4;">{main_content}{sources_html}</div>'
                    else:
                        display_value = f'<div style="text-align: left;">{main_content}</div>'
                else:
                    display_value = f'<div style="text-align: left;">{main_content}</div>'
                
                data_list.append({"Column Header": display_col, "Value": display_value})
            else:
                display_value = f'<div style="text-align: left; line-height: 1.4;">{value}</div>'
                data_list.append({"Column Header": display_col, "Value": display_value})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Enhanced Syntel BI Agent",
    layout="wide",
    page_icon="🔍"
)

st.title("🔍 Syntel Enhanced Company Intelligence Agent")
st.markdown("### 🚀 Comprehensive Business Intelligence with Accurate Data Extraction")

# Display enhanced approach
with st.expander("🚀 Enhanced Research Capabilities", expanded=True):
    st.markdown("""
    **Major Improvements Implemented:**
    
    - **💰 Financial Data Accuracy**: Revenue extraction in USD with comprehensive financial context
    - **🏢 Complete Address Extraction**: Full headquarters addresses with street-level details
    - **🔍 Enhanced Source Utilization**: Deeper analysis of source content for accurate information
    - **📊 Detailed Network Coverage**: Specific branch counts, locations, and capacities
    - **🤖 Strategic Relevance Analysis**: Evidence-based opportunity identification
    - **⚡ Comprehensive Data Coverage**: More thorough field research with better prompts
    
    **Key Focus Areas:**
    1. **Revenue & Financials**: USD conversion, income statements, growth percentages
    2. **Location Data**: Complete addresses, not just city names
    3. **Expansion Details**: Specific dates, locations, capacities, investments
    4. **Technology Initiatives**: Concrete project details and implementations
    5. **Strategic Alignment**: Evidence-based Syntel opportunity analysis
    """)

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", "Snowman Logistics")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("🚀 Start Enhanced Research", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"**🔍 Conducting comprehensive research for {company_input}...**"):
        try:
            # Perform enhanced research
            company_data = dynamic_research_company_intelligence(company_input)
            
            # Display results
            st.balloons()
            st.success(f"✅ Enhanced research complete for {company_input}!")
            
            # Display final results
            st.subheader(f"Comprehensive Business Intelligence Report for {company_input}")
            final_df = format_concise_display_with_sources(company_input, company_data)
            
            # Apply custom CSS
            st.markdown("""
            <style>
            .dataframe {
                width: 100%;
            }
            .dataframe th {
                background-color: #f0f2f6;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            .dataframe td {
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show completion metrics
            with st.expander("📊 Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "N/A")
                
                fields_with_sources = sum(1 for field in REQUIRED_FIELDS[:-2]
                                       if company_data.get(field) and 
                                       company_data.get(field) != "N/A" and
                                       "Source" in company_data.get(field, ""))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Fields with Sources", f"{fields_with_sources}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    score_color = {
                        "High": "green", 
                        "Medium": "orange", 
                        "Low": "red"
                    }.get(company_data.get("intent_scoring_level", "Medium"), "gray")
                    st.markdown(f"<h3 style='color: {score_color};'>Intent Score: {company_data.get('intent_scoring_level', 'Medium')}</h3>", unsafe_allow_html=True)
            
            # Download options
            st.subheader("💾 Download Comprehensive Report")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(company_data, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_enhanced_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_enhanced_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_enhanced_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits. Please try again in a few moments.")

# Research History
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

if st.session_state.research_history:
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                if research['data'].get(field) and 
                                research['data'].get(field) != "N/A")
            st.write(f"Fields Completed: {completed_fields}/{len(REQUIRED_FIELDS)}")
        
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Enhanced Syntel BI Agent | Comprehensive Business Intelligence"
    "</div>",
    unsafe_allow_html=True
)
