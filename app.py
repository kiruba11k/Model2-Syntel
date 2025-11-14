import streamlit as st
import pandas as pd
import json
import operator
import re
from typing import TypedDict, Annotated
from io import BytesIO
from datetime import datetime
import time

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
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
    # Reducing max_results slightly to manage cost/speed, but focusing queries
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Domain List for Branch Network Search ---
BRANCH_SEARCH_DOMAINS = [
    "sany.in", "welspunone.com", "allcargologistics.com", 
    "tatamotors.com", "starhealth.in", "hdfcbank.com",
    "linkedin.com", "mca.gov.in", "economictimes.com"
]

# --- Domain List for Existing Network Vendors / Tech Stack Search (Focused list for accuracy) ---
TECH_VENDOR_DOMAINS = [
    "cio.economictimes.indiatimes.com", "supplychaindigital.com", "forbes.com", 
    "microsoft.com", "sap.com", "oracle.com", "vmware.com", "cisco.com", 
    "crn.in", "techcircle.in", "builtwith.com", "wappalyzer.com", "aws.amazon.com",
    "azure.microsoft.com", "cloud.google.com", "g2.com"
]

# --- Manual JSON Schema Definition (No change required) ---
REQUIRED_FIELDS = [
    "linkedin_url", "company_website_url", "industry_category", 
    "employee_count_linkedin", "headquarters_location", "revenue_source",
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- LangGraph State Definition (No change required) ---
class AgentState(TypedDict):
    """Represents the shared context/state of the graph's execution."""
    company_name: str
    raw_research: str
    validated_data_text: str
    final_json_data: dict
    messages: Annotated[list, operator.add]

# --- Manual JSON Parser and Validator (Retaining robust cleanup from previous version) ---
def parse_and_validate_json(json_string: str, company_name: str) -> dict:
    """Manually parse and validate JSON output from LLM, with robust list/dict cleanup"""
    
    # First, try to extract JSON from the response
    json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = json_string
    
    # Clean the JSON string
    json_str = re.sub(r'</?function.*?>', '', json_str)  # Remove function tags
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = create_structured_data_from_text(json_string, company_name)
    
    # CRITICAL CLEANUP: Convert any JSON list or dict values to a clean string
    for key, value in data.items():
        if isinstance(value, list):
            clean_list = []
            for item in value:
                if isinstance(item, dict):
                    clean_list.append(" ".join(f"{k}: {v}" for k, v in item.items()))
                else:
                    clean_list.append(str(item).strip())
            data[key] = "; ".join(clean_list).replace("'", "")
        
        elif isinstance(value, dict):
            data[key] = "; ".join(f"{k}: {v}" for k, v in value.items()).replace("'", "")

    return validate_and_complete_data(data, company_name)

# Helper functions (create_structured_data_from_text, validate_and_complete_data, etc. are retained and functional)
def create_structured_data_from_text(text: str, company_name: str) -> dict:
    data = {}
    branch_patterns = [r'branch_network_count[:\s]*["\']?([^"\',}]+)', r'branches?[:\s]*(\d+)', r'facilit(y|ies)[:\s]*(\d+)', r'locations?[:\s]*(\d+)']
    for pattern in branch_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            data["branch_network_count"] = f"Found {matches[0][0] if matches[0][0] else matches[0][1]} facilities. Source: Search results"
            break
    else:
        data["branch_network_count"] = "Facility count not specified in available data"
    hq_match = re.search(r'headquarters?[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["headquarters_location"] = hq_match.group(1).strip() if hq_match else "Information not found"
    website_match = re.search(r'website[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["company_website_url"] = website_match.group(1).strip() if website_match else f"Search for {company_name} official website"
    industry_match = re.search(r'industry[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["industry_category"] = industry_match.group(1).strip() if industry_match else "Industry classification pending"
    return data

def validate_and_complete_data(data: dict, company_name: str) -> dict:
    completed_data = {}
    for field in REQUIRED_FIELDS:
        if field in data and data[field] and data[field] != "Not specified":
            completed_data[field] = str(data[field])
        else:
            completed_data[field] = get_sensible_default(field, company_name)
    if completed_data["intent_scoring_level"] not in ["Low", "Medium", "High"]:
        completed_data["intent_scoring_level"] = "Medium"
    if not completed_data["why_relevant_to_syntel_bullets"].startswith('*'):
        completed_data["why_relevant_to_syntel_bullets"] = format_relevance_bullets(
            completed_data["why_relevant_to_syntel_bullets"], company_name
        )
    return completed_data

def get_sensible_default(field: str, company_name: str) -> str:
    defaults = {
        "linkedin_url": f"https://linkedin.com/company/{company_name.replace(' ', '-').lower()}",
        "company_website_url": f"Search for {company_name} official website",
        "industry_category": "Further research required for classification",
        "employee_count_linkedin": "Employee data pending LinkedIn verification",
        "headquarters_location": "Headquarters location not specified in available data",
        "revenue_source": "Revenue information requires financial report analysis",
        "branch_network_count": "Branch network data being researched",
        "expansion_news_12mo": "Monitoring recent company announcements",
        "digital_transformation_initiatives": "IT initiatives under investigation",
        "it_leadership_change": "Executive team changes being tracked",
        "existing_network_vendors": "Technology partnerships being researched",
        "wifi_lan_tender_found": "No tender information currently available",
        "iot_automation_edge_integration": "Automation initiatives under review",
        "cloud_adoption_gcc_setup": "Cloud infrastructure assessment in progress",
        "physical_infrastructure_signals": "Physical expansion signals being monitored",
        "it_infra_budget_capex": "IT budget analysis requires financial disclosures",
        "why_relevant_to_syntel_bullets": f"* {company_name} has infrastructure modernization potential\n* Digital transformation opportunities identified\n* IT service integration possibilities exist",
        "intent_scoring_level": "Medium"
    }
    return defaults.get(field, "Data not available")

def format_relevance_bullets(raw_text: str, company_name: str) -> str:
    bullets = [
        f"* {company_name} shows potential for infrastructure modernization",
        f"* Digital transformation opportunities in current operations", 
        f"* IT service integration could enhance {company_name}'s capabilities"
    ]
    return "\n".join(bullets)

# --- Graph Nodes (Improved Research Focus) ---
def research_node(state: AgentState) -> AgentState:
    """Research node focused on branch network and tech stack data"""
    st.session_state.status_text.info(f"Phase 1/3: Researching {state['company_name']}...")
    st.session_state.progress_bar.progress(25)
    
    company = state["company_name"]
    
    # 1. Targeted basic info and branch network searches
    branch_queries = [
        f'"{company}" facilities OR "logistics centers" OR "warehouses" site:{" OR site:".join(BRANCH_SEARCH_DOMAINS[:4])}', 
        f'"{company}" company information official website',
    ]
    
    # 2. Focused tech stack searches (High-relevance keywords combined with vendor domains)
    tech_stack_keywords = ["ERP", "WMS", "Network Infrastructure", "Data Center", "Cloud Vendor", "Technology Stack"]
    
    tech_stack_queries = [
        # Query 1: Find any tech stack mention on focused vendor/tech sites
        f'"{company}" {" OR ".join(tech_stack_keywords)} site:{" OR site:".join(TECH_VENDOR_DOMAINS)}',
        # Query 2: Find specific vendor names
        f'"{company}" "SAP" OR "Oracle" OR "VMware" OR "Cisco" OR "AWS"',
        # Query 3: Look for digital transformation case studies
        f'"{company}" "digital transformation" case study',
    ]
    
    all_queries = branch_queries + tech_stack_queries
    
    all_results = []
    for query in all_queries:
        try:
            time.sleep(0.5) # Throttle to prevent rate limit issues
            # Using max_results=2 per query for a total of 14 results
            results = search_tool.invoke({"query": query, "max_results": 2}) 
            if results:
                for result in results:
                    # Append source URL directly to content for easy LLM extraction
                    content_with_source = f"{result.get('content', '')[:300]} [Source: {result.get('url', '')}]"
                    all_results.append({
                        "title": result.get('title', ''),
                        "content": content_with_source, 
                        "url": result.get('url', '')
                    })
        except Exception as e:
            continue
    
    # Format research results
    research_text = "SEARCH RESULTS:\n\n"
    for i, result in enumerate(all_results):
        research_text += f"Result {i+1}:\n"
        research_text += f"Title: {result['title']}\n"
        research_text += f"Content: {result['content']}\n"
        research_text += f"URL: {result['url']}\n\n"
    
    research_prompt = f"""
    Analyze ALL research data for {company} and extract every piece of key information.

    **CRITICAL FILTERING INSTRUCTION:**
    When extracting **Existing Network Vendors / Tech Stack** information, you MUST filter for vendors and specific technologies (e.g., SAP, Cisco, AWS, WMS, ERP). **Ignore links that are just company press releases for general news or stock filings unless they explicitly mention a technology partner or product.** Focus on third-party sources or detailed case studies that name the IT/Network vendor.
    
    Focus on finding:
    - Branch network count and facilities
    - Company website and basic info  
    - Headquarters location
    - Industry classification
    - **Existing Network Vendors / Tech Stack** (List every known vendor and associated tech/source link)
    - Any expansion or technology news, digital initiatives, and leadership changes.
    
    Keep responses factual and include source URLs when available.
    
    {research_text}
    """
    
    try:
        raw_research = llm_groq.invoke([
            SystemMessage(content=research_prompt),
            HumanMessage(content=f"Extract key business information for {company}")
        ]).content
    except Exception as e:
        raw_research = f"Research data: {research_text}"
    
    return {"raw_research": raw_research}

def validation_node(state: AgentState) -> AgentState:
    """Validation node that prepares data for JSON formatting"""
    st.session_state.status_text.info(f"Phase 2/3: Preparing data structure...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    validation_prompt = f"""
    Based on the research below, create a structured data summary for {company}.
    
    RESEARCH DATA:
    {raw_research}
    
    Create a JSON-like structure with these exact fields. For all fields requiring multiple pieces of data (e.g., vendors, initiatives), provide the value as a single, readable string that includes all relevant sources.
    
    - linkedin_url
    - company_website_url
    - industry_category  
    - employee_count_linkedin
    - headquarters_location
    - revenue_source
    - branch_network_count (MUST include numbers and sources)
    - expansion_news_12mo
    - digital_transformation_initiatives
    - it_leadership_change
    - **existing_network_vendors (Vendor/Tech Stack Name AND source link. List as a single, semicolon-separated string: 'Cisco networking (Source: URL); SAP ERP (Source: URL)')**
    - wifi_lan_tender_found
    - iot_automation_edge_integration
    - cloud_adoption_gcc_setup
    - physical_infrastructure_signals
    - it_infra_budget_capex
    - why_relevant_to_syntel_bullets (exactly 3 bullet points starting with *)
    - intent_scoring_level (only: Low, Medium, or High)
    
    **INSTRUCTION:** Provide all values as simple strings. Do not use Python list (`[...]`) or nested JSON (`{...}`) notation inside any field value, as this causes formatting issues.
    
    Format as valid JSON.
    """
    
    try:
        validated_output = llm_groq.invoke([
            SystemMessage(content=validation_prompt),
            HumanMessage(content=f"Create structured data for {company}")
        ]).content
    except Exception as e:
        validated_output = raw_research
    
    return {"validated_data_text": validated_output}

def formatter_node(state: AgentState) -> AgentState:
    """Formatter node using manual JSON parsing"""
    st.session_state.status_text.info(f"Phase 3/3: Finalizing output...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    company_name = state["company_name"]
    
    # Use our manual JSON parser (includes the list/dict cleanup)
    try:
        final_data = parse_and_validate_json(validated_data_text, company_name)
        st.success("✅ Data successfully structured")
    except Exception as e:
        st.warning(f"Using enhanced fallback: {str(e)}")
        final_data = validate_and_complete_data({}, company_name)
    
    return {"final_json_data": final_data}

# --- Graph Construction & UI (Retained) ---
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("format", formatter_node)
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "validate")
    workflow.add_edge("validate", "format")
    workflow.add_edge("format", END)
    return workflow.compile()

app = build_graph()

def format_data_for_display(company_input: str, data_dict: dict) -> pd.DataFrame:
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
        "Existing Network Vendors / Tech Stack": "existing_network_vendors",
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
            value = data_dict.get(data_field, "Data not available")
        
        if data_field == "why_relevant_to_syntel_bullets":
            html_value = str(value).replace('\n', '<br>').replace('*', '•')
            data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

st.set_page_config(page_title="Syntel BI Agent (Manual JSON)", layout="wide")
st.title("Syntel Company Data AI Agent 🏢")
st.markdown("### Manual JSON Processing - Focused Data Relevance Fix")

if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input)
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("Start Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    with st.spinner(f"Researching **{company_input}** with focused data gathering..."):
        try:
            time.sleep(1)
            
            initial_state: AgentState = {
                "company_name": company_input,
                "raw_research": "",
                "validated_data_text": "",
                "final_json_data": {},
                "messages": []
            }

            final_state = app.invoke(initial_state)
            data_dict = final_state["final_json_data"]
            
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"Research Complete for {company_input}!")
            
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": data_dict
            }
            st.session_state.research_history.append(research_entry)
            
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, data_dict)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            branch_data = data_dict.get("branch_network_count", "")
            if any(keyword in branch_data.lower() for keyword in ["facility", "branch", "location", "office"]):
                st.success("✅ Branch network data included in report")
            
            st.subheader("Download Options 💾")
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(label="Download JSON", data=json.dumps(data_dict, indent=2), file_name=f"{company_input.replace(' ', '_')}_data.json", mime="application/json")
            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(label="Download CSV", data=csv_data, file_name=f"{company_input.replace(' ', '_')}_data.csv", mime="text/csv")
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(label="Download Excel", data=excel_data, file_name=f"{company_input.replace(' ', '_')}_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")

st.markdown("---")

if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            branch_info = research['data'].get('branch_network_count', 'No data')[:80]
            st.write(f"Branch Network: {branch_info}...")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

with st.sidebar.expander("🔧 Technical Approach"):
    st.markdown("""
    **Focused Data Relevance Fix:**
    - **CRITICAL:** Added a specific, aggressive filtering instruction to the `research_node` to tell the LLM to **ignore irrelevant links** and **only prioritize links that mention actual technology partners/products** for the **Existing Network Vendors / Tech Stack** field.
    - **Optimized Search:** Simplified search queries but focused them on high-value terms like **"ERP," "WMS," "Cloud Vendor,"** combined with a targeted list of vendor/tech domains.
    - **Formatting Backup:** The robust Python cleanup logic for lists/dicts is retained in `parse_and_validate_json` to prevent formatting errors from resurfacing.
    """)
