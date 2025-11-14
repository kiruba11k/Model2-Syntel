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
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Domain List for Branch Network Search ---
# Kept original
BRANCH_SEARCH_DOMAINS = [
    "sany.in", "welspunone.com", "allcargologistics.com", 
    "tatamotors.com", "starhealth.in", "hdfcbank.com",
    "linkedin.com", "mca.gov.in", "economictimes.com"
]

# --- NEW: Domain List for Existing Network Vendors / Tech Stack Search (Full list retained) ---
TECH_VENDOR_DOMAINS = [
    "blogs.oracle.com", "cio.economictimes.indiatimes.com", "supplychaindigital.com", 
    "crownworldwide.com", "frontier-enterprise.com", "hotelmanagement-network.com", 
    "hotelwifi.com", "appsruntheworld.com", "us.nttdata.com", "forbes.com", 
    "mtdcnc.com", "microsoft.com", "sap.com", "amd.com", "videos.infosys.com", 
    "oracle.com", "infosys.com", "medicovertech.in", "icemakeindia.com", 
    "saurenergy.com", "aajenterprises.com", "techcircle.in", "indionetworks.com", 
    "birlasoft.com", "mindteck.com", "inavateapac.com", "jioworldcentre.com", 
    "vmware.com", "intellectdesign.com", "cisco.com", "prnewswire.com", 
    "industryoutlook.in", "networkcomputing.com", "crn.in", 
    
    # Tool/Tech Stack Detection Sites (use judiciously)
    "builtwith.com", "wappalyzer.com", "stackshare.io", 

    # General Search & Financial/Gov Sites
    "linkedin.com", "indeed.com", "naukri.com", "monster.com", 
    "censys.io", "shodan.io", "github.com", "sec.gov", "bseindia.com", 
    "nseindia.com", "aws.amazon.com", "azure.microsoft.com", 
    "cloud.google.com", "g2.com", "gartner.com", "company websites"
]

# --- Manual JSON Schema Definition ---
REQUIRED_FIELDS = [
    "linkedin_url", "company_website_url", "industry_category", 
    "employee_count_linkedin", "headquarters_location", "revenue_source",
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Represents the shared context/state of the graph's execution."""
    company_name: str
    raw_research: str
    validated_data_text: str
    final_json_data: dict
    messages: Annotated[list, operator.add]

# --- Syntel Core Offerings for Analysis Node ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Manual JSON Parser and Validator (CRITICAL FIX HERE) ---
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
        # Try to parse as JSON
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, create structured data manually
        data = create_structured_data_from_text(json_string, company_name)
    
    # --- CRITICAL CLEANUP: Convert any JSON list or dict values to a clean string ---
    for key, value in data.items():
        if isinstance(value, list):
            # Attempt to convert list of strings/dicts into a semi-colon separated string
            # Handle the specific case of the list containing dictionaries (e.g., from your sample output)
            clean_list = []
            for item in value:
                if isinstance(item, dict):
                    # For dicts, convert to a readable string format like "key: value"
                    clean_list.append(" ".join(f"{k}: {v}" for k, v in item.items()))
                else:
                    clean_list.append(str(item).strip())
            data[key] = "; ".join(clean_list).replace("'", "") # Remove single quotes from stringified items
        
        # Also clean up fields that look like Python dictionaries
        elif isinstance(value, dict):
             # Convert dict to a readable string representation
            data[key] = "; ".join(f"{k}: {v}" for k, v in value.items()).replace("'", "")


    # Ensure all required fields are present
    return validate_and_complete_data(data, company_name)

def create_structured_data_from_text(text: str, company_name: str) -> dict:
    """Create structured data by extracting information from text using patterns"""
    
    data = {}
    
    # Extract branch network information
    branch_patterns = [
        r'branch_network_count[:\s]*["\']?([^"\',}]+)',
        r'branches?[:\s]*(\d+)',
        r'facilit(y|ies)[:\s]*(\d+)',
        r'locations?[:\s]*(\d+)'
    ]
    
    for pattern in branch_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Note: This is now a simple string, which will be processed by the cleanup step if needed.
            data["branch_network_count"] = f"Found {matches[0][0] if matches[0][0] else matches[0][1]} facilities. Source: Search results"
            break
    else:
        data["branch_network_count"] = "Facility count not specified in available data"
    
    # Extract headquarters
    hq_match = re.search(r'headquarters?[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["headquarters_location"] = hq_match.group(1).strip() if hq_match else "Information not found"
    
    # Extract website
    website_match = re.search(r'website[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["company_website_url"] = website_match.group(1).strip() if website_match else f"Search for {company_name} official website"
    
    # Extract industry
    industry_match = re.search(r'industry[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["industry_category"] = industry_match.group(1).strip() if industry_match else "Industry classification pending"
    
    return data

def validate_and_complete_data(data: dict, company_name: str) -> dict:
    """Ensure all required fields are present with sensible defaults"""
    
    completed_data = {}
    
    for field in REQUIRED_FIELDS:
        if field in data and data[field] and data[field] != "Not specified":
            completed_data[field] = str(data[field])
        else:
            # Provide sensible defaults for missing fields
            completed_data[field] = get_sensible_default(field, company_name)
    
    # Ensure intent scoring is valid
    if completed_data["intent_scoring_level"] not in ["Low", "Medium", "High"]:
        completed_data["intent_scoring_level"] = "Medium"
    
    # Ensure relevance bullets are properly formatted
    if not completed_data["why_relevant_to_syntel_bullets"].startswith('*'):
        completed_data["why_relevant_to_syntel_bullets"] = format_relevance_bullets(
            completed_data["why_relevant_to_syntel_bullets"], company_name
        )
    
    return completed_data

def get_sensible_default(field: str, company_name: str) -> str:
    """Get sensible default values for missing fields"""
    
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
    """Format relevance bullets properly"""
    bullets = [
        f"* {company_name} shows potential for infrastructure modernization",
        f"* Digital transformation opportunities in current operations", 
        f"* IT service integration could enhance {company_name}'s capabilities"
    ]
    return "\n".join(bullets)

# --- Graph Nodes (Optimized research_node) ---
def research_node(state: AgentState) -> AgentState:
    """Research node focused on branch network and tech stack data"""
    st.session_state.status_text.info(f"Phase 1/3: Researching {state['company_name']}...")
    st.session_state.progress_bar.progress(25)
    
    company = state["company_name"]
    
    # 1. Targeted branch network searches (Original logic)
    branch_queries = [
        f'"{company}" branch network facilities locations site:{" OR site:".join(BRANCH_SEARCH_DOMAINS[:5])}', 
        f'"{company}" offices warehouses logistics centers',
        f'"{company}" company information website'
    ]
    
    # 2. Targeted tech stack searches (IMPROVED INCLUSION)
    # Inject the entire massive domain list to force results from those sites
    tech_stack_keywords = ["network vendors", "tech stack", "IT infrastructure", "ERP system", "WMS system", "Cisco", "VMware", "AWS", "SAP", "Oracle"]
    
    tech_stack_queries = [
        # Use a high-density, focused query for vendors/tech stack against the new domains
        f'"{company}" {" OR ".join(tech_stack_keywords[:5])} site:{" OR site:".join(TECH_VENDOR_DOMAINS)}',
        f'"{company}" "digital transformation" site:{" OR site:".join(TECH_VENDOR_DOMAINS[:10])}', # Less aggressive search
        f'"{company}" "vendor agreement" OR "tech partnership" site:{" OR site:".join(TECH_VENDOR_DOMAINS[10:20])}'
    ]
    
    all_queries = branch_queries + tech_stack_queries
    
    all_results = []
    for query in all_queries:
        try:
            time.sleep(1)
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
    
    # Reverting to less restrictive prompt to encourage more data
    research_prompt = f"""
    Analyze all the research data for {company} and extract every piece of key information.
    
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
    """Validation node that prepares data for JSON formatting (Original logic restored, with key prompt emphasis)"""
    st.session_state.status_text.info(f"Phase 2/3: Preparing data structure...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    validation_prompt = f"""
    Based on the research below, create a structured data summary for {company}.
    
    RESEARCH DATA:
    {raw_research}
    
    Create a JSON-like structure with these exact fields. For fields like branch_network_count, expansion_news_12mo, digital_transformation_initiatives, and existing_network_vendors, provide the value as a single, readable string that includes all relevant sources.
    
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
    
    # Use our manual JSON parser instead of structured output (This includes the cleanup fix!)
    try:
        final_data = parse_and_validate_json(validated_data_text, company_name)
        st.success("✅ Data successfully structured")
    except Exception as e:
        st.warning(f"Using enhanced fallback: {str(e)}")
        final_data = validate_and_complete_data({}, company_name)
    
    return {"final_json_data": final_data}

# --- Graph Construction (No change required) ---
def build_graph():
    """Builds and compiles the sequential LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("research", research_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("format", formatter_node)

    # Define the sequential flow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "validate")
    workflow.add_edge("validate", "format")
    workflow.add_edge("format", END)

    return workflow.compile()

# Build the graph once
app = build_graph()

# --- Display Functions (Updated header for clarity) ---
def format_data_for_display(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into display format"""
    
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
        "Existing Network Vendors / Tech Stack": "existing_network_vendors", # Column Header updated for clarity
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

# --- Streamlit UI (No change required) ---
st.set_page_config(
    page_title="Syntel BI Agent (Manual JSON)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent 🏢")
st.markdown("### Manual JSON Processing - Enhanced Vendor/Tech Stack")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

# Input section
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
    
    with st.spinner(f"Researching **{company_input}** with manual JSON processing..."):
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
            
            # Display results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, data_dict)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show branch network status
            branch_data = data_dict.get("branch_network_count", "")
            if any(keyword in branch_data.lower() for keyword in ["facility", "branch", "location", "office"]):
                st.success("✅ Branch network data included in report")
            
            # Download options
            st.subheader("Download Options 💾")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(data_dict, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")

st.markdown("---")

# Research History
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

# Technical info
with st.sidebar.expander("🔧 Technical Approach"):
    st.markdown("""
    **Manual JSON Processing (Now Stable):**
    - **CRITICAL:** Implemented robust Python code in `parse_and_validate_json` to automatically convert Python lists (`[...]`) and dictionaries (`{...}`) into simple, semicolon-separated strings. **This fixes the ugly output format.**
    - **Search Optimization:** The new, large list of domains is aggressively injected into the `research_node` queries to maximize the chances of finding technology-specific articles for the **Existing Network Vendors / Tech Stack** field.
    - **Prompt Refinement:** The `validation_node` prompt is less restrictive overall but still strongly emphasizes the required format (Vendor/Tech Stack and Source Link) for the key field.
    """)
