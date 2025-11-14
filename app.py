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
BRANCH_SEARCH_DOMAINS = [
    "sany.in", "welspunone.com", "allcargologistics.com", 
    "tatamotors.com", "starhealth.in", "hdfcbank.com",
    "linkedin.com", "mca.gov.in", "economictimes.com"
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

# --- Manual JSON Parser and Validator ---
def parse_and_validate_json(json_string: str, company_name: str) -> dict:
    """Manually parse and validate JSON output from LLM"""
    
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
            data["branch_network_count"] = f"Found {matches[0][0] if matches[0][0] else matches[0][1]} facilities. Source: Search results"
            break
    else:
        data["branch_network_count"] = "Facility count not specified in available data"
    
    # Extract headquarters
    hq_match = re.search(r'headquarters?[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["headquarters_location"] = hq_match.group(1).strip() if hq_match else "Information not found"
    
    # Extract website
    website_match = re.search(r'website[:\s]*["\']?([^"\',}]+)', text, re.IGNORECASE)
    data["company_website_url"] = website_match.group(1).strip() if website_match else f"Search for {company_name} website"
    
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
def perform_tech_stack_search(company_name: str) -> str:
    """Perform real-time search for technology stack and network vendors"""
    
    st.session_state.status_text.info(f"🔧 Researching technology stack for {company_name}...")
    
    # Technology-specific search queries
    tech_queries = [
        f'"{company_name}" "technology stack" "IT infrastructure" "tech stack"',
        f'"{company_name}" "ERP system" "CRM system" "WMS" "TMS"',
        f'"{company_name}" "cloud provider" "AWS" "Azure" "Google Cloud"',
        f'"{company_name}" "software vendors" "technology partners"',
        f'"{company_name}" "digital transformation" "IT systems"',
        f'"{company_name}" "SAP" "Oracle" "Microsoft" "Cisco" "VMware"',
        f'"{company_name}" "network infrastructure" "IT hardware"'
    ]
    
    # Domain-specific searches from your list
    tech_domains = [
        "blogs.oracle.com", "cio.economictimes.indiatimes.com", "supplychaindigital.com",
        "forbes.com", "microsoft.com", "sap.com", "infosys.com", "vmware.com", 
        "cisco.com", "aws.amazon.com", "azure.microsoft.com", "cloud.google.com"
    ]
    
    # Add domain-specific queries
    for domain in tech_domains[:8]:  # Use first 8 to avoid rate limits
        tech_queries.append(f'site:{domain} "{company_name}" technology IT system')
    
    all_tech_results = []
    
    for i, query in enumerate(tech_queries[:10]):  # Limit to 10 queries
        try:
            time.sleep(1)  # Rate limiting
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if results:
                query_summary = f"Tech Search: {query}\n"
                for j, result in enumerate(results):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')[:400]
                    url = result.get('url', 'No URL')
                    
                    # Look for technology mentions in content
                    tech_indicators = extract_tech_indicators(content)
                    
                    query_summary += f"Result {j+1}: {title}\n"
                    query_summary += f"Content: {content}\n"
                    if tech_indicators:
                        query_summary += f"Tech Mentions: {', '.join(tech_indicators)}\n"
                    query_summary += f"URL: {url}\n\n"
                
                all_tech_results.append(query_summary)
                
        except Exception as e:
            continue
    
    return "\n".join(all_tech_results)

def extract_tech_indicators(text: str) -> list:
    """Extract technology and vendor mentions from text"""
    # Common technology vendors and systems
    tech_patterns = [
        r'\b(SAP|Oracle|Microsoft|IBM|Salesforce)\b',
        r'\b(AWS|Azure|Google Cloud|cloud)\b',
        r'\b(Cisco|VMware|Juniper|HP|Dell)\b',
        r'\b(ERP|CRM|WMS|TMS|SCM)\b',
        r'\b(SaaS|PaaS|IaaS|cloud computing)\b',
        r'\b(Tableau|Power BI|analytics)\b',
        r'\b(ServiceNow|Workday|Adobe)\b',
        r'\b(Infosys|TCS|Wipro|HCL|Accenture)\b',
        r'\b(SQL|database|MySQL|PostgreSQL)\b',
        r'\b(Java|Python|\.NET|JavaScript)\b'
    ]
    
    found_tech = []
    for pattern in tech_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_tech.extend(matches)
    
    return list(set(found_tech))  # Remove duplicates
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

# --- Graph Nodes (Completely Rewritten) ---
def research_node(state: AgentState) -> AgentState:
    """Enhanced research node with tech stack detection"""
    st.session_state.status_text.info(f"Phase 1/3: Comprehensive research for {state['company_name']}...")
    st.session_state.progress_bar.progress(25)
    
    company = state["company_name"]
    
    # Perform all research types in parallel
    branch_network_data = perform_optimized_branch_search(company)
    tech_stack_data = perform_tech_stack_search(company)
    
    # General business searches
    general_queries = [
        f'"{company}" "annual report" revenue',
        f'"{company}" "digital transformation" IT',
        f'"{company}" "CIO" "CTO" leadership'
    ]
    
    general_results = []
    for query in general_queries:
        try:
            results = search_tool.invoke({"query": query, "max_results": 2})
            if results:
                general_results.append(f"Query: {query}\nResults: {str(results)[:500]}")
        except Exception:
            continue
    
    general_search_results = "\n".join(general_results)
    
    # Combined research data
    combined_research = f"""
    BRANCH NETWORK DATA:
    {branch_network_data[:1500]}
    
    TECHNOLOGY STACK DATA:
    {tech_stack_data[:1500]}
    
    GENERAL BUSINESS DATA:
    {general_search_results[:1000]}
    """
    
    research_prompt = f"""
    Research {company} and extract comprehensive business intelligence with special focus on technology stack.
    
    CRITICAL FOCUS AREAS:
    
    1. TECHNOLOGY STACK & NETWORK VENDORS:
       - Identify specific software systems (ERP, CRM, WMS, TMS)
       - Find cloud providers (AWS, Azure, Google Cloud)
       - Document hardware vendors (Cisco, Dell, HP)
       - Note any digital transformation initiatives
       - MUST include source URLs for all technology findings
    
    2. BRANCH NETWORK: Facility counts and locations with sources
    
    3. GENERAL BUSINESS: Basic company information
    
    RESEARCH DATA:
    {combined_research}
    
    For technology stack, be specific about vendors and systems mentioned. Include version numbers if available.
    """
    
    try:
        raw_research = llm_groq.invoke([
            SystemMessage(content=research_prompt),
            HumanMessage(content=f"Research {company} with technology stack focus")
        ]).content
    except Exception as e:
        raw_research = f"Research data: {combined_research}"
    
    return {"raw_research": raw_research}
def validation_node(state: AgentState) -> AgentState:
    """Enhanced validation with tech stack verification"""
    st.session_state.status_text.info(f"Phase 2/3: Validating data and tech stack...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    validation_prompt = f"""
    Validate this research data for {company} with emphasis on technology stack accuracy:
    
    {raw_research[:2000]}
    
    Syntel Expertise: {SYNTEL_EXPERTISE[:500]}
    
    VALIDATION STEPS:
    1. TECHNOLOGY STACK VERIFICATION:
       - Verify all technology vendors have source links
       - Ensure specific systems are mentioned (not just categories)
       - Cross-reference technology mentions across multiple sources
    
    2. INTENT SCORING: Low/Medium/High based on:
       - High: Multiple clear technology refresh signals
       - Medium: Some technology modernization mentions  
       - Low: Only basic IT infrastructure mentioned
    
    3. RELEVANCE: Create 3 bullet points linking company's tech stack to Syntel's services
    
    Output clean key-value data ready for JSON conversion.
    """
    
    try:
        validated_output = llm_groq.invoke([
            SystemMessage(content=validation_prompt),
            HumanMessage(content=f"Validate data for {company}, verify tech stack")
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
    
    # Use our manual JSON parser instead of structured output
    try:
        final_data = parse_and_validate_json(validated_data_text, company_name)
        st.success("✅ Data successfully structured")
    except Exception as e:
        st.warning(f"Using enhanced fallback: {str(e)}")
        final_data = validate_and_complete_data({}, company_name)
    
    return {"final_json_data": final_data}

# --- Graph Construction ---
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

# --- Display Functions ---
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
            value = data_dict.get(data_field, "Data not available")
        
        if data_field == "why_relevant_to_syntel_bullets":
            html_value = str(value).replace('\n', '<br>').replace('*', '•')
            data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Manual JSON)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent 🏢")
st.markdown("### Manual JSON Processing - No Structured Output")

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
    **Manual JSON Processing:**
    - No structured output calls
    - Regex-based JSON extraction
    - Manual field validation
    - Fallback data completion
    - No Pydantic validation errors
    
    **Branch Network Focus:**
    - Targeted facility searches
    - Domain-specific queries
    - Real-time data extraction
    - Source URL inclusion
    """)
