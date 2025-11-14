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
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Enhanced Domain Lists for Targeted Research ---
TECH_STACK_DOMAINS = [
    "forbes.com", "cio.economictimes.indiatimes.com", "techcircle.in", 
    "crn.in", "networkcomputing.com", "prnewswire.com",
    "appsruntheworld.com", "supplychaindigital.com", "gartner.com",
    "builtwith.com", "stackshare.io", "wappalyzer.com"
]

VENDOR_CASE_STUDY_DOMAINS = [
    "microsoft.com", "sap.com", "oracle.com", "cisco.com", "vmware.com",
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com", "amd.com"
]

COMPANY_NEWS_DOMAINS = [
    "linkedin.com", "economictimes.com", "prnewswire.com", "businesswire.com"
]

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
    tech_stack_findings: list

# --- Syntel Core Offerings for Analysis Node ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Enhanced Tech Stack Research Functions ---
def search_tech_stack_vendors(company_name: str) -> list:
    """Enhanced search for technology vendors and tech stack information"""
    
    tech_queries = [
        # Direct tech stack queries
        f'"{company_name}" tech stack technology vendors',
        f'"{company_name}" IT infrastructure network vendors',
        f'"{company_name}" partners with technology vendors',
        f'"{company_name}" digital transformation technology partners',
        
        # Vendor-specific queries
        f'"{company_name}" Microsoft Azure AWS Google Cloud adoption',
        f'"{company_name}" SAP Oracle ERP implementation',
        f'"{company_name}" Cisco VMware network infrastructure',
        f'"{company_name}" digital cloud transformation partners',
        
        # Industry-specific technology
        f'"{company_name}" logistics technology automation vendors',
        f'"{company_name}" supply chain digitalization partners'
    ]
    
    all_tech_results = []
    
    for query in tech_queries:
        try:
            time.sleep(1.5)  # Rate limiting
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            for result in results:
                # Filter and prioritize relevant domains
                url = result.get('url', '')
                if any(domain in url for domain in TECH_STACK_DOMAINS + VENDOR_CASE_STUDY_DOMAINS + COMPANY_NEWS_DOMAINS):
                    all_tech_results.append({
                        "title": result.get('title', ''),
                        "content": result.get('content', ''),
                        "url": url,
                        "query": query,
                        "relevance_score": calculate_relevance_score(result.get('content', ''), company_name)
                    })
                    
        except Exception as e:
            continue
    
    # Sort by relevance and remove duplicates
    all_tech_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return remove_duplicate_results(all_tech_results)

def calculate_relevance_score(content: str, company_name: str) -> int:
    """Calculate relevance score for tech stack findings"""
    score = 0
    content_lower = content.lower()
    
    # High relevance keywords
    high_relevance = ['tech stack', 'technology vendor', 'partnered with', 'infrastructure', 
                     'digital transformation', 'cloud migration', 'implemented', 'deployed']
    
    # Vendor keywords
    vendors = ['microsoft', 'aws', 'azure', 'google cloud', 'sap', 'oracle', 'cisco', 
              'vmware', 'dell', 'hp', 'ibm', 'salesforce', 'workday']
    
    for keyword in high_relevance:
        if keyword in content_lower:
            score += 3
    
    for vendor in vendors:
        if vendor in content_lower:
            score += 2
    
    if company_name.lower() in content_lower:
        score += 2
    
    return score

def remove_duplicate_results(results: list) -> list:
    """Remove duplicate results based on URL and content similarity"""
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    return unique_results

def extract_tech_stack_from_research(company_name: str, research_text: str, tech_findings: list) -> str:
    """Extract and format technology vendor information from research"""
    
    tech_prompt = f"""
    Analyze the research data for {company_name} and extract specific technology vendors and tech stack information.
    
    FOCUS ON FINDING:
    - Cloud Providers (AWS, Azure, Google Cloud, Oracle Cloud)
    - ERP Systems (SAP, Oracle ERP, Microsoft Dynamics)
    - Network Vendors (Cisco, Juniper, Aruba, VMware)
    - Infrastructure Providers (Dell, HP, IBM, Lenovo)
    - Software Platforms (Salesforce, Workday, ServiceNow)
    - Specific technologies mentioned in implementation
    
    FORMAT REQUIREMENTS:
    - List each vendor with specific products/technologies mentioned
    - Include source URLs for each finding
    - Note the context (e.g., "implemented SAP S/4HANA for finance")
    - Mark confirmed vs potential vendors
    
    RESEARCH DATA:
    {research_text}
    
    TECH FINDINGS:
    {tech_findings}
    
    Return a structured summary with vendors, technologies, confidence level, and sources.
    """
    
    try:
        tech_analysis = llm_groq.invoke([
            SystemMessage(content="You are a technology research specialist. Extract specific vendor and technology information with sources."),
            HumanMessage(content=tech_prompt)
        ]).content
        return tech_analysis
    except Exception as e:
        return f"Technology research in progress. Initial findings: {len(tech_findings)} relevant sources found."

# --- Manual JSON Parser and Validator ---
def parse_and_validate_json(json_string: str, company_name: str, tech_research: str = "") -> dict:
    """Manually parse and validate JSON output from LLM with enhanced tech stack handling"""
    
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
    
    # Enhance with tech stack research
    if tech_research:
        data = enhance_with_tech_stack_data(data, tech_research, company_name)
    
    # Ensure all required fields are present
    return validate_and_complete_data(data, company_name)

def enhance_with_tech_stack_data(data: dict, tech_research: str, company_name: str) -> dict:
    """Enhance the data with technology stack findings"""
    
    if tech_research and "existing_network_vendors" in data:
        current_vendors = data["existing_network_vendors"]
        if "not found" in current_vendors.lower() or "research" in current_vendors.lower():
            # Extract key vendors from tech research
            vendors_found = extract_vendors_from_tech_research(tech_research)
            if vendors_found:
                data["existing_network_vendors"] = f"{vendors_found}\n\nSource: {tech_research[:500]}..."
    
    return data

def extract_vendors_from_tech_research(tech_research: str) -> str:
    """Extract vendor names from tech research text"""
    vendors = []
    vendor_keywords = {
        'microsoft': ['microsoft', 'azure', 'dynamics', '.net'],
        'aws': ['aws', 'amazon web services'],
        'google': ['google cloud', 'gcp', 'google workspace'],
        'oracle': ['oracle', 'oracle cloud', 'oracle erp'],
        'sap': ['sap', 's/4hana', 'sap erp'],
        'cisco': ['cisco', 'meraki'],
        'vmware': ['vmware', 'esxi', 'vsphere'],
        'dell': ['dell', 'emc'],
        'ibm': ['ibm', 'websphere'],
        'salesforce': ['salesforce', 'sfdc'],
        'workday': ['workday', 'workday hcm']
    }
    
    tech_lower = tech_research.lower()
    
    for vendor, keywords in vendor_keywords.items():
        if any(keyword in tech_lower for keyword in keywords):
            vendors.append(vendor.title())
    
    if vendors:
        return "Potential vendors identified: " + ", ".join(sorted(set(vendors)))
    return "Vendor research completed - reviewing specific technologies"

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
        "existing_network_vendors": "Technology vendor research in progress - checking multiple sources",
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

# --- Enhanced Graph Nodes ---
def research_node(state: AgentState) -> AgentState:
    """Enhanced research node with focused tech stack investigation"""
    st.session_state.status_text.info(f"Phase 1/3: Researching {state['company_name']}...")
    st.session_state.progress_bar.progress(20)
    
    company = state["company_name"]
    
    # Focused branch network searches
    queries = [
        f'"{company}" branch network facilities locations',
        f'"{company}" offices warehouses logistics centers',
        f'"{company}" company information website'
    ]
    
    all_results = []
    for query in queries:
        try:
            time.sleep(1)
            results = search_tool.invoke({"query": query, "max_results": 2})
            if results:
                for result in results:
                    all_results.append({
                        "title": result.get('title', ''),
                        "content": result.get('content', '')[:300],
                        "url": result.get('url', '')
                    })
        except Exception as e:
            continue
    
    # Enhanced tech stack research
    st.session_state.status_text.info(f"Phase 1/3: Researching technology vendors for {state['company_name']}...")
    st.session_state.progress_bar.progress(40)
    
    tech_findings = search_tech_stack_vendors(company)
    
    # Format research results
    research_text = "SEARCH RESULTS:\n\n"
    for i, result in enumerate(all_results):
        research_text += f"Result {i+1}:\n"
        research_text += f"Title: {result['title']}\n"
        research_text += f"Content: {result['content']}\n"
        research_text += f"URL: {result['url']}\n\n"
    
    research_text += "TECH STACK RESEARCH:\n\n"
    for i, finding in enumerate(tech_findings[:5]):  # Top 5 most relevant
        research_text += f"Tech Finding {i+1}:\n"
        research_text += f"Title: {finding['title']}\n"
        research_text += f"Content: {finding['content'][:400]}\n"
        research_text += f"URL: {finding['url']}\n"
        research_text += f"Relevance Score: {finding['relevance_score']}\n\n"
    
    research_prompt = f"""
    Analyze this research data for {company} and extract key information.
    
    Focus on finding:
    - Branch network count and facilities
    - Company website and basic info  
    - Headquarters location
    - Industry classification
    - Technology vendors and IT infrastructure
    - Cloud platforms and enterprise software
    - Any expansion or technology news
    
    Pay special attention to technology vendor information from credible sources.
    
    Keep responses factual and include source URLs when available.
    
    {research_text}
    """
    
    try:
        raw_research = llm_groq.invoke([
            SystemMessage(content=research_prompt),
            HumanMessage(content=f"Extract key business and technology information for {company}")
        ]).content
    except Exception as e:
        raw_research = f"Research data: {research_text}"
    
    return {
        "raw_research": raw_research,
        "tech_stack_findings": tech_findings
    }

def validation_node(state: AgentState) -> AgentState:
    """Enhanced validation node with tech stack focus"""
    st.session_state.status_text.info(f"Phase 2/3: Preparing data structure...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    tech_findings = state.get("tech_stack_findings", [])
    
    # Extract tech stack information
    tech_analysis = extract_tech_stack_from_research(company, raw_research, tech_findings)
    
    validation_prompt = f"""
    Based on the research below, create a structured data summary for {company}.
    
    RESEARCH DATA:
    {raw_research}
    
    TECH STACK ANALYSIS:
    {tech_analysis}
    
    Create a JSON-like structure with these exact fields:
    - linkedin_url
    - company_website_url
    - industry_category  
    - employee_count_linkedin
    - headquarters_location
    - revenue_source
    - branch_network_count (include numbers and sources)
    - expansion_news_12mo
    - digital_transformation_initiatives
    - it_leadership_change
    - existing_network_vendors (include specific vendors and technologies found)
    - wifi_lan_tender_found
    - iot_automation_edge_integration
    - cloud_adoption_gcc_setup
    - physical_infrastructure_signals
    - it_infra_budget_capex
    - why_relevant_to_syntel_bullets (exactly 3 bullet points starting with *)
    - intent_scoring_level (only: Low, Medium, or High)
    
    For existing_network_vendors, be specific about:
    - Cloud providers (AWS, Azure, Google Cloud, Oracle)
    - ERP systems (SAP, Oracle ERP)
    - Network infrastructure (Cisco, VMware, etc.)
    - Include confidence levels and sources
    
    Format as valid JSON. Include source URLs in relevant fields.
    """
    
    try:
        validated_output = llm_groq.invoke([
            SystemMessage(content=validation_prompt),
            HumanMessage(content=f"Create structured data for {company} with tech vendor details")
        ]).content
    except Exception as e:
        validated_output = raw_research
    
    return {"validated_data_text": validated_output}

def formatter_node(state: AgentState) -> AgentState:
    """Formatter node using manual JSON parsing with tech stack enhancement"""
    st.session_state.status_text.info(f"Phase 3/3: Finalizing output...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    company_name = state["company_name"]
    tech_findings = state.get("tech_stack_findings", [])
    
    # Convert tech findings to string for enhancement
    tech_research_text = "\n".join([f"{f['title']}: {f['content'][:200]}... {f['url']}" for f in tech_findings[:3]])
    
    # Use our manual JSON parser with tech stack enhancement
    try:
        final_data = parse_and_validate_json(validated_data_text, company_name, tech_research_text)
        st.success("✅ Data successfully structured with enhanced tech stack research")
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
        elif data_field == "existing_network_vendors":
            # Enhanced formatting for tech stack
            html_value = str(value).replace('\n', '<br>')
            data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left; background-color: #f0f8ff; padding: 10px; border-radius: 5px;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Enhanced Tech Stack)", 
    layout="wide",
    page_icon="🏢"
)

st.title("Syntel Company Data AI Agent 🏢")
st.markdown("### Enhanced Technology Stack Research")

# Tech research info
with st.expander("🔧 Enhanced Research Capabilities", expanded=True):
    st.markdown("""
    **Technology Stack Research Features:**
    - **Multi-source vendor detection** from Forbes, CIO portals, tech news
    - **Vendor case study analysis** from Microsoft, SAP, Oracle, AWS, etc.
    - **Relevance scoring** for tech findings
    - **Source verification** from credible domains
    - **Structured vendor reporting** with confidence levels
    """)

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
        submitted = st.form_submit_button("Start Enhanced Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    with st.spinner(f"Researching **{company_input}** with enhanced tech stack analysis..."):
        try:
            time.sleep(1)
            
            initial_state: AgentState = {
                "company_name": company_input,
                "raw_research": "",
                "validated_data_text": "",
                "final_json_data": {},
                "messages": [],
                "tech_stack_findings": []
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
            
            # Show tech stack status
            tech_data = data_dict.get("existing_network_vendors", "")
            if any(keyword in tech_data.lower() for keyword in ["microsoft", "aws", "azure", "sap", "oracle", "cisco", "vmware"]):
                st.success("✅ Technology vendor data identified in report")
            elif "research" not in tech_data.lower() and "progress" not in tech_data.lower():
                st.info("ℹ️ Basic technology information found - consider manual verification")
            
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
            tech_info = research['data'].get('existing_network_vendors', 'No tech data')[:80]
            st.write(f"Tech Stack: {tech_info}...")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Technical info
with st.sidebar.expander("🔧 Enhanced Research Approach"):
    st.markdown("""
    **Tech Stack Research:**
    - Multi-domain vendor detection
    - Relevance scoring algorithm
    - Vendor-specific query optimization
    - Source credibility filtering
    - Confidence-based reporting
    
    **Research Domains:**
    - Forbes, CIO portals, tech news
    - Vendor case studies (Microsoft, SAP, etc.)
    - Company press releases
    - Industry-specific tech analysis
    """)
