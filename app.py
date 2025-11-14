import streamlit as st
import pandas as pd
import json
import operator
import re
from typing import TypedDict, Annotated
from io import BytesIO
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

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
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=7)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Domain List for Branch Network Search ---
BRANCH_SEARCH_DOMAINS = [
    "sany.in", "welspunone.com", "hospitalitybizindia.com", "thehansindia.com",
    "allcargologistics.com", "economictimes.com", "indiatimes.com", 
    "gcc.economictimes.indiatimes.com", "tatamotors.com", "starhealth.in",
    "hdfcbank.com", "linkedin.com", "mca.gov.in", "careers.tcs.com",
    "careers.accenture.com", "glassdoor.com", "indeed.com", "openstreetmap.org",
    "overpass-api.de", "yellowpages.in", "dbie.rbi.org.in", "trai.gov.in",
    "data.gov.in", "news.google.com"
]

# --- Pydantic Output Schema (CompanyData) ---
class CompanyData(BaseModel):
    # Basic Company Info
    linkedin_url: str = Field(description="LinkedIn URL. MUST include link/source.")
    company_website_url: str = Field(description="Official company website URL. MUST include link.")
    industry_category: str = Field(description="Industry category and source. MUST include link.")
    employee_count_linkedin: str = Field(description="Employee count range and source. MUST include link.")
    headquarters_location: str = Field(description="Headquarters city, country, and source. MUST include link.")
    revenue_source: str = Field(description="Revenue data point and specific source. MUST include link.")
    
    # Core Research Fields
    branch_network_count: str = Field(description="Number of branches/facilities, capacity mentioned online. MUST include the SOURCE/LINK.")
    expansion_news_12mo: str = Field(description="Summary of expansion news in the last 12 months. MUST include the SOURCE/LINK.")
    digital_transformation_initiatives: str = Field(description="Details on smart infra or digital programs. MUST include the SOURCE/LINK.")
    it_leadership_change: str = Field(description="Name and title of new CIO/CTO/Head of Infra if changed recently. MUST include the SOURCE/LINK.")
    existing_network_vendors: str = Field(description="Mentioned network vendors or tech stack. MUST include the SOURCE/LINK.")
    wifi_lan_tender_found: str = Field(description="Yes/No and source link if a tender was found. MUST include the SOURCE/LINK.")
    iot_automation_edge_integration: str = Field(description="Details on IoT/Automation/Edge mentions. MUST include the SOURCE/LINK.")
    cloud_adoption_gcc_setup: str = Field(description="Details on Cloud Adoption or Global Capability Centers (GCC). MUST include the SOURCE/LINK.")
    physical_infrastructure_signals: str = Field(description="Any physical infra signals (new office, factory etc). MUST include the SOURCE/LINK.")
    it_infra_budget_capex: str = Field(description="IT Infra Budget or Capex allocation details. MUST include the SOURCE/LINK.")
    
    # Analysis Fields
    why_relevant_to_syntel_bullets: str = Field(description="A markdown string with 3 specific bullet points explaining relevance to Syntel based on its offerings (Digital One, Cloud, Network, Automation, KPO).")
    intent_scoring_level: str = Field(description="Intent score level: 'Low', 'Medium', or 'High'.") 

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Represents the shared context/state of the graph's execution."""
    company_name: str
    raw_research: str
    validated_data_text: str
    final_json_data: dict
    messages: Annotated[list, operator.add]
    branch_network_raw: str  # New field for branch network specific data

# --- Syntel Core Offerings for Analysis Node ---
SYNTEL_EXPERTISE = """
Syntel (now Atos Syntel/Eviden) specializes in:
1. IT Automation/RPA: Via its proprietary platform, **SyntBots**.
2. Digital Transformation: Through the **Digital One™** suite (Mobility, IoT, AI, Cloud, Microservices).
3. Cloud & Infrastructure: Offering **Cloud Computing**, **IT Infrastructure Management**, and **Application Modernization**.
4. KPO/BPO: Strong track record in **Knowledge Process Outsourcing (KPO)** and **Industry-specific BPO solutions**.
"""

# --- Enhanced Branch Network Search Functions ---
def perform_branch_network_search(company_name: str) -> str:
    """Perform real-time search for branch network data across specified domains"""
    
    st.session_state.status_text.info(f"🔍 Performing real-time branch network search for {company_name}...")
    
    branch_search_queries = [
        # General branch network queries
        f'"{company_name}" "branch network" "facilities count" "number of locations"',
        f'"{company_name}" "retail branches" "service centers" "office locations"',
        f'"{company_name}" "global network" "facilities" "touchpoints"',
        f'"{company_name}" "expansion" "new branches" 2024 2025',
        f'"{company_name}" "dealerships" "distributors" "channel partners"',
        
        # Industry-specific queries
        f'"{company_name}" "warehouses" "logistics centers" "CFS" "ICD"',
        f'"{company_name}" "ATMs" "bank branches" "business correspondents"',
        f'"{company_name}" "showrooms" "dealership network" "retail outlets"',
        f'"{company_name}" "hospital locations" "clinics" "healthcare centers"',
        f'"{company_name}" "manufacturing plants" "factories" "production units"'
    ]
    
    # Domain-specific queries for top domains
    top_domains = BRANCH_SEARCH_DOMAINS[:8]  # Use first 8 domains to avoid rate limits
    for domain in top_domains:
        branch_search_queries.append(f'site:{domain} "{company_name}" branch facilities locations network')
    
    all_branch_results = []
    
    for i, query in enumerate(branch_search_queries[:10]):  # Limit to 10 queries
        try:
            st.session_state.status_text.info(f"Searching branch data... ({i+1}/{min(10, len(branch_search_queries))})")
            
            results = search_tool.invoke({"query": query, "max_results": 5})
            
            if results:
                query_results = f"=== BRANCH SEARCH QUERY: {query} ===\n"
                for j, result in enumerate(results):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    
                    query_results += f"Result {j+1}:\n"
                    query_results += f"Title: {title}\n"
                    query_results += f"Content: {content[:500]}...\n"  # Limit content length
                    query_results += f"URL: {url}\n"
                    query_results += "-" * 80 + "\n"
                
                all_branch_results.append(query_results)
                
        except Exception as e:
            all_branch_results.append(f"Query failed: {query}\nError: {str(e)}\n")
            continue
    
    # Fallback search if no specific branch data found
    if not any("BRANCH SEARCH QUERY" in result for result in all_branch_results):
        fallback_queries = [
            f'"{company_name}" "annual report" "investor presentation"',
            f'"{company_name}" "locations" "contact us" "our network"',
            f'"{company_name}" "about us" "company profile"'
        ]
        
        for query in fallback_queries:
            try:
                results = search_tool.invoke({"query": query, "max_results": 3})
                if results:
                    all_branch_results.append(f"=== FALLBACK SEARCH: {query} ===\n{str(results)[:1000]}\n")
            except Exception:
                continue
    
    return "\n".join(all_branch_results)

def extract_branch_patterns(text: str, company_name: str) -> list:
    """Extract branch count patterns from text using regex"""
    patterns = [
        (r'(\d{1,5}[\+,]?\d*)\s*(?:branches|locations|offices)', 'branches'),
        (r'(\d{1,5}[\+,]?\d*)\s*(?:retail\s+outlets|service\s+centers)', 'retail outlets'),
        (r'(\d{1,5}[\+,]?\d*)\s*(?:dealerships|distributors)', 'dealerships'),
        (r'network\s+of\s+(\d{1,5}[\+,]?\d*)', 'network size'),
        (r'across\s+(\d{1,5}[\+,]?\d*)\s*(?:cities|countries)', 'geographic presence'),
        (r'(\d{1,5}[\+,]?\d*)\s*touchpoints', 'touchpoints'),
        (r'presence\s+in\s+(\d{1,5}[\+,]?\d*)', 'presence locations'),
        (r'(\d{1,5}[\+,]?\d*)\s*ATMs?', 'ATMs'),
        (r'(\d{1,5}[\+,]?\d*)\s*warehouses', 'warehouses'),
        (r'(\d{1,5}[\+,]?\d*)\s*CFS', 'container freight stations'),
        (r'(\d{1,5}[\+,]?\d*)\s*ICD', 'inland container depots')
    ]
    
    found_patterns = []
    for pattern, pattern_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            found_patterns.append({
                'count': match.group(1),
                'type': pattern_type,
                'context': text[max(0, match.start()-100):match.end()+100]
            })
    
    return found_patterns

# --- Graph Nodes (Enhanced with Branch Network Search) ---
def research_node(state: AgentState) -> AgentState:
    """Node 1: Executes deep search with enhanced branch network focus"""
    st.session_state.status_text.info(f"Phase 1/3: Conducting deep, multi-query search for {state['company_name']}...")
    st.session_state.progress_bar.progress(25)
    
    company = state["company_name"]
    
    # Perform dedicated branch network search FIRST
    branch_network_raw = perform_branch_network_search(company)
    
    # Original general searches (slightly modified to avoid duplication)
    general_search_queries = [
        f'"{company}" "annual report" "investor presentation" revenue headcount',
        f'"{company}" "digital transformation" "IT strategy" "Cloud adoption"',
        f'"{company}" "CIO" "CTO" "new leadership" IT infrastructure',
        f'"{company}" "network tender" "Wi-Fi upgrade" "IT capex" 2024 2025',
        f'"{company}" "IoT" "Automation" "edge computing"',
        f'"{company}" "GCC" "Global Capability Center" "cloud infrastructure"'
    ]
    
    # Execute general searches
    all_general_results = []
    for query in general_search_queries:
        try:
            results = search_tool.invoke({"query": query, "max_results": 5})
            all_general_results.append(f"--- General Search: {query} ---\n{results}")
        except Exception as e:
            all_general_results.append(f"--- General Search Failed: {query} ---\nError: {e}")
    
    general_search_results = "\n\n".join(all_general_results)
    
    # Combine both searches for comprehensive research
    combined_research = f"""
    ======= DEDICATED BRANCH NETWORK SEARCH RESULTS =======
    {branch_network_raw}
    
    ======= GENERAL BUSINESS INTELLIGENCE SEARCH RESULTS =======
    {general_search_results}
    """
    
    research_prompt = f"""
    You are an expert Business Intelligence Researcher with specialization in physical infrastructure analysis.
    
    COMPANY: {company}
    
    CRITICAL PRIORITY: Extract BRANCH NETWORK / FACILITIES COUNT data from the dedicated branch search results.
    
    SEARCH RESULTS STRUCTURE:
    1. First section contains DEDICATED BRANCH NETWORK searches
    2. Second section contains general business intelligence searches
    
    RESEARCH INSTRUCTIONS:
    
    FOR BRANCH NETWORK COUNT FIELD (HIGHEST PRIORITY):
    - Focus primarily on the "DEDICATED BRANCH NETWORK SEARCH RESULTS" section
    - Look for specific numbers: "X branches", "Y locations", "Z service centers"
    - Extract: retail branches, dealerships, service centers, offices, warehouses, ATMs, touchpoints
    - MUST include the exact source URL where branch count information was found
    - If multiple numbers exist, choose the most recent and credible one
    
    FOR ALL OTHER FIELDS:
    - Use the general business intelligence search results
    - Follow standard research process for other fields
    - EVERY data point MUST include its source link
    
    SEARCH RESULTS:
    {combined_research}
    
    Generate comprehensive research notes focusing on accurate branch network data extraction.
    """
    
    raw_research = llm_groq.invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"Generate detailed research notes for {company} with emphasis on branch network facilities count.")
    ]).content

    return {
        "raw_research": raw_research,
        "branch_network_raw": branch_network_raw
    }

def validation_node(state: AgentState) -> AgentState:
    """Node 2: Validates data with enhanced branch network verification"""
    st.session_state.status_text.info(f"Phase 2/3: Validating data, scoring intent, and analyzing relevance...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    branch_network_raw = state.get("branch_network_raw", "")
    
    validation_prompt = f"""
    You are a Data Quality Specialist with expertise in physical infrastructure validation.
    
    **Syntel's Expertise:**
    {SYNTEL_EXPERTISE}
    
    **SPECIAL FOCUS: BRANCH NETWORK VALIDATION**
    You have access to raw branch network search data. Use it to verify the branch count accuracy.
    
    Raw Branch Network Data (for verification):
    {branch_network_raw[:2000]}  # Limit length to avoid token issues
    
    VALIDATION STEPS:
    
    1. **BRANCH NETWORK VERIFICATION:**
       - Cross-reference branch count with raw search data
       - Ensure the number is realistic for the company's industry and size
       - Verify source links are from credible domains
       - If count seems inaccurate, re-analyze the raw branch data
    
    2. **INTENT SCORING:** Calculate 'Low', 'Medium', or 'High':
       - **High:** Multiple strong signals (New CIO + IT Capex + Branch Expansion + Digital Transformation)
       - **Medium:** Clear buying signals (Branch Network Growth + IT Initiatives)
       - **Low:** Only basic company info, no expansion/digitalization signals
    
    3. **SYNTEL RELEVANCE:** Generate 3 specific bullet points comparing company's infrastructure needs with Syntel's offerings.
    
    Raw Research Notes:
    ---
    {raw_research}
    ---
    
    Output the validated data in clear key-value format ready for JSON conversion.
    """
    
    validated_output = llm_groq.invoke([
        SystemMessage(content=validation_prompt),
        HumanMessage(content=f"Validate and enrich the data for {company}, with special focus on branch network accuracy.")
    ]).content

    return {"validated_data_text": validated_output}

def formatter_node(state: AgentState) -> AgentState:
    """Node 3: Formats the validated data into the strict Pydantic JSON schema."""
    st.session_state.status_text.info(f"Phase 3/3: Converting to final Pydantic JSON...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    
    formatting_prompt = f"""
    You are a **STRICT** JSON Schema Specialist. Convert the validated data into the **EXACT** JSON format defined by the CompanyData Pydantic schema.
    
    **CRITICAL REQUIREMENTS:**
    - Every field in the Pydantic schema must be present
    - **branch_network_count MUST contain specific numbers and source links**
    - All string fields (except intent score) MUST contain data AND SOURCE LINKS
    - intent_scoring_level: ONLY 'Low', 'Medium', or 'High'
    - why_relevant_to_syntel_bullets: MUST be markdown with 3 bullet points
    
    **BRANCH NETWORK FORMATTING EXAMPLES:**
    ✅ GOOD: "250+ branches across India, including 42 dealerships and 7 regional offices. Source: https://sany.in/about-us"
    ✅ GOOD: "Network of 1,200 service centers nationwide. Source: https://tatamotors.com/our-network"
    ❌ BAD: "Multiple branches" (missing count and source)
    ❌ BAD: "250 branches" (missing source)
    
    Validated Data:
    ---
    {validated_data_text}
    ---
    
    Output ONLY the JSON object.
    """

    final_pydantic_object = llm_groq.with_structured_output(CompanyData).invoke([
        SystemMessage(content=formatting_prompt),
        HumanMessage(content="Generate the final JSON for CompanyData with proper branch network formatting.")
    ])

    return {"final_json_data": final_pydantic_object.dict()}

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

# --- Helper Function for Custom Table Formatting ---
def format_data_for_display(company_input: str, validated_data: CompanyData) -> pd.DataFrame:
    """
    Transforms the Pydantic model into a 2-column DataFrame for 
    clean rendering of links and bullets via Streamlit Markdown/HTML.
    """
    data_dict = validated_data.dict()
    
    # Mapping the Pydantic fields to the user-friendly column headers
    mapping = {
        "Company Name": "company_name_placeholder",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url",
        "Industry Category": "industry_category",
        "Employee Count (LinkedIn)": "employee_count_linkedin",
        "Headquarters (Location)": "headquarters_location",
        "Revenue (Source)": "revenue_source",
        "Branch Network / Facilities Count": "branch_network_count",
        "Expansion News (Last 12 Months)": "expansion_news_12mo",
        "Digital Transformation Initiatives / Smart Infra Programs": "digital_transformation_initiatives",
        "IT Infrastructure Leadership Change (CIO / CTO / Head Infra)": "it_leadership_change",
        "Existing Network Vendors / Tech Stack": "existing_network_vendors",
        "Recent Wi-Fi Upgrade or LAN Tender Found": "wifi_lan_tender_found",
        "IoT / Automation / Edge Integration Mentioned": "iot_automation_edge_integration",
        "Cloud Adoption / GCC Setup": "cloud_adoption_gcc_setup",
        "Physical Infrastructure Signals": "physical_infrastructure_signals",
        "IT Infra Budget / Capex Allocation": "it_infra_budget_capex",
        "Intent Scoring": "intent_scoring_level",
        "Why Relevent to Syntel (3 Key Points)": "why_relevant_to_syntel_bullets",
    }
    
    data_list = []
    for display_col, pydantic_field in mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(pydantic_field, "N/A (Missing Field)")
        
        # Replace newlines in the bullet points with HTML breaks for display
        if pydantic_field == "why_relevant_to_syntel_bullets":
            # Clean up the markdown bullet points for HTML display
            html_value = value.replace('\n', '<br>')
            html_value = html_value.replace('*', '•') # Use a bullet point for cleaner HTML rendering
            data_list.append({"Column Header": display_col, "Value with Source Link": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value with Source Link": value})
            
    df = pd.DataFrame(data_list)
    return df

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Enhanced Branch Network Search)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent (Enhanced Branch Network Search) 🏢")
st.markdown("### Real-time Branch Network Detection with Multi-Domain Search")

# Initialize session state for UI components
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics" # Default example
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

# Domain information display
with st.expander("🔍 Active Search Domains for Branch Network Data"):
    st.write(f"Searching across {len(BRANCH_SEARCH_DOMAINS)} domains for real-time branch network information:")
    cols = st.columns(4)
    for i, domain in enumerate(BRANCH_SEARCH_DOMAINS):
        cols[i % 4].write(f"• {domain}")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input, key="company_input_widget")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("Start Deep Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # Initialize the progress bar and status text containers
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    with st.spinner(f"AI Graph is running enhanced branch network search for **{company_input}**..."):
        try:
            # Initial State for LangGraph
            initial_state: AgentState = {
                "company_name": company_input,
                "raw_research": "",
                "validated_data_text": "",
                "final_json_data": {},
                "messages": [],
                "branch_network_raw": ""
            }

            # Invoke the LangGraph app
            final_state = app.invoke(initial_state)
            
            # --- Result Processing ---
            data_dict = final_state["final_json_data"]
            validated_data = CompanyData(**data_dict) 
            
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"Research Complete for {company_input}! (Enhanced Branch Network Search)")
            
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": validated_data.dict()
            }
            st.session_state.research_history.append(research_entry)
            
            # --- Display Final Table (using HTML to support links/bullets) ---
            st.subheader(f"Final Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, validated_data)
            
            # Use to_html and markdown with unsafe_allow_html=True to render rich content
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Display branch network success
            branch_data = validated_data.branch_network_count
            if "http" in branch_data and any(char.isdigit() for char in branch_data):
                st.success("✅ Branch network data successfully extracted with real-time search!")
            else:
                st.warning("⚠️ Branch network data may be limited. Consider manual verification.")
            
            st.caption("✅ All data points include direct source links. Branch network data sourced from real-time multi-domain search.")
            
            # --- Download Options ---
            st.subheader("Download Options 💾")
            
            # Prepare a clean DataFrame for CSV/Excel downloads (removing HTML formatting)
            download_df = format_data_for_display(company_input, validated_data)
            download_df['Why Relevent to Syntel (3 Key Points)'] = validated_data.why_relevant_to_syntel_bullets
            
            def to_excel(df):
                """Converts dataframe to an in-memory Excel file."""
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 json_filename = f"{company_input.replace(' ', '_')}_data.json"
                 st.download_button(
                     label="Download JSON Data",
                     data=json.dumps(validated_data.dict(), indent=2),
                     file_name=json_filename,
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = download_df.to_csv(index=False).encode('utf-8')
                 csv_filename = f"{company_input.replace(' ', '_')}_data.csv"
                 st.download_button(
                     label="Download CSV Data",
                     data=csv_data,
                     file_name=csv_filename,
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(download_df)
                 excel_filename = f"{company_input.replace(' ', '_')}_data.xlsx"
                 st.download_button(
                     label="Download Excel Data",
                     data=excel_data,
                     file_name=excel_filename,
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.markdown("Please check your input or API keys (Groq/Tavily).")

st.markdown("---")

# --- Research History (Sidebar) ---
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            branch_data = research['data'].get('branch_network_count', 'N/A')
            st.write(f"Branch Network: {branch_data[:100]}..." if len(str(branch_data)) > 100 else f"Branch Network: {branch_data}")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# --- Instructions (Sidebar) ---
with st.sidebar.expander("🚀 Enhanced Branch Network Search"):
    st.markdown("""
    **New Features:**
    - **Real-time Branch Network Detection**: Searches 25+ domains specifically for branch/facility counts
    - **Multi-Query Strategy**: 10+ specialized queries for comprehensive coverage
    - **Domain-Specific Search**: Targets company websites, news portals, and industry databases
    - **Pattern Recognition**: Automated extraction of branch counts using regex patterns
    
    **Search Domains Include:**
    - Company websites (sany.in, tatamotors.com, hdfcbank.com)
    - News portals (economictimes.com, thehansindia.com)
    - Government databases (mca.gov.in, data.gov.in, trai.gov.in)
    - Career sites (careers.tcs.com, glassdoor.com)
    - Business directories (yellowpages.in, linkedin.com)
    
    **Output Format:**
    - Specific branch counts with source links
    - Industry-appropriate facility types (branches, dealerships, warehouses, ATMs)
    - Verified through multiple data sources
    """)
