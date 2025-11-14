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
    branch_network_count: str = Field(description="Number of branches/facilities. MUST include the SOURCE/LINK.")
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

# --- Syntel Core Offerings for Analysis Node ---
SYNTEL_EXPERTISE = """
Syntel (now Atos Syntel/Eviden) specializes in:
1. IT Automation/RPA: Via its proprietary platform, SyntBots.
2. Digital Transformation: Through the Digital One suite (Mobility, IoT, AI, Cloud, Microservices).
3. Cloud & Infrastructure: Offering Cloud Computing, IT Infrastructure Management, and Application Modernization.
4. KPO/BPO: Strong track record in Knowledge Process Outsourcing and Industry-specific BPO solutions.
"""

# --- Robust JSON Formatter as Fallback ---
def create_fallback_json(company_name: str, raw_data: str) -> dict:
    """Create a fallback JSON structure when structured output fails"""
    
    # Extract basic information from raw data using regex
    branch_match = re.search(r'(\d+)\s*(?:branches|locations|facilities)', raw_data, re.IGNORECASE)
    branch_count = branch_match.group(1) if branch_match else "Unknown"
    
    location_match = re.search(r'(?:headquarters|located in|based in)[:\s]*([A-Za-z,\s]+)', raw_data, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else "Unknown"
    
    # Create sensible fallback data
    return {
        "linkedin_url": f"Search for {company_name} on LinkedIn",
        "company_website_url": f"Search for {company_name} official website",
        "industry_category": "To be determined from further research",
        "employee_count_linkedin": "Check LinkedIn for current data",
        "headquarters_location": location,
        "revenue_source": "Check annual reports or investor relations",
        "branch_network_count": f"{branch_count} facilities found. Sources: Various search results",
        "expansion_news_12mo": "Review recent news articles for expansion details",
        "digital_transformation_initiatives": "Research IT and digital projects",
        "it_leadership_change": "Check recent executive announcements",
        "existing_network_vendors": "Research technology partnerships",
        "wifi_lan_tender_found": "Check government and corporate tender portals",
        "iot_automation_edge_integration": "Research automation initiatives",
        "cloud_adoption_gcc_setup": "Investigate cloud infrastructure projects",
        "physical_infrastructure_signals": "Monitor new facility announcements",
        "it_infra_budget_capex": "Review financial reports for IT spending",
        "why_relevant_to_syntel_bullets": f"* {company_name} has physical infrastructure that may need IT modernization\n* Potential for digital transformation initiatives\n* Opportunity for infrastructure management services",
        "intent_scoring_level": "Medium"
    }

# --- Graph Nodes (With Robust Error Handling) ---
def research_node(state: AgentState) -> AgentState:
    """Research node with simplified approach"""
    st.session_state.status_text.info(f"Phase 1/3: Researching {state['company_name']}...")
    st.session_state.progress_bar.progress(25)
    
    company = state["company_name"]
    
    # Simple targeted searches
    queries = [
        f'"{company}" "branch network" "facilities"',
        f'"{company}" "locations" "offices"',
        f'"{company}" "annual report" "company"'
    ]
    
    all_results = []
    for query in queries:
        try:
            time.sleep(1)  # Rate limiting
            results = search_tool.invoke({"query": query, "max_results": 2})
            if results:
                for result in results:
                    all_results.append(f"Title: {result.get('title', '')}")
                    all_results.append(f"Content: {result.get('content', '')[:200]}")
                    all_results.append(f"URL: {result.get('url', '')}")
                    all_results.append("---")
        except Exception as e:
            all_results.append(f"Search failed: {str(e)}")
    
    raw_research = "\n".join(all_results)
    
    research_prompt = f"""
    Extract key business information for {company} from this research:
    
    {raw_research}
    
    Focus on finding:
    1. Branch network or facility count
    2. Headquarters location  
    3. Basic company information
    4. Any expansion or technology news
    
    Keep notes concise and include source URLs.
    """
    
    try:
        raw_research = llm_groq.invoke([
            SystemMessage(content=research_prompt),
            HumanMessage(content=f"Extract key info for {company}")
        ]).content
    except Exception as e:
        raw_research = f"Research limited due to API constraints. Raw data: {raw_research}"
    
    return {"raw_research": raw_research}

def validation_node(state: AgentState) -> AgentState:
    """Simplified validation node"""
    st.session_state.status_text.info(f"Phase 2/3: Validating data...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    validation_prompt = f"""
    Create a clean summary for {company} using this data:
    
    {raw_research}
    
    Format as key-value pairs for these categories:
    - Basic company info (website, LinkedIn, industry, employees, headquarters, revenue)
    - Branch network count with sources
    - Recent expansion news
    - Technology and digital initiatives
    - Infrastructure details
    - Relevance to IT services company
    
    For branch network: include specific numbers and source URLs.
    For relevance: create 3 bullet points about IT service opportunities.
    For intent: score as Low, Medium, or High.
    """
    
    try:
        validated_output = llm_groq.invoke([
            SystemMessage(content=validation_prompt),
            HumanMessage(content=f"Create summary for {company}")
        ]).content
    except Exception as e:
        validated_output = f"Validation limited. Using raw data: {raw_research[:1000]}"
    
    return {"validated_data_text": validated_output}

def formatter_node(state: AgentState) -> AgentState:
    """Formatter node with robust JSON handling"""
    st.session_state.status_text.info(f"Phase 3/3: Creating final output...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    company_name = state["company_name"]
    
    # Try structured output first
    try:
        formatting_prompt = f"""
        Create JSON output for company data using EXACTLY these fields:
        
        Fields required:
        - linkedin_url
        - company_website_url  
        - industry_category
        - employee_count_linkedin
        - headquarters_location
        - revenue_source
        - branch_network_count
        - expansion_news_12mo
        - digital_transformation_initiatives
        - it_leadership_change
        - existing_network_vendors
        - wifi_lan_tender_found
        - iot_automation_edge_integration
        - cloud_adoption_gcc_setup
        - physical_infrastructure_signals
        - it_infra_budget_capex
        - why_relevant_to_syntel_bullets (exactly 3 bullet points with *)
        - intent_scoring_level (only: Low, Medium, or High)
        
        Data to use:
        {validated_data_text[:1500]}
        
        Rules:
        - No extra fields
        - No duplicate fields
        - All fields must be strings
        - intent_scoring_level only: Low, Medium, or High
        - why_relevant_to_syntel_bullets must have exactly 3 bullet points starting with *
        - Include source URLs where possible
        
        Output ONLY valid JSON.
        """
        
        final_pydantic_object = llm_groq.with_structured_output(CompanyData).invoke([
            SystemMessage(content=formatting_prompt),
            HumanMessage(content="Create valid JSON output")
        ])
        return {"final_json_data": final_pydantic_object.dict()}
        
    except Exception as e:
        st.warning(f"Structured output failed, using fallback: {str(e)}")
        # Use fallback JSON creation
        fallback_data = create_fallback_json(company_name, validated_data_text)
        return {"final_json_data": fallback_data}

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

# --- Helper Function for Display ---
def format_data_for_display(company_input: str, validated_data: dict) -> pd.DataFrame:
    """Transforms the data into a 2-column DataFrame"""
    
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
            value = validated_data.get(data_field, "Data not available")
        
        if data_field == "why_relevant_to_syntel_bullets":
            html_value = str(value).replace('\n', '<br>').replace('*', '•')
            data_list.append({"Column Header": display_col, "Value with Source Link": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value with Source Link": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Robust)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent 🏢")
st.markdown("### Robust Version with Fallback Handling")

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
    
    with st.spinner(f"Researching **{company_input}**..."):
        try:
            time.sleep(1)  # Initial delay
            
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
            st.info("Try a different company name or check API limits.")

st.markdown("---")

# Research History
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Error handling info
with st.sidebar.expander("🛡️ Robust Features"):
    st.markdown("""
    **Error Handling:**
    - Fallback JSON generation when structured output fails
    - Regex-based data extraction as backup
    - Graceful degradation under API limits
    - Duplicate field prevention
    - Invalid JSON recovery
    
    **Current Status:**
    - Structured output with fallback
    - Branch network search active
    - Real-time domain searching
    - Source link inclusion
    """)
