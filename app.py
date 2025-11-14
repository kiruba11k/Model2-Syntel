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
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)  # Reduced from 7 to 3
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Domain List for Branch Network Search ---
BRANCH_SEARCH_DOMAINS = [
    "sany.in", "welspunone.com", "allcargologistics.com", 
    "tatamotors.com", "starhealth.in", "hdfcbank.com",
    "linkedin.com", "mca.gov.in", "economictimes.com"
]  # Reduced to most relevant domains

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

# --- Optimized Search Functions ---
def perform_optimized_branch_search(company_name: str) -> str:
    """Perform optimized real-time search for branch network data"""
    
    st.session_state.status_text.info(f"🔍 Performing optimized branch network search for {company_name}...")
    
    # Highly targeted queries only
    essential_queries = [
        f'"{company_name}" "branch network" "facilities count"',
        f'"{company_name}" "number of branches" "locations"',
        f'"{company_name}" "retail branches" "service centers"',
        f'"{company_name}" "expansion" "new branches"'
    ]
    
    all_results = []
    
    for i, query in enumerate(essential_queries):
        try:
            # Add small delay to avoid rate limits
            if i > 0:
                time.sleep(1)
                
            results = search_tool.invoke({"query": query, "max_results": 2})  # Reduced results
            
            if results:
                query_summary = f"Query: {query}\n"
                for j, result in enumerate(results[:2]):  # Only first 2 results
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')[:300]  # Reduced content length
                    url = result.get('url', 'No URL')
                    
                    query_summary += f"Result {j+1}: {title} | {content} | URL: {url}\n"
                
                all_results.append(query_summary)
                
        except Exception as e:
            continue
    
    return "\n".join(all_results)

# --- Graph Nodes (Optimized for Token Usage) ---
def research_node(state: AgentState) -> AgentState:
    """Optimized research node with reduced token usage"""
    st.session_state.status_text.info(f"Phase 1/3: Conducting optimized search for {state['company_name']}...")
    st.session_state.progress_bar.progress(25)
    
    company = state["company_name"]
    
    # Perform optimized branch network search
    branch_network_data = perform_optimized_branch_search(company)
    
    # Minimal general searches
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
    
    # Combined research with length limits
    combined_research = f"""
    BRANCH NETWORK DATA:
    {branch_network_data[:1500]}
    
    GENERAL BUSINESS DATA:
    {general_search_results[:1000]}
    """
    
    # Shorter, more focused prompt
    research_prompt = f"""
    Research {company} and extract key business intelligence.
    
    FOCUS: Find branch network count with source links.
    
    DATA:
    {combined_research}
    
    Extract: branch counts, locations, facilities. Include source URLs.
    Keep responses concise.
    """
    
    raw_research = llm_groq.invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"Research {company} focusing on branch network data.")
    ]).content

    return {"raw_research": raw_research}

def validation_node(state: AgentState) -> AgentState:
    """Optimized validation node"""
    st.session_state.status_text.info(f"Phase 2/3: Validating data...")
    st.session_state.progress_bar.progress(60)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    # Shorter validation prompt
    validation_prompt = f"""
    Validate this research data for {company}:
    
    {raw_research[:2000]}
    
    Syntel Expertise: {SYNTEL_EXPERTISE[:500]}
    
    Steps:
    1. Verify branch network data has numbers and sources
    2. Score intent: Low/Medium/High
    3. Create 3 relevance bullet points
    
    Output clean key-value data.
    """
    
    validated_output = llm_groq.invoke([
        SystemMessage(content=validation_prompt),
        HumanMessage(content=f"Validate data for {company}")
    ]).content

    return {"validated_data_text": validated_output}

def formatter_node(state: AgentState) -> AgentState:
    """Optimized formatter node"""
    st.session_state.status_text.info(f"Phase 3/3: Creating final JSON...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    
    # Concise formatting prompt
    formatting_prompt = f"""
    Convert to JSON using CompanyData schema.
    
    Data:
    {validated_data_text[:1500]}
    
    Requirements:
    - All fields must be present
    - Branch network must have numbers and sources
    - Intent: Low/Medium/High only
    - 3 bullet points for relevance
    
    Output ONLY JSON.
    """

    try:
        final_pydantic_object = llm_groq.with_structured_output(CompanyData).invoke([
            SystemMessage(content=formatting_prompt),
            HumanMessage(content="Create JSON output.")
        ])
        return {"final_json_data": final_pydantic_object.dict()}
    except Exception as e:
        st.error(f"JSON formatting failed: {e}")
        # Return minimal valid data structure
        minimal_data = {
            "linkedin_url": "Not found",
            "company_website_url": "Not found", 
            "industry_category": "Not found",
            "employee_count_linkedin": "Not found",
            "headquarters_location": "Not found",
            "revenue_source": "Not found",
            "branch_network_count": "Data unavailable - API limit",
            "expansion_news_12mo": "Not found",
            "digital_transformation_initiatives": "Not found",
            "it_leadership_change": "Not found",
            "existing_network_vendors": "Not found",
            "wifi_lan_tender_found": "Not found",
            "iot_automation_edge_integration": "Not found",
            "cloud_adoption_gcc_setup": "Not found",
            "physical_infrastructure_signals": "Not found",
            "it_infra_budget_capex": "Not found",
            "why_relevant_to_syntel_bullets": "* Data limited due to API constraints\n* Manual verification recommended\n* Check company websites directly",
            "intent_scoring_level": "Low"
        }
        return {"final_json_data": minimal_data}

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
def format_data_for_display(company_input: str, validated_data: CompanyData) -> pd.DataFrame:
    """Transforms the Pydantic model into a 2-column DataFrame"""
    data_dict = validated_data.dict()
    
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
    for display_col, pydantic_field in mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(pydantic_field, "N/A")
        
        if pydantic_field == "why_relevant_to_syntel_bullets":
            html_value = value.replace('\n', '<br>').replace('*', '•')
            data_list.append({"Column Header": display_col, "Value with Source Link": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value with Source Link": value})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Optimized)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent 🏢")
st.markdown("### Optimized for API Limits - Branch Network Focus")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

# API limit warning
st.warning("🚨 **Running in optimized mode due to Groq API limits** - Some data may be limited")

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
    
    with st.spinner(f"Running optimized research for **{company_input}**..."):
        try:
            # Add initial delay
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
            validated_data = CompanyData(**data_dict) 
            
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"Research Complete for {company_input}!")
            
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": validated_data.dict()
            }
            st.session_state.research_history.append(research_entry)
            
            # Display results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, validated_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Download options
            st.subheader("Download Options 💾")
            download_df = format_data_for_display(company_input, validated_data)
            download_df['Why Relevant to Syntel'] = validated_data.why_relevant_to_syntel_bullets
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(validated_data.dict(), indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = download_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(download_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("💡 **Tip**: Try a simpler company name or wait a minute before retrying.")

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

# Optimizations info
with st.sidebar.expander("🚀 Optimization Details"):
    st.markdown("""
    **Applied Optimizations:**
    - Reduced search queries from 10+ to 4 essential ones
    - Limited results from 7 to 2 per query
    - Shortened content length from 500 to 300 chars
    - Reduced domains from 25+ to 9 most relevant
    - Added API rate limiting delays
    - Streamlined prompts to reduce token usage
    - Added fallback error handling
    
    **Current Limits:**
    - Groq TPM: 6,000 tokens
    - Tavily results: 2-3 per query
    - Content length: 300 chars max
    - Total searches: 4-6 per company
    """)
