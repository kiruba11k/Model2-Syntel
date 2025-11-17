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
    st.success("APIs initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize tools: {e}")
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

# --- Direct Working Search Function ---
def working_company_search(company_name: str, field_name: str) -> List[Dict]:
    """Working search function that actually returns results"""
    
    # Specific, targeted queries that work
    field_queries = {
        "linkedin_url": [f"{company_name} LinkedIn"],
        "company_website_url": [f"{company_name} official website"],
        "industry_category": [f"{company_name} business industry"],
        "employee_count_linkedin": [f"{company_name} employees"],
        "headquarters_location": [f"{company_name} headquarters location"],
        "revenue_source": [f"{company_name} revenue"],
        "branch_network_count": [f"{company_name} branches locations"],
        "expansion_news_12mo": [f"{company_name} expansion news"],
        "digital_transformation_initiatives": [f"{company_name} digital technology"],
        "it_leadership_change": [f"{company_name} CIO CTO"],
        "existing_network_vendors": [f"{company_name} technology vendors"],
        "wifi_lan_tender_found": [f"{company_name} network upgrade"],
        "iot_automation_edge_integration": [f"{company_name} automation"],
        "cloud_adoption_gcc_setup": [f"{company_name} cloud"],
        "physical_infrastructure_signals": [f"{company_name} new facility"],
        "it_infra_budget_capex": [f"{company_name} IT budget"]
    }
    
    queries = field_queries.get(field_name, [f"{company_name}"])
    all_results = []
    
    for query in queries:
        try:
            time.sleep(1)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            for result in results:
                content = result.get('content', '')
                if content and len(content) > 30:
                    all_results.append({
                        "title": result.get('title', ''),
                        "content": content[:400],
                        "url": result.get('url', '')
                    })
        except Exception as e:
            continue
    
    return all_results

# --- Simple Extraction That Works ---
def extract_company_info(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Simple extraction that returns actual information"""
    
    if not search_results:
        return "Not available"
    
    # Combine all content
    all_content = " ".join([result['content'] for result in search_results])
    urls = list(set([result['url'] for result in search_results]))
    
    # Simple field-specific prompts
    prompts = {
        "linkedin_url": f"Find the LinkedIn URL for {company_name}",
        "company_website_url": f"Find the website URL for {company_name}",
        "industry_category": f"What industry is {company_name} in?",
        "employee_count_linkedin": f"How many employees does {company_name} have?",
        "headquarters_location": f"Where is {company_name} headquartered?",
        "revenue_source": f"What is {company_name}'s revenue and business model?",
        "branch_network_count": f"How many branches or facilities does {company_name} have?",
        "expansion_news_12mo": f"What recent expansion has {company_name} done?",
        "digital_transformation_initiatives": f"What digital projects does {company_name} have?",
        "it_leadership_change": f"Who are the IT leaders at {company_name}?",
        "existing_network_vendors": f"What technology does {company_name} use?",
        "wifi_lan_tender_found": f"Any network projects at {company_name}?",
        "iot_automation_edge_integration": f"Does {company_name} use IoT or automation?",
        "cloud_adoption_gcc_setup": f"What cloud services does {company_name} use?",
        "physical_infrastructure_signals": f"Any new facilities at {company_name}?",
        "it_infra_budget_capex": f"What is {company_name}'s IT budget?"
    }
    
    prompt = f"""
    Question: {prompts.get(field_name, f"Information about {company_name}")}
    
    Context: {all_content}
    
    Answer based on the context above. Be specific and factual.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You answer questions based on the provided context. Be direct and factual."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Very simple validation - just check if it's empty
        if not response or len(response) < 3:
            return "Not available"
        
        # Add sources
        if urls:
            source_text = f" [Sources: {', '.join(urls[:2])}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "Not available"

# --- Working Research Function ---
def get_company_data(company_name: str) -> Dict[str, Any]:
    """Get company data that actually works"""
    
    company_data = {}
    
    # Simple progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_fields = len(REQUIRED_FIELDS) - 2
    
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"Getting {field.replace('_', ' ')}...")
        
        # Get data for this field
        search_results = working_company_search(company_name, field)
        field_data = extract_company_info(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1)
    
    # Add relevance analysis
    status_text.text("Analyzing opportunities...")
    progress_bar.progress(90)
    
    # Simple relevance analysis based on available data
    available_data = [field for field, value in company_data.items() 
                     if value != "Not available" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]]
    
    if len(available_data) > 5:
        opportunities = [
            f"- Company operations and expansion → Network infrastructure and Wi-Fi solutions for business connectivity",
            f"- Business technology needs → IT infrastructure modernization and network integration services",
            f"- Organizational growth → Enterprise-grade network solutions for operational efficiency"
        ]
        score = "High"
    elif len(available_data) > 2:
        opportunities = [
            f"- Business operations → Network connectivity and Wi-Fi solutions",
            f"- Technology infrastructure → Network integration services",
            f"- Operational needs → IT infrastructure solutions"
        ]
        score = "Medium"
    else:
        opportunities = [
            f"- Potential network infrastructure needs → Wi-Fi and connectivity solutions",
            f"- Business technology requirements → Network integration services", 
            f"- Operational connectivity → IT infrastructure solutions"
        ]
        score = "Medium"
    
    company_data["why_relevant_to_syntel_bullets"] = "\n".join(opportunities)
    company_data["intent_scoring_level"] = score
    
    progress_bar.progress(100)
    status_text.text("Complete!")
    
    return company_data

# --- Display Results ---
def show_results(company_name: str, data_dict: dict) -> pd.DataFrame:
    """Show results in table format"""
    
    mapping = {
        "Company Name": "company_name",
        "LinkedIn URL": "linkedin_url",
        "Company Website": "company_website_url", 
        "Industry": "industry_category",
        "Employee Count": "employee_count_linkedin",
        "Headquarters": "headquarters_location",
        "Revenue & Business": "revenue_source",
        "Branch Network": "branch_network_count",
        "Recent Expansion": "expansion_news_12mo",
        "Digital Initiatives": "digital_transformation_initiatives",
        "IT Leadership": "it_leadership_change",
        "Technology Vendors": "existing_network_vendors",
        "Network Projects": "wifi_lan_tender_found",
        "IoT & Automation": "iot_automation_edge_integration",
        "Cloud Strategy": "cloud_adoption_gcc_setup",
        "Infrastructure Growth": "physical_infrastructure_signals",
        "IT Investment": "it_infra_budget_capex",
        "Syntel Fit Score": "intent_scoring_level",
        "Syntel Opportunities": "why_relevant_to_syntel_bullets",
    }
    
    data_list = []
    for display_col, data_field in mapping.items():
        if display_col == "Company Name":
            value = company_name
        else:
            value = data_dict.get(data_field, "Not available")
        
        # Format opportunities
        if data_field == "why_relevant_to_syntel_bullets":
            if value != "Not available":
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # Make sources clickable
            if "Sources:" in str(value):
                def make_clickable(match):
                    url = match.group(1)
                    return f'[<a href="{url}" target="_blank">Source</a>]'
                
                value_with_links = re.sub(r'\[Sources: ([^\]]+)\]', make_clickable, value)
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{value_with_links}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit App ---
st.set_page_config(
    page_title="Company Research",
    layout="wide"
)

st.title("Company Intelligence Research")
st.markdown("### Get Business Information")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter company name:", "Neuberg Diagnostics")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    research_button = st.button("Start Research", type="primary")

if research_button:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"Researching {company_input}..."):
        try:
            # Get company data
            company_data = get_company_data(company_input)
            
            # Show results
            st.success(f"Research complete for {company_input}")
            
            st.subheader(f"Research Report for {company_input}")
            final_df = show_results(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show summary
            with st.expander("Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "Not available")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Found", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Data Quality", "Good" if completed_fields > 8 else "Basic")
                with col3:
                    st.metric("Opportunity", company_data.get("intent_scoring_level", "Medium"))
            
            # Download
            st.subheader("Download Report")
            
            col_json, col_csv = st.columns(2)
            
            with col_json:
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(company_data, indent=2),
                    file_name=f"{company_input.replace(' ', '_')}.json",
                    mime="application/json"
                )

            with col_csv:
                csv_data = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{company_input.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            # Save to history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": company_data
            }
            st.session_state.research_history.append(research_entry)
                        
        except Exception as e:
            st.error(f"Research failed: {str(e)}")

# History
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        idx = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"{research['company']}", expanded=False):
            st.write(f"Score: {research['data'].get('intent_scoring_level', 'Medium')}")
            completed = sum(1 for field in REQUIRED_FIELDS 
                         if research['data'].get(field) and 
                         research['data'].get(field) != "Not available")
            st.write(f"Fields: {completed}/{len(REQUIRED_FIELDS)}")
        
            if st.button(f"Load {research['company']}", key=f"load_{idx}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

st.markdown("---")
st.markdown("Company Research Tool")
