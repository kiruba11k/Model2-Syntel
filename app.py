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

# --- Syntel GTM Intelligence ---
SYNTEL_GTM_INTELLIGENCE = """
SYNTEL GTM INTELLIGENCE - Wi-Fi / Network Integration (Syntel + Altai):

GEOGRAPHY: India

TARGET INDUSTRIES:
- Ports, Stadiums, Education, Manufacturing - Factories
- Healthcare, Hospitality - Hotels & Convention Centres
- Warehouses, BFSI, IT/ITES, GCC - Mumbai, Pune, Bangalore, Hyderabad, Chennai

IDEAL CUSTOMER PROFILE:
- Employee Count: 150-500+ employees
- Revenue Size: 100 Cr+

PRODUCT & SERVICE OFFERING (Focus):
- Network Integration Solutions & Services
- Wi-Fi Deployments (Active + Passive Components)
- Delivery & Implementation - Unbox + Integrate with current environment
- Multi-brand implementation support (Not only Altai)

KEY TARGET SIGNALS:
- SI partners working on Wi-Fi and networking
- Companies expanding rapidly (new offices, campuses, GCCs)
- Organizations with urgent connectivity, coverage, or security needs
- Digital transformation-aligned companies
- Facilities with large spaces (auditoriums, warehouses, universities)

SECONDARY BUYER INTENT:
- Companies opening new offices, factories, campuses, warehouses
- Companies undergoing technology modernization
- Companies announcing expansion or relocation

ALTAI Wi-Fi ADVANTAGES:
- 3-5X Coverage Advantage (1 Altai AP = 3-5 Cisco/Ruckus APs)
- High Concurrent User Handling
- Zero-Roaming Drop & Seamless Handover
- Superior Outdoor Performance
- Lower Total Cost of Ownership
- Excellent for: Manufacturing, Hospitality, GCC, Healthcare, Warehouses, Education
"""

# --- Relaxed Search with Better Coverage ---
def relaxed_field_search(company_name: str, field_name: str) -> List[Dict]:
    """Relaxed search to get more results with flexible matching"""
    
    field_search_queries = {
        "linkedin_url": [
            f'{company_name} LinkedIn',
            f'{company_name} LinkedIn company page',
            f'"{company_name}" LinkedIn'
        ],
        "company_website_url": [
            f'{company_name} official website',
            f'{company_name} company website',
            f'"{company_name}" website'
        ],
        "industry_category": [
            f'{company_name} industry',
            f'{company_name} business type',
            f'what does {company_name} do'
        ],
        "employee_count_linkedin": [
            f'{company_name} employees',
            f'{company_name} employee count',
            f'{company_name} number of employees'
        ],
        "headquarters_location": [
            f'{company_name} headquarters',
            f'{company_name} corporate office',
            f'where is {company_name} located'
        ],
        "revenue_source": [
            f'{company_name} revenue',
            f'{company_name} financials',
            f'{company_name} business model revenue'
        ],
        "branch_network_count": [
            f'{company_name} branches locations',
            f'{company_name} facilities network',
            f'{company_name} offices'
        ],
        "expansion_news_12mo": [
            f'{company_name} expansion 2024',
            f'{company_name} new facilities 2024',
            f'{company_name} growth news'
        ],
        "digital_transformation_initiatives": [
            f'{company_name} digital transformation',
            f'{company_name} IT initiatives',
            f'{company_name} technology projects'
        ],
        "it_leadership_change": [
            f'{company_name} CIO CTO',
            f'{company_name} IT leadership',
            f'{company_name} technology executives'
        ],
        "existing_network_vendors": [
            f'{company_name} technology vendors',
            f'{company_name} IT partners',
            f'{company_name} software systems'
        ],
        "wifi_lan_tender_found": [
            f'{company_name} network upgrade',
            f'{company_name} IT infrastructure',
            f'{company_name} network project'
        ],
        "iot_automation_edge_integration": [
            f'{company_name} IoT',
            f'{company_name} automation',
            f'{company_name} smart technology'
        ],
        "cloud_adoption_gcc_setup": [
            f'{company_name} cloud',
            f'{company_name} AWS Azure',
            f'{company_name} cloud computing'
        ],
        "physical_infrastructure_signals": [
            f'{company_name} new facility',
            f'{company_name} infrastructure',
            f'{company_name} construction'
        ],
        "it_infra_budget_capex": [
            f'{company_name} IT budget',
            f'{company_name} technology investment',
            f'{company_name} IT spending'
        ]
    }
    
    all_results = []
    queries = field_search_queries.get(field_name, [f'{company_name} {field_name}'])
    
    for query in queries:
        try:
            time.sleep(1)
            results = search_tool.invoke({"query": query, "max_results": 4})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    # Relaxed company matching - just check if company name appears
                    if company_name.lower() in content.lower() or company_name.lower() in title.lower():
                        if len(content) > 30:  # Reduced minimum content length
                            all_results.append({
                                "title": title,
                                "content": content[:500],
                                "url": result.get('url', ''),
                                "field": field_name
                            })
        except Exception as e:
            continue
    
    return all_results[:5]  # Return up to 5 results

# --- Flexible Field Extraction ---
def flexible_field_extraction(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Flexible extraction that works with available information"""
    
    if not search_results:
        return "Not available"
    
    # Build research context
    research_context = "Available information:\n\n"
    for i, result in enumerate(search_results):
        research_context += f"Source {i+1}: {result['content']}\n"
        research_context += f"URL: {result['url']}\n\n"
    
    # Field-specific guidance
    field_guidance = {
        "linkedin_url": "Find any LinkedIn URL for this company",
        "company_website_url": "Find the main company website URL",
        "industry_category": "What industry or business sector is this company in?",
        "employee_count_linkedin": "How many employees does this company have?",
        "headquarters_location": "Where is the company headquarters located?",
        "revenue_source": "What is the company's revenue and business model?",
        "branch_network_count": "How many branches or facilities does the company have?",
        "expansion_news_12mo": "Any recent expansion or growth news?",
        "digital_transformation_initiatives": "What digital or IT projects is the company working on?",
        "it_leadership_change": "Who are the IT leaders (CIO, CTO, IT Director)?",
        "existing_network_vendors": "What technology vendors or partners does the company use?",
        "wifi_lan_tender_found": "Any network or IT infrastructure projects?",
        "iot_automation_edge_integration": "Any IoT, automation, or smart technology use?",
        "cloud_adoption_gcc_setup": "What cloud platforms does the company use?",
        "physical_infrastructure_signals": "Any new facilities or construction?",
        "it_infra_budget_capex": "What is the company's IT budget or spending?"
    }
    
    guidance = field_guidance.get(field_name, f"Information about {field_name}")
    
    # Get source URLs
    unique_urls = list(set([result['url'] for result in search_results]))[:2]
    
    extraction_prompt = f"""
    Company: {company_name}
    Looking for: {guidance}
    
    Research data:
    {research_context}
    
    Instructions:
    - Extract any relevant information you can find
    - Be concise but informative
    - If you find partial information, provide what's available
    - Don't make up information
    - If truly nothing relevant is found, say "Not available"
    
    Extracted information:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract company information from research data. Provide whatever relevant information you can find, even if incomplete. Be helpful and factual."),
            HumanMessage(content=extraction_prompt)
        ]).content.strip()
        
        # Simple cleaning
        response = re.sub(r'^(.*?):\s*', '', response)
        response = response.strip()
        
        # Very relaxed validation
        if (not response or 
            len(response) < 5 or
            response.lower() in ['not available', 'n/a', 'no information', 'none', 'unknown']):
            return "Not available"
        
        # Add sources if we have information
        if unique_urls and response != "Not available":
            source_text = f" [Sources: {', '.join(unique_urls)}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "Not available"

# --- Improved Relevance Analysis ---
def improved_relevance_analysis(company_data: Dict, company_name: str) -> tuple:
    """Improved relevance analysis that works with available data"""
    
    # Collect available data points
    available_data = []
    for field, value in company_data.items():
        if value and value != "Not available" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\s*\[Source[^\]]*\]', '', value)
            available_data.append(f"{field}: {clean_value}")
    
    data_context = "\n".join(available_data) if available_data else "Basic company information available"
    
    relevance_prompt = f"""
    Company: {company_name}
    
    Available Information:
    {data_context}
    
    Syntel specializes in:
    - Network integration and Wi-Fi solutions
    - Altai Wi-Fi with 3-5x better coverage
    - IT infrastructure modernization
    - Multi-vendor implementation
    
    Based on the available information, identify potential opportunities for Syntel.
    
    Provide 3 specific opportunities in this format:
    1. [Opportunity based on available data]
    2. [Another opportunity]
    3. [Third opportunity]
    
    Then provide an intent score: High/Medium/Low
    
    Be practical and focus on what's actually in the data.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="Identify practical business opportunities based on available company data. Be specific and realistic."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse opportunities
        bullets = []
        score = "Medium"
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('-'):
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                clean_line = re.sub(r'^-\s*', '', clean_line)
                if len(clean_line) > 10:
                    bullets.append(f"- {clean_line}")
            elif 'score:' in line.lower():
                if 'high' in line.lower():
                    score = "High"
                elif 'low' in line.lower():
                    score = "Low"
        
        # Ensure we have 3 opportunities
        while len(bullets) < 3:
            bullets.append(f"- Additional opportunity based on {company_name}'s business operations")
        
        formatted_bullets = "\n".join(bullets[:3])
        return formatted_bullets, score
        
    except Exception as e:
        fallback_bullets = [
            f"- {company_name}'s operations present network infrastructure opportunities",
            f"- Potential for IT modernization and connectivity solutions",
            f"- Business scale suggests need for enterprise-grade network services"
        ]
        return "\n".join(fallback_bullets), "Medium"

# --- Main Research Function ---
def practical_company_research(company_name: str) -> Dict[str, Any]:
    """Practical research that works with available information"""
    
    company_data = {}
    
    # Simple progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"Researching {field.replace('_', ' ')}...")
        
        # Search and extract
        search_results = relaxed_field_search(company_name, field)
        field_data = flexible_field_extraction(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1)
    
    # Generate relevance
    status_text.text("Analyzing opportunities...")
    progress_bar.progress(90)
    
    relevance_bullets, intent_score = improved_relevance_analysis(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
    company_data["intent_scoring_level"] = intent_score
    
    progress_bar.progress(100)
    status_text.text("Research complete!")
    
    return company_data

# --- Simple Display ---
def simple_display_format(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Simple display format"""
    
    mapping = {
        "Company Name": "company_name",
        "LinkedIn URL": "linkedin_url",
        "Website": "company_website_url", 
        "Industry": "industry_category",
        "Employees": "employee_count_linkedin",
        "Headquarters": "headquarters_location",
        "Revenue": "revenue_source",
        "Branch Network": "branch_network_count",
        "Expansion News": "expansion_news_12mo",
        "Digital Initiatives": "digital_transformation_initiatives",
        "IT Leadership": "it_leadership_change",
        "Technology Vendors": "existing_network_vendors",
        "Network Projects": "wifi_lan_tender_found",
        "IoT & Automation": "iot_automation_edge_integration",
        "Cloud Adoption": "cloud_adoption_gcc_setup",
        "Infrastructure": "physical_infrastructure_signals",
        "IT Budget": "it_infra_budget_capex",
        "Intent Score": "intent_scoring_level",
        "Syntel Opportunities": "why_relevant_to_syntel_bullets",
    }
    
    data_list = []
    for display_col, data_field in mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(data_field, "Not available")
        
        # Format opportunities with line breaks
        if data_field == "why_relevant_to_syntel_bullets":
            if value != "Not available":
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # Make sources clickable
            if "Source:" in str(value):
                def make_clickable(match):
                    url = match.group(1)
                    return f'[<a href="{url}" target="_blank">Source</a>]'
                
                value_with_links = re.sub(r'\[Source: ([^\]]+)\]', make_clickable, value)
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{value_with_links}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit App ---
st.set_page_config(
    page_title="Company Intelligence",
    layout="wide"
)

st.title("Company Intelligence Research")
st.markdown("### Business Intelligence Reporting")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter company name:", "Neuberg Diagnostics")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("Start Research", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"Researching {company_input}..."):
        try:
            # Perform research
            company_data = practical_company_research(company_input)
            
            # Display results
            st.success(f"Research complete for {company_input}")
            
            # Show results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = simple_display_format(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show metrics
            with st.expander("Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "Not available")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Data Quality", "Good" if completed_fields > 8 else "Basic")
                with col3:
                    st.metric("Opportunity Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download
            st.subheader("Download Report")
            
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
            
            # Store in history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": company_data
            }
            st.session_state.research_history.append(research_entry)
                        
        except Exception as e:
            st.error(f"Research failed: {str(e)}")
            st.info("Please try again in a few moments.")

# History
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        idx = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"{research['company']} - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Score: {research['data'].get('intent_scoring_level', 'Medium')}")
            completed = sum(1 for field in REQUIRED_FIELDS 
                         if research['data'].get(field) and 
                         research['data'].get(field) != "Not available")
            st.write(f"Fields: {completed}/{len(REQUIRED_FIELDS)}")
        
            if st.button(f"Load {research['company']}", key=f"load_{idx}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

st.markdown("---")
st.markdown("Company Intelligence Research")
