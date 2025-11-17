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

# --- Field-Specific Search Queries ---
FIELD_SEARCH_QUERIES = {
    "linkedin_url": ['"{company}" LinkedIn company page'],
    "company_website_url": ['"{company}" official website'],
    "industry_category": ['"{company}" industry business'],
    "employee_count_linkedin": ['"{company}" employee count LinkedIn'],
    "headquarters_location": ['"{company}" headquarters location'],
    "revenue_source": ['"{company}" revenue business model'],
    "branch_network_count": ['"{company}" branch network facilities locations pallet capacity'],
    "expansion_news_12mo": ['"{company}" expansion 2024 2025 new facilities'],
    "digital_transformation_initiatives": ['"{company}" digital transformation IT initiatives'],
    "it_leadership_change": ['"{company}" CIO CTO IT leadership'],
    "existing_network_vendors": ['"{company}" technology vendors Cisco VMware SAP Oracle'],
    "wifi_lan_tender_found": ['"{company}" WiFi LAN tender network upgrade'],
    "iot_automation_edge_integration": ['"{company}" IoT automation robotics'],
    "cloud_adoption_gcc_setup": ['"{company}" cloud adoption AWS Azure GCC'],
    "physical_infrastructure_signals": ['"{company}" new construction facility expansion'],
    "it_infra_budget_capex": ['"{company}" IT budget capex investment']
}

# --- Enhanced Extraction Functions with Source URLs ---
def search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Search for specific field information"""
    queries = FIELD_SEARCH_QUERIES.get(field_name, [f'"{company_name}" {field_name}'])
    all_results = []
    
    for query_template in queries[:2]:
        query = query_template.format(company=company_name)
        try:
            time.sleep(1)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    if len(content) > 50:
                        all_results.append({
                            "title": result.get('title', ''),
                            "content": content[:400],
                            "url": result.get('url', ''),
                            "field": field_name
                        })
        except Exception as e:
            continue
    
    return all_results

def extract_concise_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Extract SHORT, CRISP information for a single field WITH SOURCE URLs"""
    
    if not search_results:
        return "N/A"
    
    # Format research data concisely
    research_text = f"Research for {field_name}:\n"
    for i, result in enumerate(search_results):
        research_text += f"Source {i+1}: {result['content'][:200]}\n"
        research_text += f"URL: {result['url']}\n\n"
    
    # Get unique source URLs (limit to 2 for conciseness)
    unique_urls = list(set([result['url'] for result in search_results]))[:2]
    
    # Concise extraction prompts for each field
    concise_prompts = {
        "linkedin_url": f"Extract ONLY the LinkedIn URL for {company_name}. Return just the URL or 'N/A'.",
        
        "company_website_url": f"Extract ONLY the official website URL for {company_name}. Return just the URL or 'N/A'.",
        
        "industry_category": f"Extract the industry category for {company_name} in 3-5 words max. Examples: 'Warehouse / Cold-chain', 'Manufacturing (Factories)'. Return just the category.",
        
        "employee_count_linkedin": f"Extract employee count for {company_name} in format like '581' or '1,001-5,000'. Return just the number/range.",
        
        "headquarters_location": f"Extract headquarters location for {company_name} in format 'City, State'. Return just the location.",
        
        "revenue_source": f"Extract revenue source for {company_name} in 5-7 words max. Examples: '$60.2 Million', 'Cold chain warehousing'. Return just the revenue info.",
        
        "branch_network_count": f"Extract branch network count for {company_name} in format like '44 warehouses across 21 cities, 154,330 pallet capacity'. Be concise with numbers.",
        
        "expansion_news_12mo": f"Extract recent expansion news for {company_name} in format like 'Jun 2025: Kolkata-5,630 pallets, Krishnapatnam-3,927 pallets'. List key expansions only.",
        
        "digital_transformation_initiatives": f"Extract digital transformation initiatives for {company_name} in 10-15 words max. List key technologies like 'ERP, WMS, IoT, RFID'.",
        
        "it_leadership_change": f"Extract IT leadership info for {company_name} in format 'Name - Position (Year)'. If no change, say 'N/A'.",
        
        "existing_network_vendors": f"Extract technology vendors for {company_name} in format like 'Microsoft Dynamics 365, SAP, Oracle, Cisco'. List 3-5 main vendors.",
        
        "wifi_lan_tender_found": f"Extract WiFi/LAN tender info for {company_name}. If found, describe briefly like 'Commissioning - Wi-Fi needed for automation'. If not, 'N/A'.",
        
        "iot_automation_edge_integration": f"Extract IoT/Automation status for {company_name}. Use 'Yes - [brief tech]' or 'No' format. Examples: 'Yes - IoT temperature monitoring, RFID'.",
        
        "cloud_adoption_gcc_setup": f"Extract cloud adoption status for {company_name}. Use 'Yes - [platforms]' or 'No' or 'Partial - [details]'.",
        
        "physical_infrastructure_signals": f"Extract physical infrastructure signals for {company_name} in 10-15 words max. Focus on new facilities/expansions.",
        
        "it_infra_budget_capex": f"Extract IT budget/capex for {company_name} in format like 'Not publicly disclosed; Total capex ~Rs 200 cr'. Be concise."
    }
    
    prompt = f"""
    {concise_prompts.get(field_name, f"Extract {field_name} for {company_name} in 10-15 words max.")}
    
    {research_text}
    
    Return ONLY the concise answer. No explanations. No full sentences. Just the factual information.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract concise business information. Return only short factual answers."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Clean up response - remove any explanations
        if "N/A" in response or "not found" in response.lower() or len(response) < 5:
            return "N/A"
        
        # Limit response length
        if len(response) > 200:
            response = response[:197] + "..."
        
        # Add source URLs to the response
        if unique_urls and response != "N/A":
            if len(unique_urls) == 1:
                response += f" [Source: {unique_urls[0]}]"
            else:
                response += f" [Sources: {', '.join(unique_urls)}]"
            
        return response
            
    except Exception as e:
        return "N/A"

def generate_concise_relevance_with_sources(company_data: Dict, company_name: str, search_results: List[Dict]) -> tuple:
    """Generate concise Syntel relevance analysis with source references"""
    
    # Create brief context with source references
    context = ""
    key_fields = ["branch_network_count", "expansion_news_12mo", "iot_automation_edge_integration", 
                  "digital_transformation_initiatives", "it_infra_budget_capex"]
    
    for field in key_fields:
        if company_data.get(field) and company_data[field] != "N/A":
            context += f"{field}: {company_data[field]}\n"
    
    # Get relevant URLs from all search results for context
    all_urls = []
    for result in search_results:
        if result['url'] not in all_urls:
            all_urls.append(result['url'])
    
    source_context = "Based on comprehensive research including: " + ", ".join(all_urls[:3]) + "."
    
    relevance_prompt = f"""
    Based on this company data, create 3 CONCISE bullet points for Syntel relevance and score intent.
    
    Company: {company_name}
    {source_context}
    
    Key Data:
    {context}
    
    Create exactly 3 bullet points in this format:
    1) [Specific infrastructure signal] requiring [Syntel solution]
    2) [Technology adoption] indicating [opportunity area]
    3) [Investment/expansion] aligning with [Syntel expertise]
    
    Then score: High/Medium/Low
    
    Return ONLY this format:
    BULLETS:
    1) ...
    2) ... 
    3) ...
    SCORE: High/Medium/Low
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="Create concise, actionable business relevance points."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse bullets and score
        bullets = []
        score = "Medium"
        
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith(('1)', '2)', '3)')):
                bullets.append(line.strip())
            elif 'SCORE:' in line.upper():
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
        
        if bullets:
            formatted_bullets = "\n".join(bullets[:3])  # Take max 3 bullets
        else:
            # Fallback bullets
            formatted_bullets = f"""1) {company_name} infrastructure expansion presents network solutions opportunity
2) Digital transformation initiatives align with Syntel's automation expertise  
3) IT modernization needs match Syntel's cloud and infrastructure services"""
            
        return formatted_bullets, score
        
    except Exception as e:
        fallback_bullets = f"""1) {company_name} shows potential for IT infrastructure modernization
2) Digital transformation opportunities identified
3) IT service integration possibilities exist"""
        return fallback_bullets, "Medium"

# --- Main Research Function ---
def research_concise_intelligence_with_sources(company_name: str) -> Dict[str, Any]:
    """Main function to research all fields with concise outputs AND source URLs"""
    
    company_data = {}
    all_search_results = []  # Collect all results for relevance analysis
    
    # Research each field individually with concise extraction and sources
    total_fields = len(REQUIRED_FIELDS) - 2  # Exclude relevance fields
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        st.session_state.progress_bar.progress(int(progress))
        st.session_state.status_text.info(f"🔍 Researching {field.replace('_', ' ').title()}...")
        
        # Search and extract concise data with sources
        search_results = search_for_field(company_name, field)
        all_search_results.extend(search_results)  # Collect for relevance analysis
        
        field_data = extract_concise_field_with_sources(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1)
    
    # Generate concise relevance analysis with source context
    st.session_state.status_text.info(" Analyzing Syntel relevance...")
    st.session_state.progress_bar.progress(90)
    
    relevance_bullets, intent_score = generate_concise_relevance_with_sources(
        company_data, company_name, all_search_results
    )
    company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
    company_data["intent_scoring_level"] = intent_score
    
    return company_data

# --- Display Functions ---
def format_concise_display_with_sources(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into concise display format with sources"""
    
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
        
        # Format bullet points
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str):
                # Clean and format bullets
                cleaned_value = value.replace('1)', '•').replace('2)', '•').replace('3)', '•')
                html_value = cleaned_value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # For fields with URLs, make them clickable
            if isinstance(value, str) and "http" in value:
                # Convert URLs to clickable links
                url_pattern = r'(\[Source: (https?://[^\]]+)\])'
                def make_clickable(match):
                    full_text = match.group(1)
                    url = match.group(2)
                    return f'[<a href="{url}" target="_blank">Source</a>]'
                
                value_with_links = re.sub(url_pattern, make_clickable, value)
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{value_with_links}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Concise + Sources)",
    layout="wide",
    page_icon=""
)

st.title("Syntel Company Data AI Agent")
st.markdown("###  Concise Format with Source URLs")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"

# Display enhanced approach
with st.expander("Concise Research with Source URLs", expanded=True):
    st.markdown("""
    ** Short & Crisp Outputs WITH SOURCE URLs:**
    
    - ** Concise Format**: Brief, to-the-point information
    - ** Source URLs**: Every field includes clickable source links
    - ** Key Facts Only**: No verbose explanations
    - ** Standardized Format**: Consistent output style
    
    **Output Style:**
    - Industry: "Warehouse / Cold-chain [Source: url]" 
    - Employees: "581 [Source: url]"
    - Expansion: "Jun 2025: Kolkata-5,630 pallets [Sources: url1, url2]"
    - Vendors: "Microsoft Dynamics, SAP, Oracle [Source: url]"
    """)

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input)
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button(" Start Research with Sources", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # Create progress and status display
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    # Show initial status
    st.session_state.status_text.info(" Starting research with source URLs...")
    st.session_state.progress_bar.progress(5)
    
    with st.spinner(f"**Researching {company_input} with source URLs...**"):
        try:
            # Perform concise research with sources
            company_data = research_concise_intelligence_with_sources(company_input)
            
            # Update final progress
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f" Research Complete for {company_input}!")
            
            # Store in history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": company_data
            }
            st.session_state.research_history.append(research_entry)
            
            # Display results
            st.balloons()
            st.success(" All fields researched with source URLs!")
            
            # Display final results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = format_concise_display_with_sources(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show completion metrics
            with st.expander("Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "N/A")
                
                fields_with_sources = sum(1 for field in REQUIRED_FIELDS[:-2]  # Exclude relevance fields
                                       if company_data.get(field) and 
                                       company_data.get(field) != "N/A" and
                                       "Source" in company_data.get(field, ""))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Fields with Sources", f"{fields_with_sources}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    st.metric("Intent Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download options
            st.subheader(" Download Report")
            
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
                     file_name=f"{company_input.replace(' ', '_')}_sources_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_sources_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_sources_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits. Please try again in a few moments.")

# Research History
if st.session_state.research_history:
    st.sidebar.header(" Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f" Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
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
    " "
    "</div>",
    unsafe_allow_html=True
)
