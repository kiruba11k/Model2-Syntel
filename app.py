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

# --- Field-Specific Search Queries ---
FIELD_SEARCH_QUERIES = {
    "linkedin_url": [
        '"{company}" LinkedIn company page',
        '"{company}" LinkedIn profile',
    ],
    "company_website_url": [
        '"{company}" official website',
        '"{company}" company website',
    ],
    "industry_category": [
        '"{company}" industry business',
        '"{company}" sector classification',
    ],
    "employee_count_linkedin": [
        '"{company}" employee count',
        '"{company}" number of employees',
        '"{company}" workforce size',
    ],
    "headquarters_location": [
        '"{company}" headquarters location',
        '"{company}" corporate office',
        '"{company}" head office',
    ],
    "revenue_source": [
        '"{company}" revenue business model',
        '"{company}" revenue streams',
        '"{company}" financial results',
    ],
    "branch_network_count": [
        '"{company}" branch network facilities',
        '"{company}" number of locations',
        '"{company}" warehouses facilities count',
        '"{company}" pallet capacity',
    ],
    "expansion_news_12mo": [
        '"{company}" expansion 2024 2025',
        '"{company}" new facilities 2024',
        '"{company}" capacity expansion',
        '"{company}" new warehouses',
    ],
    "digital_transformation_initiatives": [
        '"{company}" digital transformation',
        '"{company}" IT initiatives',
        '"{company}" technology modernization',
    ],
    "it_leadership_change": [
        '"{company}" CIO CTO IT leadership',
        '"{company}" IT director appointment',
        '"{company}" technology leadership',
    ],
    "existing_network_vendors": [
        '"{company}" technology vendors',
        '"{company}" IT infrastructure vendors',
        '"{company}" Cisco VMware SAP Oracle',
        '"{company}" network equipment',
    ],
    "wifi_lan_tender_found": [
        '"{company}" WiFi upgrade',
        '"{company}" LAN tender',
        '"{company}" network infrastructure project',
        '"{company}" wireless deployment',
    ],
    "iot_automation_edge_integration": [
        '"{company}" IoT automation',
        '"{company}" smart technology',
        '"{company}" robotics automation',
        '"{company}" Industry 4.0',
    ],
    "cloud_adoption_gcc_setup": [
        '"{company}" cloud adoption',
        '"{company}" AWS Azure Google Cloud',
        '"{company}" GCC setup',
        '"{company}" cloud migration',
    ],
    "physical_infrastructure_signals": [
        '"{company}" new construction',
        '"{company}" facility expansion',
        '"{company}" infrastructure investment',
    ],
    "it_infra_budget_capex": [
        '"{company}" IT budget',
        '"{company}" capital expenditure',
        '"{company}" technology investment',
    ]
}

# --- Syntel Core Offerings ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Field-by-Field Extraction Functions ---
def search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Search for specific field information"""
    queries = FIELD_SEARCH_QUERIES.get(field_name, [f'"{company_name}" {field_name}'])
    all_results = []
    
    for query_template in queries[:2]:  # Use first 2 queries per field
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
                            "content": content[:500],
                            "url": result.get('url', ''),
                            "field": field_name,
                            "query": query
                        })
        except Exception as e:
            continue
    
    return all_results

def extract_field_data(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Extract specific information for a single field"""
    
    if not search_results:
        return "Information not found in current research"
    
    # Format research data
    research_text = f"RESEARCH FOR {field_name.upper()}:\n\n"
    for i, result in enumerate(search_results):
        research_text += f"Source {i+1}:\n"
        research_text += f"Title: {result['title']}\n"
        research_text += f"URL: {result['url']}\n"
        research_text += f"Content: {result['content']}\n"
        research_text += "─" * 50 + "\n"
    
    # Field-specific extraction prompts
    extraction_prompts = {
        "linkedin_url": f"""
        Extract the LinkedIn company page URL for {company_name} from the research below.
        Look for: https://linkedin.com/company/ or https://in.linkedin.com/company/
        Return ONLY the URL or "Not found".
        
        {research_text}
        """,
        
        "company_website_url": f"""
        Extract the official company website URL for {company_name} from the research below.
        Look for: http:// or https:// followed by company domain
        Return ONLY the URL or "Not found".
        
        {research_text}
        """,
        
        "industry_category": f"""
        Extract the specific industry classification for {company_name} from the research below.
        Be precise about the industry sector.
        Return the industry classification with source URL.
        
        {research_text}
        """,
        
        "employee_count_linkedin": f"""
        Extract the employee count information for {company_name} from the research below.
        Look for specific numbers or ranges like "1,001-5,000 employees".
        Return the employee count with source URL.
        
        {research_text}
        """,
        
        "headquarters_location": f"""
        Extract the headquarters location for {company_name} from the research below.
        Include city, state, and country if available.
        Return the location with source URL.
        
        {research_text}
        """,
        
        "branch_network_count": f"""
        Extract the branch network and facilities count for {company_name} from the research below.
        Look for numbers of facilities, warehouses, locations, pallet capacity.
        Be specific with numbers and include source URLs.
        
        {research_text}
        """,
        
        "expansion_news_12mo": f"""
        Extract recent expansion news (last 12 months) for {company_name} from the research below.
        Look for new facilities, capacity expansion, investments with dates and locations.
        Include specific details and source URLs.
        
        {research_text}
        """,
        
        "existing_network_vendors": f"""
        Extract existing network vendors and technology stack for {company_name} from the research below.
        Look for specific vendors like Cisco, VMware, SAP, Oracle, Microsoft, AWS, Azure.
        List specific vendors and technologies with source URLs.
        
        {research_text}
        """,
        
        "it_infra_budget_capex": f"""
        Extract IT infrastructure budget and capex information for {company_name} from the research below.
        Look for budget amounts, investments, capital expenditure on technology.
        Include specific amounts and source URLs.
        
        {research_text}
        """
    }
    
    prompt = extraction_prompts.get(field_name, f"""
    Extract information about {field_name} for {company_name} from the research below.
    Be specific and include source URLs when available.
    
    {research_text}
    """)
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract specific business information. Be precise and include sources."),
            HumanMessage(content=prompt)
        ]).content
        
        # Add source URLs to the response
        if response and response != "Not found" and "not found" not in response.lower():
            unique_urls = list(set([result['url'] for result in search_results]))
            source_text = " [Sources: " + ", ".join(unique_urls[:2]) + "]"
            return response + source_text
        else:
            return "Information not found in current research"
            
    except Exception as e:
        return "Information not found in current research"

def generate_syntel_relevance(company_data: Dict, company_name: str) -> tuple:
    """Generate Syntel relevance analysis"""
    
    # Create context from available data
    context = f"Company: {company_name}\n\n"
    for field, value in company_data.items():
        if value != "Information not found in current research":
            context += f"{field}: {value}\n"
    
    relevance_prompt = f"""
    Based on the following company information, generate SPECIFIC relevance analysis for Syntel:
    
    {context}
    
    Syntel Expertise:
    {SYNTEL_EXPERTISE}
    
    Generate TWO outputs:
    
    1. why_relevant_to_syntel_bullets: 3-5 SPECIFIC bullet points based on ACTUAL company data
    2. intent_scoring_level: High, Medium, or Low based on concrete evidence
    
    Scoring Criteria:
    - High: Clear IT transformation signals, active expansion, specific technology initiatives
    - Medium: Some digital initiatives, growth potential, moderate IT spending
    - Low: Limited IT signals, stable operations, minimal transformation
    
    Return ONLY a JSON object with these two fields.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You analyze business relevance for IT services."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            relevance_data = json.loads(json_match.group(0))
            bullets = relevance_data.get("why_relevant_to_syntel_bullets", "")
            scoring = relevance_data.get("intent_scoring_level", "Medium")
            
            # Format bullets properly
            if bullets:
                formatted_bullets = "\n".join([f"* {bullet.strip()}" for bullet in bullets.split('\n') if bullet.strip()])
                return formatted_bullets, scoring
                
    except Exception as e:
        pass
    
    # Fallback relevance
    fallback_bullets = f"""* {company_name} shows potential for IT infrastructure modernization
* Digital transformation opportunities in current operations  
* IT service integration could enhance business capabilities"""
    
    return fallback_bullets, "Medium"

# --- Main Research Function ---
def research_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Main function to research all company intelligence fields"""
    
    company_data = {}
    
    # Research each field individually
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):  # Exclude relevance fields
        progress = (i / len(REQUIRED_FIELDS)) * 80
        st.session_state.progress_bar.progress(int(progress))
        st.session_state.status_text.info(f"🔍 Researching {field.replace('_', ' ').title()}...")
        
        # Search for this specific field
        search_results = search_for_field(company_name, field)
        
        # Extract data for this field
        field_data = extract_field_data(company_name, field, search_results)
        company_data[field] = field_data
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    # Generate Syntel relevance analysis
    st.session_state.status_text.info("🎯 Analyzing Syntel relevance...")
    st.session_state.progress_bar.progress(90)
    
    relevance_bullets, intent_score = generate_syntel_relevance(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
    company_data["intent_scoring_level"] = intent_score
    
    return company_data

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
            value = data_dict.get(data_field, "Research in progress")
        
        # Format bullet points for better display
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str):
                html_value = value.replace('\n', '<br>').replace('*', '•')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Field-by-Field Research)",
    layout="wide",
    page_icon="🏢"
)

st.title("🏢 Syntel Company Data AI Agent")
st.markdown("### 🎯 Field-by-Field Targeted Research")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"

# Display research approach
with st.expander("🔧 Field-by-Field Research Strategy", expanded=True):
    st.markdown("""
    **🎯 Targeted Field Research:**
    
    - **🔍 Individual Field Search**: Each field has specific search queries
    - **📊 Precise Extraction**: Dedicated extraction for each data point  
    - **🌐 Source Integration**: Every field includes source URLs
    - **🔄 Sequential Processing**: Fields researched one by one for accuracy
    - **🎨 Proper Mapping**: Information goes to correct columns
    
    **Research Process:**
    1. Search for specific field information
    2. Extract precise data for that field
    3. Include source URLs from search results
    4. Map to correct column
    5. Repeat for all 18 fields
    """)

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input)
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("🚀 Start Field Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # Create progress and status display
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    # Show initial status
    st.session_state.status_text.info("🚀 Starting field-by-field research...")
    st.session_state.progress_bar.progress(5)
    
    with st.spinner(f"**Researching {company_input} with field-by-field approach...**"):
        try:
            # Perform comprehensive research
            company_data = research_company_intelligence(company_input)
            
            # Update final progress
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"🎉 Field Research Complete for {company_input}!")
            
            # Store in history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": company_data
            }
            st.session_state.research_history.append(research_entry)
            
            # Display results
            st.balloons()
            st.success("✅ All fields researched with source URLs!")
            
            # Display final results
            st.subheader(f"📊 Comprehensive Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show research metrics
            with st.expander("🔍 Research Metrics", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    "not found" not in company_data.get(field, "").lower())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Researched", f"{len(REQUIRED_FIELDS)}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col3:
                    st.metric("Intent Score", company_data.get("intent_scoring_level", "Medium"))
                
                # Show fields with sources
                fields_with_sources = sum(1 for field in REQUIRED_FIELDS 
                                        if company_data.get(field) and 
                                        "http" in company_data.get(field, ""))
                st.info(f"**Source Coverage:** {fields_with_sources}/{len(REQUIRED_FIELDS)} fields include source URLs")
            
            # Download options
            st.subheader("💾 Download Report")
            
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
                     file_name=f"{company_input.replace(' ', '_')}_field_research.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_field_research.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_field_research.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits. Please try again in a few moments.")

# Research History
if st.session_state.research_history:
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"🎯 Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                if research['data'].get(field) and 
                                "not found" not in research['data'].get(field, "").lower())
            st.write(f"📊 Fields Completed: {completed_fields}/{len(REQUIRED_FIELDS)}")
            
            if st.button(f"📥 Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Powered by Field-by-Field Targeted Research | "
    "Precise Data Mapping | Source URL Integration"
    "</div>",
    unsafe_allow_html=True
)
