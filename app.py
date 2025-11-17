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

# --- Enhanced Search with Better Filtering ---
def enhanced_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Enhanced search with better filtering to avoid mixed company data"""
    
    search_queries = {
        "linkedin_url": [
            f'"{company_name}" LinkedIn company page',
            f'"{company_name}" LinkedIn official'
        ],
        "company_website_url": [
            f'"{company_name}" official website',
            f'"{company_name}" company website'
        ],
        "industry_category": [
            f'"{company_name}" industry business',
            f'"{company_name}" business sector'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employee count',
            f'"{company_name}" number of employees'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters location',
            f'"{company_name}" corporate office'
        ],
        "revenue_source": [
            f'"{company_name}" revenue business model',
            f'"{company_name}" annual revenue'
        ],
        "branch_network_count": [
            f'"{company_name}" branches locations network',
            f'"{company_name}" facilities offices'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024 2025',
            f'"{company_name}" new facilities growth'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT',
            f'"{company_name}" technology modernization'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT leadership',
            f'"{company_name}" technology leadership'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors',
            f'"{company_name}" IT infrastructure vendors'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN network upgrade',
            f'"{company_name}" network infrastructure'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation',
            f'"{company_name}" smart technology'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption',
            f'"{company_name}" cloud migration'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility',
            f'"{company_name}" infrastructure development'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget investment',
            f'"{company_name}" technology spending'
        ]
    }
    
    all_results = []
    queries = search_queries.get(field_name, [f'"{company_name}" {field_name}'])
    
    for query in queries[:2]:
        try:
            time.sleep(1.5)
            results = search_tool.invoke({"query": query, "max_results": 4})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    # Filter out results that don't match the company name well
                    company_match_score = sum([
                        1 for word in company_name.split() 
                        if word.lower() in content.lower() or word.lower() in title.lower()
                    ])
                    
                    # Only include results with good company matching
                    if company_match_score >= 1 and len(content) > 50:
                        all_results.append({
                            "title": title,
                            "content": content[:400],
                            "url": result.get('url', ''),
                            "field": field_name
                        })
        except Exception as e:
            continue
    
    return all_results

# --- Field-Specific Extraction Prompts ---
FIELD_EXTRACTION_PROMPTS = {
    "linkedin_url": "Extract the primary LinkedIn company page URL. Return only the URL or N/A.",
    "company_website_url": "Extract the official company website URL. Return only the URL or N/A.",
    "industry_category": "Extract the primary industry category in 2-4 words. Examples: Healthcare Diagnostics, Logistics, IT Services.",
    "employee_count_linkedin": "Extract employee count information. Return the number or range.",
    "headquarters_location": "Extract headquarters location in City, State format.",
    "revenue_source": "Extract revenue information and business model. Be specific with numbers if available.",
    "branch_network_count": "Extract branch network and facility information with specific counts.",
    "expansion_news_12mo": "Extract recent expansion news and growth initiatives from last 12-24 months.",
    "digital_transformation_initiatives": "Extract digital transformation projects and technology modernization efforts.",
    "it_leadership_change": "Extract IT leadership names and positions. Focus on CIO, CTO, IT Director roles.",
    "existing_network_vendors": "Extract technology vendors and partners used by the company.",
    "wifi_lan_tender_found": "Extract any network upgrade projects, WiFi deployments, or infrastructure tenders.",
    "iot_automation_edge_integration": "Extract IoT, automation, and smart technology implementations.",
    "cloud_adoption_gcc_setup": "Extract cloud adoption strategy and global capability centers.",
    "physical_infrastructure_signals": "Extract physical infrastructure developments and facility expansions.",
    "it_infra_budget_capex": "Extract IT infrastructure budget and capital expenditure information."
}

def precise_field_extraction(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Precise field extraction with strict matching"""
    
    if not search_results:
        return "N/A"
    
    # Build research context
    research_context = "Research data for extraction:\n\n"
    for i, result in enumerate(search_results[:3]):
        research_context += f"Source {i+1}: {result['content']}\n"
        research_context += f"URL: {result['url']}\n\n"
    
    # Get unique source URLs
    unique_urls = list(set([result['url'] for result in search_results]))[:2]
    
    prompt = f"""
    COMPANY: {company_name}
    FIELD TO EXTRACT: {field_name}
    
    {FIELD_EXTRACTION_PROMPTS.get(field_name, f"Extract {field_name} information")}
    
    {research_context}
    
    CRITICAL REQUIREMENTS:
    1. Extract information ONLY for {company_name}
    2. If information is not clearly about {company_name}, return N/A
    3. Be specific and factual
    4. Do not include field names or labels in response
    5. Return only the extracted information
    
    Extracted information:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract precise company information. Only return information that clearly matches the specified company. Return N/A if uncertain."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Clean response
        response = re.sub(r'^(.*?):\s*', '', response)  # Remove any labels
        response = response.strip()
        
        # Validate response
        if (not response or 
            response.upper() == 'N/A' or 
            len(response) < 3 or
            'not found' in response.lower() or
            'no information' in response.lower()):
            return "N/A"
        
        # Check if response seems to be about wrong company
        wrong_company_indicators = [
            'hcl', 'tcs', 'infosys', 'wipro', 'tech mahindra', 'cognizant',
            'unrelated', 'different company', 'another company'
        ]
        if any(indicator in response.lower() for indicator in wrong_company_indicators):
            return "N/A"
        
        # Limit length
        if len(response) > 250:
            response = response[:247] + "..."
        
        # Add sources
        if unique_urls and response != "N/A":
            source_text = f" [Sources: {', '.join(unique_urls)}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "N/A"

def generate_precise_relevance_analysis(company_data: Dict, company_name: str) -> tuple:
    """Generate precise relevance analysis based on actual company data"""
    
    # Build company profile from available data
    profile_parts = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\s*\[Source[^\]]*\]', '', value)  # Remove source tags
            profile_parts.append(clean_value)
    
    company_profile = "\n".join(profile_parts)
    
    relevance_prompt = f"""
    COMPANY: {company_name}
    
    COMPANY PROFILE:
    {company_profile}
    
    SYNTEL TARGET PROFILE:
    - Target Industries: Healthcare, Manufacturing, Warehouses, IT/ITES, GCC
    - Employee Range: 150-500+ employees  
    - Revenue: 100 Cr+
    - Solutions: Network Integration, Wi-Fi Deployment, Altai Wi-Fi
    
    ANALYSIS TASK:
    Analyze the company data and identify specific opportunities for Syntel's network integration and Wi-Fi solutions.
    
    REQUIREMENTS:
    1. Focus on concrete opportunities based on the company's expansion, facilities, or technology needs
    2. Match against Syntel's target industries and solutions
    3. Be specific and evidence-based
    4. Format as 3 bullet points without numbering
    5. Provide intent score: High/Medium/Low
    
    FORMAT:
    BULLETS:
    - [Specific opportunity with evidence from company data]
    - [Technology need matching Syntel solutions]
    - [Business alignment with target profile]
    SCORE: High/Medium/Low
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="Analyze company data for Syntel business opportunities. Be precise and evidence-based."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse response
        bullets = []
        score = "Medium"
        
        lines = response.split('\n')
        in_bullets = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('BULLETS:'):
                in_bullets = True
                continue
            elif line.startswith('SCORE:'):
                in_bullets = False
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
                continue
            
            if in_bullets and line.startswith('-') and len(line) > 10:
                clean_line = line[1:].strip()
                bullets.append(f"- {clean_line}")
        
        # Ensure we have 3 bullets
        while len(bullets) < 3:
            bullets.append(f"- Additional network integration opportunity identified")
        
        formatted_bullets = "\n".join(bullets[:3])
        return formatted_bullets, score
        
    except Exception as e:
        fallback_bullets = f"- {company_name} shows potential for network infrastructure solutions\n- Digital transformation initiatives present integration opportunities\n- Expansion activities indicate technology upgrade needs"
        return fallback_bullets, "Medium"

# --- Main Research Function ---
def comprehensive_company_research(company_name: str) -> Dict[str, Any]:
    """Main function for comprehensive company research"""
    
    company_data = {}
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    # Research each field
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"Researching {field.replace('_', ' ').title()} for {company_name}...")
        
        # Search and extract
        search_results = enhanced_search_for_field(company_name, field)
        field_data = precise_field_extraction(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1)
    
    # Generate relevance analysis
    status_text.text("Analyzing Syntel relevance...")
    progress_bar.progress(90)
    
    relevance_bullets, intent_score = generate_precise_relevance_analysis(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
    company_data["intent_scoring_level"] = intent_score
    
    progress_bar.progress(100)
    status_text.text("Research complete!")
    
    return company_data

# --- Clean Display Function ---
def create_research_dataframe(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Create clean dataframe without emojis or separators"""
    
    mapping = {
        "Company Name": "company_name",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url", 
        "Industry Category": "industry_category",
        "Employee Count": "employee_count_linkedin",
        "Headquarters Location": "headquarters_location",
        "Revenue Business Model": "revenue_source",
        "Branch Network Facilities": "branch_network_count",
        "Recent Expansion News": "expansion_news_12mo",
        "Digital Transformation": "digital_transformation_initiatives",
        "IT Leadership": "it_leadership_change",
        "Technology Vendors": "existing_network_vendors",
        "Network Upgrade Signals": "wifi_lan_tender_found",
        "IoT Automation": "iot_automation_edge_integration",
        "Cloud Adoption": "cloud_adoption_gcc_setup",
        "Physical Infrastructure": "physical_infrastructure_signals",
        "IT Infrastructure Budget": "it_infra_budget_capex",
        "Syntel Intent Score": "intent_scoring_level",
        "Syntel Relevance Analysis": "why_relevant_to_syntel_bullets",
    }
    
    data_list = []
    for display_col, data_field in mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(data_field, "N/A")
        
        # Clean the value
        if isinstance(value, str):
            # Remove any residual field names
            value = re.sub(r'^.*?:\s*', '', value)
            # Ensure it's not just URLs
            if value.startswith('http') and ' ' not in value:
                value = "N/A"
        
        # Format for display
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str) and value != "N/A":
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # Make URLs clickable
            if isinstance(value, str) and "http" in value:
                url_pattern = r'(\[Source: (https?://[^\]]+)\])'
                def make_clickable(match):
                    url = match.group(2)
                    return f'[<a href="{url}" target="_blank">Source</a>]'
                
                value_with_links = re.sub(url_pattern, make_clickable, value)
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{value_with_links}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel Company Intelligence",
    layout="wide",
    page_icon=""
)

st.title("Syntel Company Intelligence Research")
st.markdown("### Comprehensive Business Intelligence Reporting")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter company name for research:", "Neuberg Diagnostics")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("Start Comprehensive Research", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # FIX: Use company_input instead of company_name
    with st.spinner(f"Conducting comprehensive research for {company_input}..."):
        try:
            # Perform research
            company_data = comprehensive_company_research(company_input)
            
            # Display results
            st.success(f"Research complete for {company_input}")
            
            # Display final results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = create_research_dataframe(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show research metrics
            with st.expander("Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "N/A")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Fields with Sources", f"{sum(1 for field in REQUIRED_FIELDS[:-2] if 'Source' in str(company_data.get(field, '')))}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    st.metric("Intent Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download options
            st.subheader("Download Research Report")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyResearch')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(company_data, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_research_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_research_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_research_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
            
            # Store current research in history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": company_data
            }
            st.session_state.research_history.append(research_entry)
                        
        except Exception as e:
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits. Please try again in a few moments.")

# Research History
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"{research['company']} - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
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
    "Syntel Company Intelligence Research"
    "</div>",
    unsafe_allow_html=True
)
