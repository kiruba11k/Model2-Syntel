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

# --- Robust Search Function ---
def robust_company_search(company_name: str, field_name: str) -> List[Dict]:
    """Robust search function that ensures we get results"""
    
    # Comprehensive search queries for each field
    field_queries = {
        "linkedin_url": [
            f"{company_name} LinkedIn company page",
            f"{company_name} LinkedIn official",
            f"site:linkedin.com/company {company_name}"
        ],
        "company_website_url": [
            f"{company_name} official website",
            f"{company_name} company website",
            f"site:{company_name.replace(' ', '').lower()}.com {company_name}"
        ],
        "industry_category": [
            f"{company_name} industry business",
            f"What does {company_name} do",
            f"{company_name} business sector"
        ],
        "employee_count_linkedin": [
            f"{company_name} employees",
            f"{company_name} employee count",
            f"{company_name} number of employees"
        ],
        "headquarters_location": [
            f"{company_name} headquarters",
            f"{company_name} corporate office",
            f"Where is {company_name} located"
        ],
        "revenue_source": [
            f"{company_name} revenue",
            f"{company_name} financials",
            f"{company_name} annual revenue"
        ],
        "branch_network_count": [
            f"{company_name} branches locations",
            f"{company_name} facilities network",
            f"{company_name} offices"
        ],
        "expansion_news_12mo": [
            f"{company_name} expansion 2024",
            f"{company_name} new facilities",
            f"{company_name} growth news"
        ],
        "digital_transformation_initiatives": [
            f"{company_name} digital transformation",
            f"{company_name} IT initiatives",
            f"{company_name} technology upgrade"
        ],
        "it_leadership_change": [
            f"{company_name} CIO CTO",
            f"{company_name} IT leadership",
            f"{company_name} technology executives"
        ],
        "existing_network_vendors": [
            f"{company_name} technology vendors",
            f"{company_name} IT partners",
            f"{company_name} software systems"
        ],
        "wifi_lan_tender_found": [
            f"{company_name} network upgrade",
            f"{company_name} IT infrastructure",
            f"{company_name} network project"
        ],
        "iot_automation_edge_integration": [
            f"{company_name} IoT",
            f"{company_name} automation",
            f"{company_name} smart technology"
        ],
        "cloud_adoption_gcc_setup": [
            f"{company_name} cloud",
            f"{company_name} AWS Azure",
            f"{company_name} cloud computing"
        ],
        "physical_infrastructure_signals": [
            f"{company_name} new facility",
            f"{company_name} infrastructure",
            f"{company_name} construction"
        ],
        "it_infra_budget_capex": [
            f"{company_name} IT budget",
            f"{company_name} technology investment",
            f"{company_name} IT spending"
        ]
    }
    
    all_results = []
    queries = field_queries.get(field_name, [f"{company_name} {field_name}"])
    
    for query in queries:
        try:
            time.sleep(1)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    url = result.get('url', '')
                    
                    # Check if content is meaningful
                    if content and len(content) > 50:
                        all_results.append({
                            "title": title,
                            "content": content[:500],
                            "url": url
                        })
        except Exception as e:
            continue
    
    return all_results

# --- Smart Extraction Function ---
def smart_field_extraction(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Smart extraction that analyzes sources and extracts relevant information"""
    
    if not search_results:
        return "Information not available"
    
    # Build research context
    research_context = ""
    source_urls = []
    
    for i, result in enumerate(search_results):
        research_context += f"Source {i+1}:\n"
        research_context += f"Title: {result['title']}\n"
        research_context += f"Content: {result['content']}\n\n"
        source_urls.append(result['url'])
    
    # Field-specific extraction guidance
    extraction_guides = {
        "linkedin_url": "Extract the LinkedIn company page URL. Look for linkedin.com/company/ patterns.",
        "company_website_url": "Extract the official company website URL.",
        "industry_category": "What industry is this company in? Be specific about their core business.",
        "employee_count_linkedin": "How many employees does this company have? Provide specific numbers or ranges.",
        "headquarters_location": "Where is the company headquarters located? Provide city and state/country.",
        "revenue_source": "What is the company's revenue and business model? Provide specific financial information.",
        "branch_network_count": "How many branches, facilities, or locations does the company have?",
        "expansion_news_12mo": "What recent expansion or growth activities has the company undertaken?",
        "digital_transformation_initiatives": "What digital or IT projects is the company working on?",
        "it_leadership_change": "Who are the IT leaders (CIO, CTO, IT Director)?",
        "existing_network_vendors": "What technology vendors or partners does the company use?",
        "wifi_lan_tender_found": "Any network upgrade projects or IT infrastructure initiatives?",
        "iot_automation_edge_integration": "Any IoT, automation, or smart technology implementations?",
        "cloud_adoption_gcc_setup": "What cloud platforms does the company use?",
        "physical_infrastructure_signals": "Any new facilities, construction, or infrastructure developments?",
        "it_infra_budget_capex": "What is the company's IT budget or technology investment?"
    }
    
    guide = extraction_guides.get(field_name, f"Extract information about {field_name}")
    
    prompt = f"""
    Company: {company_name}
    Information Needed: {field_name.replace('_', ' ').title()}
    
    Instruction: {guide}
    
    Research Data:
    {research_context}
    
    Please analyze the research data and provide the requested information. 
    If the information is clearly available, provide it with specific details.
    If the information is not found after careful analysis, say "Information not available".
    
    Extracted Information:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a research analyst. Extract specific information from the provided research data. Be factual and detailed."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Clean the response
        response = re.sub(r'^(.*?):\s*', '', response)
        response = response.strip()
        
        # Validate response
        if (not response or 
            len(response) < 5 or
            any(phrase in response.lower() for phrase in ['not available', 'not found', 'no information', 'n/a'])):
            return "Information not available"
        
        # Add source attribution
        unique_urls = list(set(source_urls))[:2]
        if unique_urls and response != "Information not available":
            source_text = f" [Sources: {', '.join(unique_urls)}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "Information not available"

# --- Comprehensive Relevance Analysis ---
def comprehensive_syntel_analysis(company_data: Dict, company_name: str) -> tuple:
    """Comprehensive analysis for Syntel relevance"""
    
    # Build evidence base from available data
    evidence_base = []
    for field, value in company_data.items():
        if value and value != "Information not available" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\s*\[Source[^\]]*\]', '', value)
            evidence_base.append(f"{field}: {clean_value}")
    
    evidence_text = "\n".join(evidence_base) if evidence_base else "Company operational data available"
    
    analysis_prompt = f"""
    COMPANY ANALYSIS: {company_name}
    
    EVIDENCE BASE:
    {evidence_text}
    
    SYNTEL SOLUTIONS FOCUS:
    - Network Integration & Wi-Fi Deployment
    - Altai Wi-Fi (3-5x coverage advantage)
    - IT Infrastructure Modernization
    - Multi-vendor Implementation
    - Digital Transformation Services
    
    TARGET PROFILE MATCHING:
    - Industries: Warehouses, Logistics, Manufacturing, Healthcare, IT/ITES
    - Company Size: 150+ employees preferred
    - Growth Signals: Expansion, Modernization, Infrastructure projects
    
    OPPORTUNITY IDENTIFICATION TASK:
    Based on the evidence above, identify 3 SPECIFIC opportunities where Syntel can provide value.
    
    Each opportunity must:
    1. Reference specific evidence from the company data
    2. Connect to exact Syntel solutions
    3. Explain the business value
    4. Be actionable and concrete
    
    FORMAT:
    1. [Specific company evidence] → [Syntel solution with benefit]
    2. [Specific company evidence] → [Syntel solution with benefit]
    3. [Specific company evidence] → [Syntel solution with benefit]
    
    SCORE: High/Medium/Low (based on strategic fit and opportunity size)
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business development analyst. Identify specific, evidence-based opportunities by connecting company data to solutions. Be concrete and avoid generic statements."),
            HumanMessage(content=analysis_prompt)
        ]).content
        
        # Parse opportunities and score
        opportunities = []
        score = "Medium"
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line) and '→' in line:
                # Clean and format the opportunity
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                opportunities.append(f"- {clean_line}")
            elif 'SCORE:' in line.upper():
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
        
        # Ensure we have 3 quality opportunities
        if len(opportunities) < 3:
            # Create evidence-based fallbacks
            available_fields = [field for field, value in company_data.items() 
                              if value and value != "Information not available"]
            
            fallbacks = []
            if any(field in available_fields for field in ['expansion_news_12mo', 'branch_network_count', 'physical_infrastructure_signals']):
                fallbacks.append("- Company expansion and facility growth → Network infrastructure scaling with Altai Wi-Fi for new locations providing 3-5x better coverage")
            
            if 'digital_transformation_initiatives' in available_fields:
                fallbacks.append("- Digital transformation initiatives → IT infrastructure modernization and network integration for improved operational efficiency")
            
            if any(field in available_fields for field in ['iot_automation_edge_integration', 'existing_network_vendors']):
                fallbacks.append("- Technology adoption and automation → Network solutions to support IoT and automation systems with reliable connectivity")
            
            # Fill remaining opportunities
            while len(opportunities) < 3 and fallbacks:
                opportunities.append(fallbacks.pop(0))
            
            # Generic but relevant fallbacks
            while len(opportunities) < 3:
                opportunities.append(f"- Business operations and scale → Network integration and Wi-Fi solutions for operational connectivity and efficiency")
        
        formatted_opportunities = "\n".join(opportunities[:3])
        return formatted_opportunities, score
        
    except Exception as e:
        # Quality fallback opportunities
        fallbacks = [
            f"- {company_name}'s operational infrastructure → Network integration and Wi-Fi deployment for business connectivity",
            f"- Technology and business needs → IT infrastructure solutions for operational efficiency and growth",
            f"- Organizational operations → Enterprise network solutions for reliable connectivity and productivity"
        ]
        return "\n".join(fallbacks), "Medium"

# --- Main Research Engine ---
def execute_company_research(company_name: str) -> Dict[str, Any]:
    """Main function to execute comprehensive company research"""
    
    company_data = {}
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    # Research each field
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"Researching {field.replace('_', ' ')}...")
        
        # Search and extract
        search_results = robust_company_search(company_name, field)
        field_data = smart_field_extraction(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1)
    
    # Generate Syntel relevance analysis
    status_text.text("Analyzing Syntel opportunities...")
    progress_bar.progress(90)
    
    opportunities, score = comprehensive_syntel_analysis(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = opportunities
    company_data["intent_scoring_level"] = score
    
    progress_bar.progress(100)
    status_text.text("Research complete!")
    
    return company_data

# --- Results Display ---
def display_research_report(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Display research results in professional format"""
    
    display_mapping = {
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
    for display_col, data_field in display_mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(data_field, "Information not available")
        
        # Clean the value
        if isinstance(value, str) and value != "Information not available":
            value = re.sub(r'^(?:the|this|company)\s+', '', value, flags=re.IGNORECASE)
        
        # Special formatting for opportunities
        if data_field == "why_relevant_to_syntel_bullets":
            if value != "Information not available":
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
                value_with_links = re.sub(r'\[Sources: ([^\]]+)\]', make_clickable, value_with_links)
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{value_with_links}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit Application ---
st.set_page_config(
    page_title="Company Intelligence Platform",
    layout="wide",
    page_icon="🔍"
)

st.title("🔍 Company Intelligence Research Platform")
st.markdown("### Comprehensive Business Intelligence with Syntel Opportunity Mapping")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter company name for research:", "Snowman Logistics")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    research_button = st.button("🚀 Start Comprehensive Research", type="primary")

if research_button:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"Conducting comprehensive research for {company_input}..."):
        try:
            # Execute research
            company_data = execute_company_research(company_input)
            
            # Display success
            st.success(f"✅ Research completed for {company_input}")
            
            # Display results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = display_research_report(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Research metrics
            with st.expander("📊 Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "Information not available")
                
                meaningful_fields = sum(1 for field in REQUIRED_FIELDS[:-2] 
                                     if company_data.get(field) and 
                                     company_data.get(field) != "Information not available" and
                                     len(str(company_data.get(field)).split()) > 4)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Researched", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Quality Data Points", f"{meaningful_fields}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    st.metric("Opportunity Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download section
            st.subheader("💾 Download Research Report")
            
            def create_excel_file(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyIntelligence')
                return output.getvalue()
            
            col_json, col_csv, col_excel = st.columns(3)
            
            with col_json:
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(company_data, indent=2),
                    file_name=f"{company_input.replace(' ', '_')}_intelligence.json",
                    mime="application/json"
                )

            with col_csv:
                csv_data = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{company_input.replace(' ', '_')}_intelligence.csv",
                    mime="text/csv"
                )
                
            with col_excel:
                excel_data = create_excel_file(final_df)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{company_input.replace(' ', '_')}_intelligence.xlsx",
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
            st.info("Please check your API keys and try again.")

# Research History
if st.session_state.research_history:
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"{research['company']} - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Score: {research['data'].get('intent_scoring_level', 'Medium')}")
            completed = sum(1 for field in REQUIRED_FIELDS 
                         if research['data'].get(field) and 
                         research['data'].get(field) != "Information not available")
            st.write(f"Fields: {completed}/{len(REQUIRED_FIELDS)}")
        
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Company Intelligence Research Platform | Syntel Opportunity Mapping"
    "</div>",
    unsafe_allow_html=True
)
