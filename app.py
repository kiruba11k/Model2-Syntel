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

# --- Smart Search with Multiple Query Strategies ---
def smart_field_search(company_name: str, field_name: str) -> List[Dict]:
    """Smart search with multiple query strategies for each field"""
    
    field_search_strategies = {
        "linkedin_url": [
            f'"{company_name}" LinkedIn company',
            f'"{company_name}" LinkedIn page',
            f'"{company_name}" LinkedIn official'
        ],
        "company_website_url": [
            f'"{company_name}" official website',
            f'"{company_name}" company website',
            f'site:{company_name}.com'
        ],
        "industry_category": [
            f'"{company_name}" industry',
            f'"{company_name}" business type',
            f'"{company_name}" sector'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employees',
            f'"{company_name}" employee count',
            f'"{company_name}" workforce size'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters',
            f'"{company_name}" corporate office',
            f'"{company_name}" main office location'
        ],
        "revenue_source": [
            f'"{company_name}" revenue',
            f'"{company_name}" financials',
            f'"{company_name}" business model'
        ],
        "branch_network_count": [
            f'"{company_name}" branches',
            f'"{company_name}" locations',
            f'"{company_name}" facilities network'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024',
            f'"{company_name}" new facilities',
            f'"{company_name}" growth news'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation',
            f'"{company_name}" IT modernization',
            f'"{company_name}" technology upgrade'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO',
            f'"{company_name}" IT leadership',
            f'"{company_name}" technology executives'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors',
            f'"{company_name}" IT partners',
            f'"{company_name}" software vendors'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" network upgrade',
            f'"{company_name}" WiFi project',
            f'"{company_name}" IT infrastructure tender'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT',
            f'"{company_name}" automation',
            f'"{company_name}" smart technology'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud',
            f'"{company_name}" AWS Azure',
            f'"{company_name}" cloud migration'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new facility',
            f'"{company_name}" construction',
            f'"{company_name}" infrastructure expansion'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget',
            f'"{company_name}" technology investment',
            f'"{company_name}" capex IT'
        ]
    }
    
    all_results = []
    queries = field_search_strategies.get(field_name, [f'"{company_name}" {field_name}'])
    
    for query in queries:
        try:
            time.sleep(1.2)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    # Enhanced company matching
                    company_words = company_name.lower().split()
                    content_lower = content.lower()
                    title_lower = title.lower()
                    
                    match_score = sum(2 for word in company_words if word in content_lower)
                    match_score += sum(1 for word in company_words if word in title_lower)
                    
                    # Only include highly relevant results
                    if match_score >= 2 and len(content) > 80:
                        all_results.append({
                            "title": title,
                            "content": content[:600],
                            "url": result.get('url', ''),
                            "field": field_name,
                            "score": match_score
                        })
        except Exception as e:
            continue
    
    # Sort by relevance score and return
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return all_results[:4]  # Return top 4 most relevant results

# --- Intelligent Field Extraction ---
def intelligent_field_extraction(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Intelligent extraction that deeply analyzes sources and extracts relevant information"""
    
    if not search_results:
        return "Information not available"
    
    # Build comprehensive analysis context
    analysis_context = "RESEARCH SOURCES TO ANALYZE:\n\n"
    for i, result in enumerate(search_results):
        analysis_context += f"SOURCE {i+1}:\n"
        analysis_context += f"Title: {result['title']}\n"
        analysis_context += f"Content: {result['content']}\n"
        analysis_context += f"URL: {result['url']}\n\n"
    
    # Field-specific analysis instructions
    field_analysis_guides = {
        "linkedin_url": "Find and extract the actual LinkedIn company profile URL. Look for linkedin.com/company/ patterns.",
        "company_website_url": "Find and extract the main company website URL. Look for official corporate domains.",
        "industry_category": "Analyze what industry this company operates in. Be specific about their core business.",
        "employee_count_linkedin": "Extract employee count information from reliable sources. Provide specific numbers or ranges.",
        "headquarters_location": "Find the company's headquarters location with city and state/country details.",
        "revenue_source": "Extract revenue figures and understand the business model. Provide specific financial information.",
        "branch_network_count": "Analyze the company's physical presence - branches, facilities, locations with counts.",
        "expansion_news_12mo": "Find recent expansion activities, new openings, or growth initiatives in the last year.",
        "digital_transformation_initiatives": "Identify digital transformation projects, IT modernization efforts, and technology upgrades.",
        "it_leadership_change": "Find information about IT leadership roles, appointments, or organizational changes.",
        "existing_network_vendors": "Identify technology vendors, software partners, and IT infrastructure providers used.",
        "wifi_lan_tender_found": "Look for network upgrade projects, WiFi deployments, or IT infrastructure tenders.",
        "iot_automation_edge_integration": "Identify IoT implementations, automation systems, and smart technology usage.",
        "cloud_adoption_gcc_setup": "Find cloud adoption strategies, cloud platforms used, and global capability centers.",
        "physical_infrastructure_signals": "Identify new construction, facility expansions, or physical infrastructure developments.",
        "it_infra_budget_capex": "Extract IT budget information, technology investments, and capital expenditure details."
    }
    
    analysis_guide = field_analysis_guides.get(field_name, f"Extract relevant information about {field_name}")
    
    # Get unique source URLs for attribution
    unique_urls = list(set([result['url'] for result in search_results]))[:2]
    
    extraction_prompt = f"""
    COMPANY: {company_name}
    INFORMATION NEEDED: {field_name.replace('_', ' ').title()}
    
    ANALYSIS TASK: {analysis_guide}
    
    AVAILABLE RESEARCH DATA:
    {analysis_context}
    
    EXTRACTION REQUIREMENTS:
    1. Thoroughly analyze ALL provided sources
    2. Extract the most relevant and specific information
    3. Provide factual, quantitative information when available
    4. Do NOT include field names or labels in your response
    5. If information is not found after careful analysis, say "Information not available"
    6. Be concise but comprehensive
    7. Focus only on information that clearly relates to {company_name}
    
    EXTRACTED INFORMATION:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a meticulous research analyst. Carefully analyze all provided sources and extract the most relevant, specific information. Never include field names or generic labels in your responses."),
            HumanMessage(content=extraction_prompt)
        ]).content.strip()
        
        # Advanced response cleaning
        response = re.sub(r'^.*?(?:information|data|result|extracted|analysis):\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^(?:the|this|company)\s+', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # Validate response quality
        if (not response or 
            len(response) < 10 or
            any(phrase in response.lower() for phrase in [
                'not available', 'not found', 'no information', 'n/a', 
                'unavailable', 'unknown', 'not specified'
            ])):
            return "Information not available"
        
        # Ensure meaningful content (not just URLs or sources)
        words = response.split()
        if len(words) < 5:
            return "Information not available"
        
        # Add source attribution for credible information
        if unique_urls and response != "Information not available":
            source_text = f" [Sources: {', '.join(unique_urls)}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "Information not available"

# --- Enhanced Relevance Analysis ---
def enhanced_syntel_relevance_analysis(company_data: Dict, company_name: str) -> tuple:
    """Enhanced relevance analysis with specific Syntel opportunity mapping"""
    
    # Build detailed evidence base from company data
    evidence_base = []
    
    # Map company data to Syntel opportunity areas
    opportunity_areas = {
        "expansion_news_12mo": "Expansion and Growth",
        "branch_network_count": "Network Infrastructure", 
        "digital_transformation_initiatives": "Digital Transformation",
        "physical_infrastructure_signals": "Physical Infrastructure",
        "it_infra_budget_capex": "IT Investment",
        "iot_automation_edge_integration": "Automation Needs",
        "cloud_adoption_gcc_setup": "Cloud Services",
        "employee_count_linkedin": "Organization Scale",
        "revenue_source": "Financial Capacity"
    }
    
    for field, area in opportunity_areas.items():
        value = company_data.get(field, "Information not available")
        if value != "Information not available":
            clean_value = re.sub(r'\s*\[Source[^\]]*\]', '', value)
            evidence_base.append(f"{area}: {clean_value}")
    
    evidence_text = "\n".join(evidence_base) if evidence_base else "Limited company data available"
    
    relevance_prompt = f"""
    COMPANY: {company_name}
    
    EVIDENCE BASE FOR ANALYSIS:
    {evidence_text}
    
    SYNTEL SOLUTIONS FOCUS:
    - Network Integration & Wi-Fi Deployment
    - Altai Wi-Fi Solutions (3-5x better coverage)
    - Multi-vendor Implementation Support
    - IT Infrastructure Modernization
    - Digital Transformation Services
    
    TARGET PROFILE MATCH:
    - Industries: Healthcare, Manufacturing, Warehouses, IT/ITES, GCC
    - Company Size: 150+ employees, 100Cr+ revenue
    - Growth Signals: Expansion, Modernization, Infrastructure projects
    
    ANALYSIS TASK:
    Based on the evidence above, identify 3 SPECIFIC, EVIDENCE-BASED opportunities for Syntel.
    Each opportunity must:
    1. Reference specific company data from the evidence base
    2. Map to exact Syntel solutions
    3. Be actionable and concrete
    4. Avoid generic statements
    
    FORMAT:
    OPPORTUNITY 1: [Specific evidence from company data] → [Specific Syntel solution with benefit]
    OPPORTUNITY 2: [Specific evidence from company data] → [Specific Syntel solution with benefit] 
    OPPORTUNITY 3: [Specific evidence from company data] → [Specific Syntel solution with benefit]
    SCORE: High/Medium/Low
    
    Be precise and evidence-driven.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business development analyst. Identify specific, evidence-based opportunities by mapping company data to exact solutions. Be concrete and avoid generic statements."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse opportunities and score
        opportunities = []
        score = "Medium"
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('OPPORTUNITY') and ':' in line:
                # Extract the opportunity content
                content = line.split(':', 1)[1].strip()
                if len(content) > 20:  # Meaningful content
                    opportunities.append(f"- {content}")
            elif line.startswith('SCORE:'):
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
        
        # Ensure we have 3 quality opportunities
        if len(opportunities) < 3:
            # Create evidence-based fallback opportunities
            fallbacks = []
            
            # Check for expansion signals
            if company_data.get('expansion_news_12mo', 'Information not available') != "Information not available":
                fallbacks.append("- Company expansion activities → Network infrastructure scaling with Altai Wi-Fi for new facilities")
            
            # Check for digital transformation
            if company_data.get('digital_transformation_initiatives', 'Information not available') != "Information not available":
                fallbacks.append("- Digital transformation initiatives → IT infrastructure modernization and network integration services")
            
            # Check for physical infrastructure
            if company_data.get('physical_infrastructure_signals', 'Information not available') != "Information not available":
                fallbacks.append("- Physical infrastructure growth → Wi-Fi deployment and network solutions for expanded facilities")
            
            # Check for employee scale
            if company_data.get('employee_count_linkedin', 'Information not available') != "Information not available":
                fallbacks.append("- Organizational scale and operations → Enterprise-grade network solutions for improved connectivity")
            
            # Fill remaining opportunities
            while len(opportunities) < 3 and fallbacks:
                opportunities.append(fallbacks.pop(0))
            
            # Generic but relevant fallbacks
            while len(opportunities) < 3:
                opportunities.append(f"- Operational technology needs → Network integration and Wi-Fi solutions for business operations")
        
        formatted_opportunities = "\n".join(opportunities[:3])
        return formatted_opportunities, score
        
    except Exception as e:
        # Evidence-based fallback
        fallback_opportunities = [
            f"- {company_name}'s business operations → Network infrastructure and connectivity solutions",
            f"- Technology modernization needs → IT infrastructure services and network upgrades", 
            f"- Organizational scale and growth → Enterprise Wi-Fi and network integration services"
        ]
        return "\n".join(fallback_opportunities), "Medium"

# --- Main Research Engine ---
def comprehensive_company_analysis(company_name: str) -> Dict[str, Any]:
    """Main function that performs comprehensive company analysis"""
    
    company_data = {}
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    # Analyze each field with intelligent extraction
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"Analyzing {field.replace('_', ' ')} for {company_name}...")
        
        # Smart search and intelligent extraction
        search_results = smart_field_search(company_name, field)
        field_data = intelligent_field_extraction(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1.2)
    
    # Generate enhanced relevance analysis
    status_text.text("Identifying Syntel opportunities...")
    progress_bar.progress(90)
    
    relevance_opportunities, intent_score = enhanced_syntel_relevance_analysis(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = relevance_opportunities
    company_data["intent_scoring_level"] = intent_score
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    return company_data

# --- Clean Display Formatting ---
def format_analysis_results(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Format analysis results in clean, professional layout"""
    
    display_mapping = {
        "Company Name": "company_name",
        "LinkedIn Profile": "linkedin_url",
        "Website": "company_website_url", 
        "Industry": "industry_category",
        "Employee Count": "employee_count_linkedin",
        "Headquarters": "headquarters_location",
        "Revenue & Business": "revenue_source",
        "Branch Network": "branch_network_count",
        "Recent Expansion": "expansion_news_12mo",
        "Digital Initiatives": "digital_transformation_initiatives",
        "IT Leadership": "it_leadership_change",
        "Technology Partners": "existing_network_vendors",
        "Network Projects": "wifi_lan_tender_found",
        "Automation & IoT": "iot_automation_edge_integration",
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
        
        # Clean and format the value
        if isinstance(value, str) and value != "Information not available":
            # Remove any introductory phrases and ensure clean content
            value = re.sub(r'^(?:the|this|company|organization)\s+', '', value, flags=re.IGNORECASE)
        
        # Special formatting for opportunities
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str) and value != "Information not available":
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # Make sources clickable while keeping content
            if isinstance(value, str) and "Source:" in value:
                # Extract content and sources separately
                content_part = re.sub(r'\s*\[Source[^\]]*\]', '', value)
                sources = re.findall(r'\[Source: ([^\]]+)\]', value)
                
                if sources:
                    source_links = " ".join([f'[<a href="{source}" target="_blank">Source</a>]' for source in sources])
                    display_value = f"{content_part} {source_links}"
                    data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{display_value}</div>'})
                else:
                    data_list.append({"Column Header": display_col, "Value": content_part})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit Application ---
st.set_page_config(
    page_title="Syntel Business Intelligence",
    layout="wide",
    page_icon=""
)

st.title("Syntel Business Intelligence Analysis")
st.markdown("### Comprehensive Company Intelligence with Opportunity Mapping")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter company name for analysis:", "Neuberg Diagnostics")
with col2:
    with st.form("analysis_form"):
        submitted = st.form_submit_button("Start Comprehensive Analysis", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"Conducting comprehensive analysis for {company_input}..."):
        try:
            # Perform comprehensive analysis
            company_data = comprehensive_company_analysis(company_input)
            
            # Display results
            st.success(f"Analysis complete for {company_input}")
            
            # Display analysis results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = format_analysis_results(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show analysis metrics
            with st.expander("Analysis Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "Information not available")
                
                meaningful_fields = sum(1 for field in REQUIRED_FIELDS[:-2] 
                                     if company_data.get(field) and 
                                     company_data.get(field) != "Information not available" and
                                     len(str(company_data.get(field)).split()) > 4)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Analyzed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Quality Data Points", f"{meaningful_fields}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    st.metric("Opportunity Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download options
            st.subheader("Download Analysis Report")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='BusinessIntelligence')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(company_data, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_business_intelligence.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_business_intelligence.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_business_intelligence.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
            
            # Store analysis in history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": company_data
            }
            st.session_state.research_history.append(research_entry)
                        
        except Exception as e:
            st.error(f"Analysis failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits. Please try again in a few moments.")

# Analysis History
if st.session_state.research_history:
    st.sidebar.header("Analysis History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"{research['company']} - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Opportunity Score: {research['data'].get('intent_scoring_level', 'Medium')}")
            completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                if research['data'].get(field) and 
                                research['data'].get(field) != "Information not available")
            st.write(f"Fields Analyzed: {completed_fields}/{len(REQUIRED_FIELDS)}")
        
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Syntel Business Intelligence Analysis"
    "</div>",
    unsafe_allow_html=True
)
