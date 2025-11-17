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

# --- Deep Research Agent ---
def deep_research_agent(company_name: str, field_name: str) -> str:
    """Each agent deeply researches its specific field by reading and analyzing sources"""
    
    # Field-specific search strategies
    search_strategies = {
        "linkedin_url": [
            f'"{company_name}" LinkedIn company page',
            f'"{company_name}" LinkedIn official',
            f'"{company_name}" LinkedIn profile'
        ],
        "company_website_url": [
            f'"{company_name}" official website',
            f'"{company_name}" company website', 
            f'site:{company_name.replace(" ", "").lower()}.com'
        ],
        "industry_category": [
            f'"{company_name}" industry',
            f'"{company_name}" business type',
            f'"{company_name}" sector',
            f'what does {company_name} do'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employees',
            f'"{company_name}" employee count',
            f'"{company_name}" number of employees',
            f'"{company_name}" workforce size'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters',
            f'"{company_name}" corporate office',
            f'"{company_name}" main office',
            f'where is {company_name} located'
        ],
        "revenue_source": [
            f'"{company_name}" revenue',
            f'"{company_name}" financials',
            f'"{company_name}" business model',
            f'"{company_name}" annual revenue'
        ],
        "branch_network_count": [
            f'"{company_name}" branches',
            f'"{company_name}" locations',
            f'"{company_name}" facilities',
            f'"{company_name}" offices network'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024',
            f'"{company_name}" new facilities 2024',
            f'"{company_name}" growth news',
            f'"{company_name}" recent expansion'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation',
            f'"{company_name}" IT initiatives',
            f'"{company_name}" technology projects',
            f'"{company_name}" digital initiatives'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO',
            f'"{company_name}" IT leadership',
            f'"{company_name}" technology executives',
            f'"{company_name}" IT management'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors',
            f'"{company_name}" IT partners',
            f'"{company_name}" software systems',
            f'"{company_name}" technology stack'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" network upgrade',
            f'"{company_name}" IT infrastructure',
            f'"{company_name}" network project',
            f'"{company_name}" WiFi deployment'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT',
            f'"{company_name}" automation',
            f'"{company_name}" smart technology',
            f'"{company_name}" robotics'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud',
            f'"{company_name}" AWS Azure',
            f'"{company_name}" cloud computing',
            f'"{company_name}" cloud migration'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new facility',
            f'"{company_name}" infrastructure',
            f'"{company_name}" construction',
            f'"{company_name}" building expansion'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget',
            f'"{company_name}" technology investment',
            f'"{company_name}" IT spending',
            f'"{company_name}" capex IT'
        ]
    }
    
    # Collect all research data
    all_research_data = []
    source_urls = []
    
    queries = search_strategies.get(field_name, [f'"{company_name}" {field_name}'])
    
    for query in queries[:3]:  # Try 3 different queries
        try:
            time.sleep(1.5)
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            for result in results:
                content = result.get('content', '')
                url = result.get('url', '')
                
                # Only include if it mentions the company
                if company_name.lower() in content.lower():
                    all_research_data.append({
                        'content': content[:800],  # More content for deep analysis
                        'url': url,
                        'query': query
                    })
                    if url not in source_urls:
                        source_urls.append(url)
        except Exception as e:
            continue
    
    if not all_research_data:
        return "Information not available from research"
    
    # Build comprehensive research context
    research_context = "DETAILED RESEARCH FINDINGS:\n\n"
    for i, research in enumerate(all_research_data):
        research_context += f"RESEARCH SOURCE {i+1} (from query: '{research['query']}'):\n"
        research_context += f"CONTENT: {research['content']}\n"
        research_context += f"URL: {research['url']}\n\n"
    
    # Field-specific analysis instructions
    analysis_instructions = {
        "linkedin_url": """
        Carefully read through all research findings to find the LinkedIn company page URL.
        Look for patterns like linkedin.com/company/ or specific LinkedIn profile links.
        Return ONLY the complete LinkedIn URL if found.
        """,
        
        "company_website_url": """
        Thoroughly analyze all research to find the official company website URL.
        Look for main domain URLs, corporate websites, not subpages.
        Return ONLY the main website URL if found.
        """,
        
        "industry_category": """
        Deeply analyze the research to determine the primary industry and business focus.
        Look for specific industry mentions, business descriptions, core activities.
        Be specific: "Healthcare Diagnostics", "Logistics Services", "IT Consulting"
        Return 2-4 word industry description.
        """,
        
        "employee_count_linkedin": """
        Carefully examine all research for employee count information.
        Look for specific numbers, ranges, LinkedIn employee data, workforce size.
        Return the employee count with context like "500-1000 employees" or specific number.
        """,
        
        "headquarters_location": """
        Deeply analyze the research to find headquarters location details.
        Look for city, state, country information, corporate office addresses.
        Format as "City, State" or "City, Country".
        Return specific location information.
        """,
        
        "revenue_source": """
        Thoroughly examine research for revenue and business model information.
        Look for revenue numbers, financial performance, business model descriptions.
        Be quantitative and specific with numbers if available.
        """,
        
        "branch_network_count": """
        Deeply analyze the research for branch network and facility information.
        Look for numbers of locations, branches, facilities, geographical presence.
        Provide specific counts and descriptions of operational footprint.
        """,
        
        "expansion_news_12mo": """
        Carefully examine research for recent expansion activities.
        Look for new facilities, growth initiatives, expansion news from last 12-24 months.
        Include specific locations, investments, timelines.
        """,
        
        "digital_transformation_initiatives": """
        Deeply analyze research for digital transformation projects.
        Look for IT modernization, digital initiatives, technology projects, system implementations.
        Describe specific technologies and initiatives mentioned.
        """,
        
        "it_leadership_change": """
        Thoroughly examine research for IT leadership information.
        Look for CIO, CTO, IT Director names, appointments, organizational changes.
        Provide specific names and positions if mentioned.
        """,
        
        "existing_network_vendors": """
        Deeply analyze research for technology vendors and partners.
        Look for mentions of specific vendors: Cisco, VMware, SAP, Oracle, Microsoft, AWS, etc.
        List the specific technology vendors and platforms used.
        """,
        
        "wifi_lan_tender_found": """
        Carefully examine research for network infrastructure projects.
        Look for WiFi deployments, network upgrades, IT infrastructure tenders, connectivity projects.
        Describe any network-related initiatives found.
        """,
        
        "iot_automation_edge_integration": """
        Deeply analyze research for IoT, automation, and smart technology implementations.
        Look for IoT sensors, automation systems, robotics, smart technology usage.
        Describe specific automation and IoT capabilities mentioned.
        """,
        
        "cloud_adoption_gcc_setup": """
        Thoroughly examine research for cloud adoption and global operations.
        Look for AWS, Azure, Google Cloud usage, cloud migration, global capability centers.
        Describe cloud strategy and global operations mentioned.
        """,
        
        "physical_infrastructure_signals": """
        Deeply analyze research for physical infrastructure developments.
        Look for new construction, facility expansions, building projects, infrastructure growth.
        Describe physical infrastructure developments found.
        """,
        
        "it_infra_budget_capex": """
        Carefully examine research for IT budget and investment information.
        Look for technology spending, IT investments, capex announcements, budget details.
        Provide specific budget information if available.
        """
    }
    
    instruction = analysis_instructions.get(field_name, f"Extract comprehensive information about {field_name}")
    
    analysis_prompt = f"""
    TASK: Deep Research Analysis for {field_name.replace('_', ' ').title()}
    COMPANY: {company_name}
    
    RESEARCH INSTRUCTION:
    {instruction}
    
    AVAILABLE RESEARCH DATA:
    {research_context}
    
    ANALYSIS APPROACH:
    1. Read and comprehend ALL research sources thoroughly
    2. Extract the most relevant and specific information for the requested field
    3. Provide detailed, factual information based on the research
    4. If multiple sources contain information, synthesize the most accurate data
    5. Be specific, quantitative, and detailed where possible
    6. If after thorough analysis no relevant information is found, state "Information not available from research"
    
    EXTRACTED INFORMATION:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a deep research analyst. Thoroughly read and analyze all research sources. Extract specific, detailed information. Provide comprehensive answers based on the research findings."),
            HumanMessage(content=analysis_prompt)
        ]).content.strip()
        
        # Clean the response while keeping meaningful content
        response = re.sub(r'^(.*?):\s*', '', response)
        response = response.strip()
        
        # Validate we have meaningful information
        if (not response or 
            len(response) < 10 or
            any(phrase in response.lower() for phrase in [
                'not available', 'not found', 'no information', 'n/a', 
                'unavailable', 'unknown', 'not specified', 'none found'
            ])):
            return "Information not available from research"
        
        # Add source attribution
        if source_urls and response != "Information not available from research":
            source_text = f" [Research Sources: {', '.join(source_urls[:2])}]" if len(source_urls) > 1 else f" [Research Source: {source_urls[0]}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "Information not available from research"

# --- Enhanced Syntel Relevance Analysis ---
def deep_syntel_analysis(company_data: Dict, company_name: str) -> tuple:
    """Deep analysis for Syntel relevance based on all available company data"""
    
    # Build comprehensive company profile
    company_profile = "COMPREHENSIVE COMPANY PROFILE:\n\n"
    for field, value in company_data.items():
        if value and value != "Information not available from research" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            clean_value = re.sub(r'\s*\[Research Source[^\]]*\]', '', value)
            company_profile += f"{field.replace('_', ' ').title()}: {clean_value}\n\n"
    
    analysis_prompt = f"""
    COMPANY: {company_name}
    
    {company_profile}
    
    SYNTEL BUSINESS OPPORTUNITY ANALYSIS:
    
    Syntel specializes in:
    - Network Integration & Wi-Fi Solutions (Altai Wi-Fi with 3-5x coverage advantage)
    - IT Infrastructure Modernization
    - Digital Transformation Services
    - Multi-vendor Implementation Support
    
    Target Customer Profile:
    - Industries: Healthcare, Manufacturing, Warehouses, IT/ITES, GCC, Education
    - Size: 150+ employees, 100Cr+ revenue
    - Growth Signals: Expansion, Digital Initiatives, Infrastructure Projects
    
    ANALYSIS TASK:
    Based on the comprehensive company profile above, identify 3 SPECIFIC, EVIDENCE-BASED opportunities for Syntel.
    
    For each opportunity:
    1. Reference specific data points from the company profile
    2. Connect to exact Syntel solutions and capabilities
    3. Explain the business value and relevance
    4. Be concrete and actionable
    
    OPPORTUNITIES:
    1. [Specific opportunity based on company data] → [Syntel solution with quantified benefit]
    2. [Specific opportunity based on company data] → [Syntel solution with quantified benefit]
    3. [Specific opportunity based on company data] → [Syntel solution with quantified benefit]
    
    SCORE: High/Medium/Low (based on alignment with Syntel's target profile and solution fit)
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business analyst. Analyze company data deeply and identify specific, evidence-based opportunities. Connect company needs directly to solutions with clear business value."),
            HumanMessage(content=analysis_prompt)
        ]).content
        
        # Parse the response
        opportunities = []
        score = "Medium"
        
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if re.match(r'^\d+\.', line) and '→' in line:
                opportunities.append(f"- {line}")
            elif 'SCORE:' in line.upper():
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
        
        # Ensure we have 3 quality opportunities
        if len(opportunities) < 3:
            # Create evidence-based fallbacks
            available_data = []
            for field, value in company_data.items():
                if value and value != "Information not available from research":
                    available_data.append(field)
            
            fallbacks = []
            if 'expansion_news_12mo' in available_data or 'branch_network_count' in available_data:
                fallbacks.append("- Company expansion activities → Network infrastructure scaling with Altai Wi-Fi for new facilities providing 3-5x better coverage")
            
            if 'digital_transformation_initiatives' in available_data:
                fallbacks.append("- Digital transformation initiatives → IT infrastructure modernization and network integration services for improved operational efficiency")
            
            if 'physical_infrastructure_signals' in available_data:
                fallbacks.append("- Physical infrastructure growth → Wi-Fi deployment and network solutions for expanded facilities requiring reliable connectivity")
            
            if 'employee_count_linkedin' in available_data:
                fallbacks.append("- Organizational scale and operations → Enterprise-grade network solutions for improved employee connectivity and productivity")
            
            # Fill remaining
            while len(opportunities) < 3 and fallbacks:
                opportunities.append(fallbacks.pop(0))
            
            while len(opportunities) < 3:
                opportunities.append(f"- Business operations and technology needs → Network integration and Wi-Fi solutions for operational efficiency and connectivity")
        
        formatted_opportunities = "\n".join(opportunities[:3])
        return formatted_opportunities, score
        
    except Exception as e:
        opportunities = [
            f"- {company_name}'s business operations → Network infrastructure and connectivity solutions for improved operations",
            f"- Technology and infrastructure needs → IT modernization services and network upgrades for business efficiency",
            f"- Organizational scale and growth → Enterprise Wi-Fi and network integration services for reliable connectivity"
        ]
        return "\n".join(opportunities), "Medium"

# --- Main Research Execution ---
def execute_deep_research(company_name: str) -> Dict[str, Any]:
    """Execute deep research for all fields"""
    
    company_data = {}
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    # Deep research for each field
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"🤔 Deep researching {field.replace('_', ' ')} for {company_name}...")
        
        # Execute deep research for this field
        field_data = deep_research_agent(company_name, field)
        company_data[field] = field_data
        
        time.sleep(2)  # Allow time for deep analysis
    
    # Deep Syntel relevance analysis
    status_text.text("🎯 Conducting deep Syntel opportunity analysis...")
    progress_bar.progress(90)
    
    relevance_opportunities, intent_score = deep_syntel_analysis(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = relevance_opportunities
    company_data["intent_scoring_level"] = intent_score
    
    progress_bar.progress(100)
    status_text.text("✅ Deep research complete!")
    
    return company_data

# --- Display Results ---
def display_research_results(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Display research results in clean format"""
    
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
            value = company_input
        else:
            value = data_dict.get(data_field, "Information not available from research")
        
        # Format opportunities with proper line breaks
        if data_field == "why_relevant_to_syntel_bullets":
            if value != "Information not available from research":
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # Make research sources clickable
            if "Research Source" in str(value):
                def make_clickable(match):
                    url = match.group(1)
                    return f'[<a href="{url}" target="_blank">Research Source</a>]'
                
                value_with_links = re.sub(r'\[Research Source: ([^\]]+)\]', make_clickable, value)
                value_with_links = re.sub(r'\[Research Sources: ([^\]]+)\]', make_clickable, value_with_links)
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{value_with_links}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit Application ---
st.set_page_config(
    page_title="Deep Company Research",
    layout="wide",
    page_icon="🔍"
)

st.title("🔍 Deep Company Intelligence Research")
st.markdown("### Comprehensive Research with Source Analysis")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter company name for deep research:", "Neuberg Diagnostics")
with col2:
    with st.form("deep_research_form"):
        submitted = st.form_submit_button("🚀 Start Deep Research", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"🔍 Conducting deep research for {company_input}..."):
        try:
            # Execute deep research
            company_data = execute_deep_research(company_input)
            
            # Display results
            st.success(f"✅ Deep research complete for {company_input}")
            
            # Show research results
            st.subheader(f"Deep Research Report for {company_input}")
            final_df = display_research_results(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Research metrics
            with st.expander("📊 Research Metrics", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "Information not available from research")
                
                meaningful_fields = sum(1 for field in REQUIRED_FIELDS[:-2] 
                                     if company_data.get(field) and 
                                     company_data.get(field) != "Information not available from research" and
                                     len(str(company_data.get(field)).split()) > 5)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Researched", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Quality Data Points", f"{meaningful_fields}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    st.metric("Opportunity Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download options
            st.subheader("💾 Download Research Report")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='DeepResearch')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(company_data, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_deep_research.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_deep_research.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_deep_research.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
            
            # Store research in history
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
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Opportunity Score: {research['data'].get('intent_scoring_level', 'Medium')}")
            completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                if research['data'].get(field) and 
                                research['data'].get(field) != "Information not available from research")
            st.write(f"Fields Researched: {completed_fields}/{len(REQUIRED_FIELDS)}")
        
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Deep Company Intelligence Research | Source-Based Analysis"
    "</div>",
    unsafe_allow_html=True
)
