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
- Ports, Stadiums, Education, Manufacturing – Factories
- Healthcare, Hospitality – Hotels & Convention Centres
- Warehouses, BFSI, IT/ITES, GCC – Mumbai, Pune, Bangalore, Hyderabad, Chennai

IDEAL CUSTOMER PROFILE:
- Employee Count: 150–500+ employees
- Revenue Size: ₹100 Cr+

PRODUCT & SERVICE OFFERING (Focus):
- Network Integration Solutions & Services
- Wi-Fi Deployments (Active + Passive Components)
- Delivery & Implementation – Unbox + Integrate with current environment
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
- 3–5X Coverage Advantage (1 Altai AP = 3–5 Cisco/Ruckus APs)
- High Concurrent User Handling
- Zero-Roaming Drop & Seamless Handover
- Superior Outdoor Performance
- Lower Total Cost of Ownership
- Excellent for: Manufacturing, Hospitality, GCC, Healthcare, Warehouses, Education
"""

# --- Enhanced Deep Research Functions ---
def deep_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Deep search with multiple queries and comprehensive coverage"""
    
    # Enhanced search queries for deeper research
    deep_search_queries = {
        "linkedin_url": [
            f'"{company_name}" LinkedIn company page official',
            f'"{company_name}" LinkedIn careers employees',
            f'"{company_name}" LinkedIn about overview'
        ],
        "company_website_url": [
            f'"{company_name}" official website',
            f'"{company_name}" corporate site',
            f'"{company_name}" contact us headquarters'
        ],
        "industry_category": [
            f'"{company_name}" industry business sector',
            f'"{company_name}" core business services',
            f'"{company_name}" market segment specialization'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employee count size',
            f'"{company_name}" team size employees',
            f'"{company_name}" workforce headcount'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters corporate office',
            f'"{company_name}" main office location',
            f'"{company_name}" registered office address'
        ],
        "revenue_source": [
            f'"{company_name}" revenue business model financial',
            f'"{company_name}" annual revenue turnover',
            f'"{company_name}" funding investment business model'
        ],
        "branch_network_count": [
            f'"{company_name}" branches locations network facilities',
            f'"{company_name}" offices centers locations',
            f'"{company_name}" expansion facilities capacity'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024 2025 growth new facilities',
            f'"{company_name}" new offices campuses expansion',
            f'"{company_name}" investment growth facilities'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT modernization',
            f'"{company_name}" technology upgrade digital initiatives',
            f'"{company_name}" AI automation digital projects'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT director technology leadership',
            f'"{company_name}" chief information officer technology head',
            f'"{company_name}" IT leadership management'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors partners IT infrastructure',
            f'"{company_name}" Cisco VMware SAP Oracle Microsoft partners',
            f'"{company_name}" network infrastructure vendors'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN network upgrade tender',
            f'"{company_name}" network infrastructure project',
            f'"{company_name}" connectivity upgrade initiative'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation smart technology',
            f'"{company_name}" robotics automation digital',
            f'"{company_name}" smart factory Industry 4.0'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption AWS Azure Google Cloud',
            f'"{company_name}" cloud migration strategy',
            f'"{company_name}" GCC global capability center'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility expansion',
            f'"{company_name}" infrastructure development campus',
            f'"{company_name}" real estate expansion'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget investment capex',
            f'"{company_name}" technology spending budget',
            f'"{company_name}" IT infrastructure investment'
        ]
    }
    
    all_results = []
    queries = deep_search_queries.get(field_name, [f'"{company_name}" {field_name}'])
    
    for query in queries[:3]:  # Try multiple queries
        try:
            time.sleep(1.5)  # Rate limiting
            results = search_tool.invoke({"query": query, "max_results": 4})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    if len(content) > 100:  # Filter out very short results
                        all_results.append({
                            "title": result.get('title', ''),
                            "content": content[:600],  # More context for deep analysis
                            "url": result.get('url', ''),
                            "field": field_name,
                            "query": query
                        })
        except Exception as e:
            continue
    
    return all_results

def deep_analyze_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Deep analysis of search results to extract comprehensive information"""
    
    if not search_results:
        return "N/A"
    
    # Build comprehensive research context
    research_context = "COMPREHENSIVE RESEARCH DATA:\n\n"
    for i, result in enumerate(search_results[:4]):  # Use top 4 results for depth
        research_context += f"RESULT {i+1} - {result['title']}:\n"
        research_context += f"CONTENT: {result['content']}\n"
        research_context += f"URL: {result['url']}\n\n"
    
    # Get unique source URLs
    unique_urls = list(set([result['url'] for result in search_results]))[:3]
    
    # Deep analysis prompts that avoid field names in output
    deep_analysis_prompts = {
        "linkedin_url": f"""
        Analyze the research data to find the primary LinkedIn company page URL for {company_name}.
        Return ONLY the complete LinkedIn URL or 'N/A' if not found.
        """,
        
        "company_website_url": f"""
        From the research data, identify the official corporate website URL for {company_name}.
        Return ONLY the main website URL or 'N/A'.
        """,
        
        "industry_category": f"""
        Based on the comprehensive research, determine the primary industry and business focus of {company_name}.
        Provide a concise 3-5 word description of their core business sector.
        Examples: "Healthcare Diagnostics", "Logistics Warehousing", "IT Services"
        """,
        
        "employee_count_linkedin": f"""
        Analyze the research to find current employee count information for {company_name}.
        Look for specific numbers, ranges, or LinkedIn employee data.
        Return just the employee count information.
        """,
        
        "headquarters_location": f"""
        From the research data, extract the headquarters or main corporate office location for {company_name}.
        Format as 'City, State' or 'City, Country'.
        """,
        
        "revenue_source": f"""
        Based on comprehensive analysis, describe the revenue model and financial scale of {company_name}.
        Include revenue numbers, business model, and key financial metrics if available.
        Be specific and quantitative.
        """,
        
        "branch_network_count": f"""
        Analyze the research to determine the scale of operations for {company_name}.
        Look for numbers of facilities, branches, locations, capacity metrics, or geographical spread.
        Provide specific counts and descriptions of their operational footprint.
        """,
        
        "expansion_news_12mo": f"""
        From the research, identify recent expansion activities, new facilities, or growth initiatives for {company_name}.
        Focus on the last 12-24 months. Include specific locations, investments, and timelines.
        Provide detailed expansion information.
        """,
        
        "digital_transformation_initiatives": f"""
        Analyze the research to identify digital transformation and technology modernization efforts at {company_name}.
        Look for mentions of AI, automation, digital platforms, ERP, cloud, or IT modernization projects.
        Describe their digital initiatives specifically.
        """,
        
        "it_leadership_change": f"""
        From the research, extract information about IT leadership and technology executives at {company_name}.
        Look for CIO, CTO, IT Director names, appointments, or organizational changes.
        Provide specific names and positions if available.
        """,
        
        "existing_network_vendors": f"""
        Analyze the research to identify technology vendors and partners used by {company_name}.
        Look for mentions of Cisco, VMware, SAP, Oracle, Microsoft, or other IT infrastructure providers.
        List specific vendors and technologies mentioned.
        """,
        
        "wifi_lan_tender_found": f"""
        From the research, identify any network infrastructure projects, WiFi upgrades, or technology tenders for {company_name}.
        Look for network modernization, connectivity upgrades, or IT infrastructure projects.
        Describe any network-related initiatives.
        """,
        
        "iot_automation_edge_integration": f"""
        Analyze the research to determine IoT, automation, and edge computing adoption at {company_name}.
        Look for smart technology, robotics, automation, IoT sensors, or digital automation initiatives.
        Describe their automation and IoT capabilities specifically.
        """,
        
        "cloud_adoption_gcc_setup": f"""
        From the research, identify cloud adoption strategy and global capability centers for {company_name}.
        Look for AWS, Azure, Google Cloud adoption, cloud migration, or GCC setup.
        Describe their cloud strategy and capabilities.
        """,
        
        "physical_infrastructure_signals": f"""
        Analyze the research to identify physical infrastructure developments and facility expansions for {company_name}.
        Look for new construction, facility upgrades, campus expansions, or real estate developments.
        Describe their physical infrastructure growth.
        """,
        
        "it_infra_budget_capex": f"""
        From the research, determine IT infrastructure budgeting and capital expenditure for {company_name}.
        Look for technology investments, IT spending, capex announcements, or digital infrastructure budgets.
        Provide specific budget information if available.
        """
    }
    
    prompt = f"""
    TASK: {deep_analysis_prompts.get(field_name, f"Analyze comprehensive research about {company_name}")}
    
    {research_context}
    
    CRITICAL INSTRUCTIONS:
    - Analyze ALL provided research data thoroughly
    - Extract specific, quantitative information where available
    - Avoid generic statements - be specific and evidence-based
    - If information is not found after thorough analysis, return 'N/A'
    - Do NOT include field names or labels in your response
    - Provide only the extracted factual information
    
    EXTRACTED INFORMATION:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a deep research analyst. Thoroughly analyze all research data and extract specific, evidence-based information. Never include field names or labels in responses."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Advanced cleaning and validation
        response = re.sub(r'^(.*?):\s*', '', response)  # Remove any label prefixes
        response = re.sub(r'\b(field|information|data|result):\s*', '', response, flags=re.IGNORECASE)
        
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', 'not available', ''] or 
            len(response) < 3):
            return "N/A"
        
        # Limit response length but keep it comprehensive
        if len(response) > 300:
            response = response[:297] + "..."
        
        # Add source URLs for credibility
        if unique_urls and response != "N/A":
            if len(unique_urls) == 1:
                response += f" [Source: {unique_urls[0]}]"
            else:
                response += f" [Sources: {', '.join(unique_urls[:2])}]"
            
        return response
            
    except Exception as e:
        return "N/A"

def generate_gtm_relevance_analysis(company_data: Dict, company_name: str, all_search_results: List[Dict]) -> tuple:
    """Generate relevance analysis based on Syntel GTM intelligence"""
    
    # Build comprehensive company profile
    company_profile = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            # Remove field names from display
            clean_value = re.sub(r'.*:\s*', '', value) if ':' in value else value
            company_profile.append(clean_value)
    
    context = "\n".join(company_profile)
    
    # Get credible sources
    unique_urls = list(set([result['url'] for result in all_search_results]))[:3]
    source_context = f"Research sources: {', '.join(unique_urls)}" if unique_urls else "Based on comprehensive research"
    
    relevance_prompt = f"""
    COMPANY PROFILE: {company_name}
    {source_context}
    
    COMPANY DATA:
    {context}
    
    SYNTEL GTM INTELLIGENCE:
    {SYNTEL_GTM_INTELLIGENCE}
    
    ANALYSIS TASK:
    1. Match company data against Syntel's Ideal Customer Profile (150+ employees, ₹100Cr+ revenue)
    2. Check alignment with Syntel's Target Industries
    3. Identify specific Network Integration opportunities
    4. Assess Wi-Fi/Network upgrade potential based on expansion, facilities, digital initiatives
    5. Evaluate Altai Wi-Fi suitability based on company's space type and needs
    
    FORMAT REQUIREMENTS:
    BULLETS:
    1) [Specific infrastructure signal] - [Syntel solution match with quantitative benefit]
    2) [Technology gap/opportunity] - [Altai Wi-Fi or Network Integration advantage]
    3) [Business expansion alignment] - [Service implementation opportunity]
    SCORE: High/Medium/Low
    
    Focus on concrete, evidence-based opportunities from the company data.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic GTM analyst. Match company data against Syntel's target profile and identify specific, evidence-based opportunities for network integration and Wi-Fi solutions."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse response with improved bullet detection
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
            
            if in_bullets and line and (line.startswith(('1)', '2)', '3)', '•', '-')) or any(num in line for num in ['1)', '2)', '3)'])):
                # Clean and format bullet
                clean_line = re.sub(r'^[1-3\)•\-]\s*', '', line)
                if len(clean_line) > 20:  # Ensure meaningful content
                    bullets.append(f"• {clean_line}")
        
        # Ensure we have 3 quality bullets
        while len(bullets) < 3:
            additional_bullets = [
                f"• {company_name}'s operational scale presents network infrastructure modernization opportunities",
                f"• Digital transformation initiatives align with Syntel's integration expertise",
                f"• Facility expansion signals potential for Altai Wi-Fi deployment"
            ]
            bullets.append(additional_bullets[len(bullets)])
        
        formatted_bullets = "\n".join(bullets[:3])
        return formatted_bullets, score
        
    except Exception as e:
        # Fallback with GTM context
        fallback_bullets = f"""• {company_name} matches Syntel's target profile for network integration services
• Expansion activities indicate potential Wi-Fi infrastructure needs
• Digital initiatives align with Syntel's technology modernization expertise"""
        return fallback_bullets, "Medium"

# --- Main Deep Research Function ---
def deep_research_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Main function to perform deep research on all fields"""
    
    company_data = {}
    all_search_results = []
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    # Deep research for each field
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.info(f"🔍 Deep researching {field.replace('_', ' ').title()} for {company_name}...")
        
        # Perform deep search and analysis
        search_results = deep_search_for_field(company_name, field)
        all_search_results.extend(search_results)
        
        field_data = deep_analyze_field_with_sources(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1.5)  # Respect rate limits
    
    # Generate GTM-based relevance analysis
    status_text.info("🎯 Analyzing Syntel GTM relevance...")
    progress_bar.progress(90)
    
    relevance_bullets, intent_score = generate_gtm_relevance_analysis(
        company_data, company_name, all_search_results
    )
    company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
    company_data["intent_scoring_level"] = intent_score
    
    progress_bar.progress(100)
    status_text.success("✅ Deep research complete!")
    
    return company_data

# --- Enhanced Display Function ---
def format_deep_research_display(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean display format without field names"""
    
    mapping = {
        "Company Name": "company_name",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url", 
        "Industry Category": "industry_category",
        "Employee Count": "employee_count_linkedin",
        "Headquarters Location": "headquarters_location",
        "Revenue & Business Model": "revenue_source",
        "Branch Network & Facilities": "branch_network_count",
        "Recent Expansion News": "expansion_news_12mo",
        "Digital Transformation": "digital_transformation_initiatives",
        "IT Leadership": "it_leadership_change",
        "Technology Vendors": "existing_network_vendors",
        "Network Upgrade Signals": "wifi_lan_tender_found",
        "IoT & Automation": "iot_automation_edge_integration",
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
        
        # Clean field names from values
        if isinstance(value, str):
            value = re.sub(r'^.*?:\s*', '', value)  # Remove any prefix labels
        
        # Format bullet points for relevance analysis
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str):
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # For fields with URLs, make them clickable
            if isinstance(value, str) and "http" in value:
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
    page_title="Syntel Deep Research Agent",
    layout="wide",
    page_icon="🔍"
)

st.title("🔍 Syntel Deep Company Intelligence Agent")
st.markdown("### 🎯 GTM-Aligned Research with Network Integration Focus")

# Display enhanced approach
with st.expander("🚀 Deep Research Methodology", expanded=True):
    st.markdown("""
    **Enhanced Deep Research Features:**
    
    - **🔍 Multi-Query Depth**: 3+ search queries per field for comprehensive coverage
    - **📊 Evidence-Based Analysis**: LLM deeply analyzes all search results before extraction
    - **🎯 GTM Intelligence**: Relevance analysis based on Syntel's target profile and Altai Wi-Fi advantages
    - **🚫 No Field Names**: Clean output without repetitive field labels
    - **💡 Strategic Insights**: Focus on network integration and Wi-Fi deployment opportunities
    
    **GTM Alignment:**
    - Targets: 150+ employees, ₹100Cr+ revenue companies
    - Industries: Manufacturing, Healthcare, Warehouses, GCC, Education, Hospitality
    - Solutions: Network Integration, Altai Wi-Fi, Multi-vendor implementation
    """)

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name for deep research:", "Neuberg Diagnostics")
with col2:
    with st.form("deep_research_form"):
        submitted = st.form_submit_button("🚀 Start Deep Research", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"**🔍 Conducting deep GTM-aligned research for {company_input}...**"):
        try:
            # Perform deep research
            company_data = deep_research_company_intelligence(company_input)
            
            # Display results
            st.balloons()
            st.success(f"✅ Deep research complete for {company_input}!")
            
            # Display final results
            st.subheader(f"GTM Intelligence Report for {company_input}")
            final_df = format_deep_research_display(company_input, company_data)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show GTM alignment metrics
            with st.expander("🎯 GTM Alignment Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "N/A")
                
                # Check Ideal Customer Profile alignment
                employee_data = company_data.get("employee_count_linkedin", "")
                revenue_data = company_data.get("revenue_source", "")
                
                icp_alignment = "Partial"
                if "150" in employee_data or "500" in employee_data or "1000" in employee_data:
                    if "100" in revenue_data or "Cr" in revenue_data or "crore" in revenue_data.lower():
                        icp_alignment = "High"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Researched", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("ICP Alignment", icp_alignment)
                with col3:
                    st.metric("Intent Score", company_data.get("intent_scoring_level", "Medium"))
            
            # Download options
            st.subheader("💾 Download Deep Research Report")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='GTM_Intelligence')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(company_data, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_gtm_intelligence.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_gtm_intelligence.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_gtm_intelligence.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.error(f"Deep research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits. Please try again in a few moments.")

# Initialize session state for history
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Store current research in history
if submitted and 'company_data' in locals():
    research_entry = {
        "company": company_input,
        "timestamp": datetime.now().isoformat(),
        "data": company_data
    }
    st.session_state.research_history.append(research_entry)

# Research History Sidebar
if st.session_state.research_history:
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
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
    "Syntel Deep Research Agent | GTM-Aligned Network Intelligence"
    "</div>",
    unsafe_allow_html=True
)
