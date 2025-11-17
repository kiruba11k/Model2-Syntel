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
            f'"{company_name}" business sector',
            f'"{company_name}" type of company'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employee count',
            f'"{company_name}" number of employees',
            f'"{company_name}" workforce size'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters location',
            f'"{company_name}" corporate office',
            f'"{company_name}" main office address'
        ],
        "revenue_source": [
            f'"{company_name}" revenue business model',
            f'"{company_name}" annual revenue',
            f'"{company_name}" financial performance'
        ],
        "branch_network_count": [
            f'"{company_name}" branches locations network',
            f'"{company_name}" facilities offices',
            f'"{company_name}" laboratories centers'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024 2025',
            f'"{company_name}" new facilities growth',
            f'"{company_name}" recent expansion news'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT',
            f'"{company_name}" technology modernization',
            f'"{company_name}" digital initiatives projects'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT leadership',
            f'"{company_name}" technology leadership',
            f'"{company_name}" IT management executives'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors',
            f'"{company_name}" IT infrastructure vendors',
            f'"{company_name}" software hardware partners'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN network upgrade',
            f'"{company_name}" network infrastructure',
            f'"{company_name}" connectivity upgrade'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation',
            f'"{company_name}" smart technology',
            f'"{company_name}" robotics automation'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption',
            f'"{company_name}" cloud migration',
            f'"{company_name}" AWS Azure Google Cloud'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility',
            f'"{company_name}" infrastructure development',
            f'"{company_name}" building expansion'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget investment',
            f'"{company_name}" technology spending',
            f'"{company_name}" IT infrastructure budget'
        ]
    }
    
    all_results = []
    queries = search_queries.get(field_name, [f'"{company_name}" {field_name}'])
    
    for query in queries[:3]:  # Use 3 queries for better coverage
        try:
            time.sleep(1.5)
            results = search_tool.invoke({"query": query, "max_results": 4})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    # Filter out results that don't match the company name well
                    company_match_score = sum([
                        2 for word in company_name.split() 
                        if word.lower() in content.lower() or word.lower() in title.lower()
                    ])
                    
                    # Only include results with good company matching and substantial content
                    if company_match_score >= 2 and len(content) > 100:
                        all_results.append({
                            "title": title,
                            "content": content[:500],  # More content for better analysis
                            "url": result.get('url', ''),
                            "field": field_name
                        })
        except Exception as e:
            continue
    
    return all_results

# --- Improved Field-Specific Extraction Prompts ---
FIELD_EXTRACTION_PROMPTS = {
    "linkedin_url": """
    Extract the primary LinkedIn company page URL. 
    Look for linkedin.com/company/ patterns.
    Return ONLY the complete LinkedIn URL or N/A if not found.
    """,
    
    "company_website_url": """
    Extract the official company website URL. 
    Look for the main corporate website, not subpages.
    Return ONLY the main website URL or N/A.
    """,
    
    "industry_category": """
    Extract the primary industry and business category. 
    Be specific about their core business.
    Examples: "Healthcare Diagnostics", "Logistics and Supply Chain", "IT Services"
    Return 2-4 words describing their main industry.
    """,
    
    "employee_count_linkedin": """
    Extract employee count information. 
    Look for specific numbers, ranges, or LinkedIn employee data.
    Return the employee count with context like "500-1000 employees" or specific number.
    """,
    
    "headquarters_location": """
    Extract the headquarters or main corporate office location.
    Format as "City, State" or "City, Country".
    Be specific about the location.
    """,
    
    "revenue_source": """
    Extract revenue information and business model details.
    Look for specific revenue numbers, growth rates, and business model description.
    Be quantitative and specific.
    """,
    
    "branch_network_count": """
    Extract detailed information about branch network, facilities, and operational footprint.
    Look for specific counts of locations, laboratories, offices, or facilities.
    Provide comprehensive details about their geographical presence.
    """,
    
    "expansion_news_12mo": """
    Extract recent expansion activities, new facilities, or growth initiatives from the last 12-24 months.
    Include specific locations, investments, timelines, and strategic moves.
    Be detailed about their expansion strategy.
    """,
    
    "digital_transformation_initiatives": """
    Extract digital transformation projects and technology modernization efforts.
    Look for specific technologies like ERP, AI, automation, cloud platforms, digital platforms.
    Describe their digital initiatives with specific technologies mentioned.
    """,
    
    "it_leadership_change": """
    Extract IT leadership and technology executive information.
    Focus on CIO, CTO, IT Director roles, appointments, and organizational changes.
    Provide specific names, positions, and any leadership changes.
    """,
    
    "existing_network_vendors": """
    Extract technology vendors, partners, and infrastructure providers used by the company.
    Look for specific vendors like Cisco, VMware, SAP, Oracle, Microsoft, AWS, Azure.
    List the specific technology vendors and platforms mentioned.
    """,
    
    "wifi_lan_tender_found": """
    Extract information about network infrastructure projects, WiFi upgrades, or technology tenders.
    Look for network modernization, connectivity upgrades, or IT infrastructure projects.
    Describe any network-related initiatives or upgrades.
    """,
    
    "iot_automation_edge_integration": """
    Extract IoT, automation, robotics, and edge computing implementations.
    Look for smart technology, automation systems, robotics, IoT sensors.
    Describe their automation and IoT capabilities with specific technologies.
    """,
    
    "cloud_adoption_gcc_setup": """
    Extract cloud adoption strategy and global capability centers setup.
    Look for AWS, Azure, Google Cloud adoption, cloud migration, GCC establishment.
    Describe their cloud strategy and global operations.
    """,
    
    "physical_infrastructure_signals": """
    Extract physical infrastructure developments and facility expansions.
    Look for new construction, facility upgrades, campus expansions, real estate developments.
    Describe their physical infrastructure growth and expansions.
    """,
    
    "it_infra_budget_capex": """
    Extract IT infrastructure budgeting and capital expenditure information.
    Look for technology investments, IT spending, capex announcements, digital infrastructure budgets.
    Provide specific budget information and investment details.
    """
}

def deep_field_extraction(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Deep field extraction with comprehensive analysis of search results"""
    
    if not search_results:
        return "N/A"
    
    # Build comprehensive research context
    research_context = "DETAILED RESEARCH DATA FOR ANALYSIS:\n\n"
    for i, result in enumerate(search_results[:4]):  # Use top 4 results for depth
        research_context += f"RESEARCH SOURCE {i+1}:\n"
        research_context += f"Title: {result['title']}\n"
        research_context += f"Content: {result['content']}\n"
        research_context += f"URL: {result['url']}\n\n"
    
    # Get unique source URLs
    unique_urls = list(set([result['url'] for result in search_results]))[:2]
    
    prompt = f"""
    COMPANY: {company_name}
    INFORMATION TO EXTRACT: {field_name.replace('_', ' ').title()}
    
    EXTRACTION GUIDELINES:
    {FIELD_EXTRACTION_PROMPTS.get(field_name, f"Extract comprehensive {field_name} information")}
    
    RESEARCH DATA:
    {research_context}
    
    CRITICAL INSTRUCTIONS:
    1. Analyze ALL research data thoroughly and extract the most relevant information
    2. Provide comprehensive, detailed information - not just URLs
    3. Be specific, factual, and quantitative where possible
    4. Focus only on information that clearly relates to {company_name}
    5. If the research data doesn't contain clear information about this field, return "N/A"
    6. Do not include field names or labels in your response
    7. Provide meaningful content, not just source references
    
    EXTRACTED INFORMATION:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a deep research analyst. Thoroughly analyze all research data and extract comprehensive, meaningful information. Provide detailed content, not just URLs or vague statements."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Comprehensive cleaning and validation
        response = re.sub(r'^(.*?):\s*', '', response)  # Remove any label prefixes
        response = re.sub(r'\b(field|information|data|result|extracted|analysis):\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # Enhanced validation
        invalid_indicators = [
            'n/a', 'not found', 'no information', 'not available', 
            'unavailable', 'unknown', 'not specified', 'none'
        ]
        
        if (not response or 
            any(indicator in response.lower() for indicator in invalid_indicators) or
            len(response) < 10 or  # Minimum meaningful content length
            response.lower().startswith('http') or  # Should not be just URLs
            'source:' in response.lower() and len(response.split()) < 10):  # Should not be just sources
            return "N/A"
        
        # Check for wrong company data
        wrong_company_indicators = [
            'hcl', 'tcs', 'infosys', 'wipro', 'tech mahindra', 'cognizant',
            'unrelated', 'different company', 'another company', 'irrelevant'
        ]
        if any(indicator in response.lower() for indicator in wrong_company_indicators):
            return "N/A"
        
        # Ensure meaningful content (not just URLs or sources)
        words = response.split()
        if len(words) < 5 and any('http' in word for word in words):
            return "N/A"
        
        # Limit length but keep comprehensive content
        if len(response) > 350:
            response = response[:347] + "..."
        
        # Add source URLs at the end for credibility
        if unique_urls and response != "N/A":
            source_text = f" [Sources: {', '.join(unique_urls)}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
            
        return response
        
    except Exception as e:
        return "N/A"

def generate_comprehensive_relevance_analysis(company_data: Dict, company_name: str) -> tuple:
    """Generate comprehensive relevance analysis based on actual company data"""
    
    # Build detailed company profile from available data
    profile_parts = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            # Clean value and remove source tags for analysis
            clean_value = re.sub(r'\s*\[Source[^\]]*\]', '', value)
            if len(clean_value.strip()) > 10:  # Only include meaningful data
                profile_parts.append(f"{field.replace('_', ' ').title()}: {clean_value}")
    
    company_profile = "\n".join(profile_parts)
    
    relevance_prompt = f"""
    COMPANY: {company_name}
    
    DETAILED COMPANY PROFILE:
    {company_profile}
    
    SYNTEL TARGET PROFILE:
    - Target Industries: Healthcare, Manufacturing, Warehouses, IT/ITES, GCC, Education, Hospitality
    - Employee Range: 150-500+ employees  
    - Revenue: 100 Cr+
    - Solutions: Network Integration, Wi-Fi Deployment, Altai Wi-Fi, Multi-vendor Implementation
    - Key Signals: Expansion, Digital Transformation, Network Upgrades, Large Facilities
    
    ANALYSIS TASK:
    Based on the detailed company profile above, identify specific, evidence-based opportunities for Syntel's solutions.
    
    REQUIREMENTS:
    1. Analyze each piece of company data and match it with Syntel's target profile
    2. Identify concrete opportunities for network integration and Wi-Fi solutions
    3. Focus on expansion activities, digital initiatives, and infrastructure needs
    4. Be specific and reference actual company data points
    5. Format as 3 detailed bullet points
    6. Provide intent score: High/Medium/Low based on alignment with Syntel's target criteria
    
    FORMAT:
    BULLETS:
    - [Specific opportunity with concrete evidence from company data and how it matches Syntel solutions]
    - [Technology or infrastructure need identified from company data and relevant Syntel capability]
    - [Business expansion or modernization signal and corresponding Syntel service alignment]
    SCORE: High/Medium/Low
    
    IMPORTANT: Be specific and evidence-based, not generic.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic business analyst. Analyze company data thoroughly and identify specific, evidence-based opportunities for IT infrastructure and network solutions. Be detailed and concrete in your analysis."),
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
            
            if in_bullets and line and (line.startswith('-') or line.startswith('•')):
                clean_line = re.sub(r'^[-•]\s*', '', line)
                if len(clean_line) > 30:  # Ensure meaningful content
                    bullets.append(f"- {clean_line}")
        
        # Ensure we have 3 quality, specific bullets
        if len(bullets) < 3:
            # Create fallback bullets based on actual company data
            fallback_bullets = []
            if any('expansion' in str(company_data.get(field, '')).lower() for field in ['expansion_news_12mo', 'branch_network_count']):
                fallback_bullets.append(f"- {company_name}'s expansion activities indicate need for scalable network infrastructure and Wi-Fi deployment across new facilities")
            if any('digital' in str(company_data.get(field, '')).lower() for field in ['digital_transformation_initiatives', 'iot_automation_edge_integration']):
                fallback_bullets.append(f"- Digital transformation initiatives present opportunities for network integration and modernization with Syntel's multi-vendor implementation expertise")
            if any('facilit' in str(company_data.get(field, '')).lower() for field in ['physical_infrastructure_signals', 'branch_network_count']):
                fallback_bullets.append(f"- Physical infrastructure growth signals potential for Altai Wi-Fi solutions to provide superior coverage for large spaces and facilities")
            
            while len(bullets) < 3 and fallback_bullets:
                bullets.append(fallback_bullets.pop(0))
            
            # Fill remaining with generic but relevant bullets
            while len(bullets) < 3:
                bullets.append(f"- Additional network integration opportunity identified based on {company_name}'s operational scale and technology needs")
        
        formatted_bullets = "\n".join(bullets[:3])
        return formatted_bullets, score
        
    except Exception as e:
        # Comprehensive fallback based on common patterns
        fallback_bullets = [
            f"- {company_name}'s operational scale and expansion activities indicate potential for network infrastructure solutions",
            f"- Digital transformation initiatives align with Syntel's network integration and modernization expertise", 
            f"- Facility expansion and technology upgrades present opportunities for Wi-Fi deployment and network optimization"
        ]
        return "\n".join(fallback_bullets), "Medium"

# --- Main Research Function ---
def comprehensive_company_research(company_name: str) -> Dict[str, Any]:
    """Main function for comprehensive company research"""
    
    company_data = {}
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_fields = len(REQUIRED_FIELDS) - 2
    
    # Research each field with deep analysis
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.text(f"Researching {field.replace('_', ' ').title()} for {company_name}...")
        
        # Enhanced search and deep extraction
        search_results = enhanced_search_for_field(company_name, field)
        field_data = deep_field_extraction(company_name, field, search_results)
        company_data[field] = field_data
        
        time.sleep(1.5)  # Rate limiting
    
    # Generate comprehensive relevance analysis
    status_text.text("Analyzing Syntel relevance...")
    progress_bar.progress(90)
    
    relevance_bullets, intent_score = generate_comprehensive_relevance_analysis(company_data, company_name)
    company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
    company_data["intent_scoring_level"] = intent_score
    
    progress_bar.progress(100)
    status_text.text("Research complete!")
    
    return company_data

# --- Clean Display Function ---
def create_research_dataframe(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Create clean dataframe with meaningful content"""
    
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
        
        # Clean the value - ensure it's meaningful content
        if isinstance(value, str) and value != "N/A":
            # Remove any residual field names or labels
            value = re.sub(r'^.*?:\s*', '', value)
            # Ensure it's not just URLs
            if value.startswith('http') and len(value.split()) <= 2:
                value = "N/A"
        
        # Format for display
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str) and value != "N/A":
                html_value = value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # Make URLs clickable but ensure there's content
            if isinstance(value, str) and "http" in value and len(value) > 50:
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
                
                # Count fields with meaningful content (not just URLs)
                meaningful_fields = sum(1 for field in REQUIRED_FIELDS[:-2] 
                                     if company_data.get(field) and 
                                     company_data.get(field) != "N/A" and
                                     len(str(company_data.get(field)).split()) > 3)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Meaningful Data Fields", f"{meaningful_fields}/{len(REQUIRED_FIELDS)-2}")
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
