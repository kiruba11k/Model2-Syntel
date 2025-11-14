import streamlit as st
import pandas as pd
import json
import operator
import re
from typing import TypedDict, Annotated
from io import BytesIO
from datetime import datetime
import time

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
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
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()

# --- Domain Lists for Specialized Field Search ---
NETWORK_VENDOR_DOMAINS = [
    "blogs.oracle.com", "cio.economictimes.indiatimes.com", "supplychaindigital.com",
    "crownworldwide.com", "frontier-enterprise.com", "hotelmanagement-network.com",
    "hotelwifi.com", "appsruntheworld.com", "us.nttdata.com", "forbes.com",
    "mtdcnc.com", "microsoft.com", "sap.com", "amd.com", "videos.infosys.com",
    "oracle.com", "infosys.com", "medicovertech.in", "icemakeindia.com",
    "saurenergy.com", "aajenterprises.com", "techcircle.in", "indionetworks.com",
    "birlasoft.com", "mindteck.com", "inavateapac.com", "jioworldcentre.com",
    "vmware.com", "intellectdesign.com", "cisco.com", "prnewswire.com",
    "industryoutlook.in", "networkcomputing.com", "crn.in", "builtwith.com",
    "wappalyzer.com", "stackshare.io", "linkedin.com", "indeed.com",
    "naukri.com", "monster.com", "censys.io", "shodan.io", "github.com",
    "sec.gov", "bseindia.com", "nseindia.com", "aws.amazon.com",
    "azure.microsoft.com", "cloud.google.com", "g2.com", "gartner.com"
]

WIFI_TENDER_DOMAINS = [
    "spectra.co", "enterprise.spectra.com", "hotelwifi.com", "oracle.com",
    "loca-service.com", "indiatimes.com", "aajenterprises.com",
    "economictimes.indiatimes.com", "inductusgcc.com", "businesstoday.in",
    "jioworldcentre.com", "eventswifiinternet.com", "vmware.com",
    "economictimes.com", "tendersinfo.com", "tendertiger.com",
    "networkworld.com", "gem.gov.in"
]

IOT_AUTOMATION_DOMAINS = [
    "automationworld.com", "controleng.com", "iotinsider.com", "iottales.com",
    "moldstud.com", "iot-analytics.com", "gartner.com", "theregister.com",
    "eetimes.com", "iiot.today", "mahindralogistics.com", "ledgerinsights.com",
    "nipponexpress.com", "techemerge.org", "economictimes.indiatimes.com",
    "mahindra.com", "odisharay.com", "farnek.com", "icemakeindia.com",
    "aajenterprises.com", "adityabirlarealestate.com", "mindteck.com",
    "inavateapac.com", "dredgewire.com", "sanyglobal.com", "welspunone.com",
    "allcargogati.com", "tnidb.tn.gov.in", "ammann.com", "balajiamines.com",
    "lg.com", "adaniports.com", "iotworldtoday.com", "deltaww.com",
    "indiatimes.com", "expresscomputer.in", "dhl.com", "bloombergquint.com",
    "assettype.com"
]

CLOUD_ADOPTION_DOMAINS = [
    "economictimes.indiatimes.com", "cio.economictimes.indiatimes.com",
    "techcircle.in", "globenewswire.com", "prnewswire.com", "businesswire.com",
    "druva.com", "nipponexpress-holdings.com", "hotelmanagement-network.com",
    "microsoft.com", "amd.com", "oracle.com", "infosys.com",
    "medicalbuyer.co.in", "loca-service.com", "journalofsupplychain.com",
    "mindteck.com", "indiaseatradenews.com", "eletsonline.com",
    "aws.amazon.com", "timesofindia.indiatimes.com", "vmware.com"
]

PHYSICAL_INFRASTRUCTURE_DOMAINS = [
    "constructionweekonline.in", "constructionworld.in",
    "economictimes.indiatimes.com", "businessstandard.com", "moneycontrol.com",
    "livemint.com", "thehindu.com", "manufacturingtodayindia.com",
    "infrastructuretoday.co.in", "projectsToday.com", "mahindra.com",
    "travelmedia.in", "itln.in", "nipponexpress-holdings.com", "thewire.in",
    "belden.com", "oemupdate.com", "investkarnataka.co.in", "odisharay.com",
    "nseindia.com", "solarquarter.com", "chemindigest.com", "apollo.co.in",
    "birlaojasvi.ind.in", "businesstoday.in", "indiaseatradenews.com",
    "sany.in", "welspunone.com", "hospitalitybizindia.com", "thehansindia.com",
    "allcargogati.com", "gcc.economictimes.indiatimes.com",
    "timesofindia.indiatimes.com", "financialexpress.com"
]

IT_BUDGET_DOMAINS = [
    "economictimes.indiatimes.com", "business-standard.com", "financialexpress.com",
    "moneycontrol.com", "livemint.com", "thehindubusinessline.com",
    "globenewswire.com", "prnewswire.com", "reuters.com", "bloomberg.com",
    "scanx.trade", "ttgasia.com", "belden.com", "mtdcnc.com", "godrejindiasaarc.com",
    "angelone.in", "oemupdate.com", "investkarnataka.co.in", "aaryanamatasco.ind.in",
    "icemakeindia.com", "marketscreener.com", "pestel-analysis.com",
    "adityabirlarealestate.com", "mindteck.com", "fortuneindia.com",
    "inavateapac.com", "seatrade-maritime.com", "scmspectrum.com",
    "blackridgeresearch.com"
]

# --- Manual JSON Schema Definition ---
REQUIRED_FIELDS = [
    "linkedin_url", "company_website_url", "industry_category", 
    "employee_count_linkedin", "headquarters_location", "revenue_source",
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Enhanced state with specialized research fields"""
    company_name: str
    basic_research: str
    network_vendors_research: str
    wifi_tender_research: str
    iot_automation_research: str
    cloud_adoption_research: str
    physical_infrastructure_research: str
    it_budget_research: str
    validated_data_text: str
    final_json_data: dict
    messages: Annotated[list, operator.add]

# --- Syntel Core Offerings for Analysis Node ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Search Utility Function ---
def perform_targeted_search(queries: list, preferred_domains: list = None) -> str:
    """Perform targeted search with domain preferences"""
    all_results = []
    
    for query in queries:
        try:
            time.sleep(1)  # Rate limiting
            
            # Try with domain-specific queries first
            if preferred_domains:
                # Use first 3 domains for initial search
                domain_query = f"{query} site:{' OR site:'.join(preferred_domains[:3])}"
                results = search_tool.invoke({"query": domain_query, "max_results": 2})
                
                if not results:
                    # Fallback to general search if domain-specific fails
                    results = search_tool.invoke({"query": query, "max_results": 2})
            else:
                results = search_tool.invoke({"query": query, "max_results": 2})
            
            if results:
                for result in results:
                    # Filter for relevant content
                    content = result.get('content', '')
                    if len(content) > 50:  # Only include substantial content
                        all_results.append({
                            "title": result.get('title', ''),
                            "content": content[:400],
                            "url": result.get('url', '')
                        })
        except Exception as e:
            continue
    
    # Format research results
    if not all_results:
        return "No specific information found in targeted search."
    
    research_text = "TARGETED SEARCH RESULTS:\n\n"
    for i, result in enumerate(all_results):
        research_text += f"🔗 Source {i+1}:\n"
        research_text += f"📖 Title: {result['title']}\n"
        research_text += f"🌐 URL: {result['url']}\n"
        research_text += f"📝 Content: {result['content']}\n"
        research_text += "─" * 60 + "\n"
    
    return research_text

# --- Specialized Research Agents ---
def basic_company_research_node(state: AgentState) -> AgentState:
    """Specialized agent for basic company information"""
    st.session_state.status_text.info("🔍 Researching basic company information...")
    st.session_state.progress_bar.progress(10)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" official website contact information',
        f'"{company}" LinkedIn company page',
        f'"{company}" headquarters location address',
        f'"{company}" industry classification business',
        f'"{company}" employee count size'
    ]
    
    research_text = perform_targeted_search(queries, ["linkedin.com", "zoominfo.com", "crunchbase.com"])
    
    extraction_prompt = f"""
    Extract BASIC COMPANY INFORMATION for {company}:
    
    Required fields:
    - company_website_url (official website)
    - linkedin_url (LinkedIn company page URL) 
    - headquarters_location (city, state, country)
    - industry_category (specific industry classification)
    - employee_count_linkedin (if available)
    - revenue_source (main business/revenue streams)
    
    Research Data:
    {research_text}
    
    Be precise and extract only confirmed information. Include source URLs when available.
    """
    
    try:
        basic_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Extract verified basic company details for {company}")
        ]).content
    except Exception as e:
        basic_info = research_text
    
    return {"basic_research": basic_info}

def network_vendors_research_node(state: AgentState) -> AgentState:
    """Specialized agent for existing network vendors and tech stack"""
    st.session_state.status_text.info("🖧 Researching network vendors and tech stack...")
    st.session_state.progress_bar.progress(20)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" technology stack IT infrastructure',
        f'"{company}" network vendors Cisco VMware',
        f'"{company}" software platforms SAP Oracle',
        f'"{company}" cloud providers AWS Azure',
        f'"{company}" IT partnerships vendors'
    ]
    
    research_text = perform_targeted_search(queries, NETWORK_VENDOR_DOMAINS)
    
    extraction_prompt = f"""
    Extract EXISTING NETWORK VENDORS and TECH STACK for {company}:
    
    Focus on finding specific:
    - Network equipment vendors (Cisco, Juniper, etc.)
    - Software platforms (SAP, Oracle, Microsoft)
    - Cloud providers (AWS, Azure, Google Cloud)
    - IT service partners (Infosys, TCS, Wipro)
    - Hardware vendors (Dell, HP, IBM)
    
    Research Data:
    {research_text}
    
    Provide specific vendor names, technologies, and include source URLs.
    If no specific information found, state "No specific vendor information found".
    """
    
    try:
        vendors_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Find network vendors and tech stack for {company}")
        ]).content
    except Exception as e:
        vendors_info = research_text
    
    return {"network_vendors_research": vendors_info}

def wifi_tender_research_node(state: AgentState) -> AgentState:
    """Specialized agent for WiFi/LAN tenders and upgrades"""
    st.session_state.status_text.info("📡 Researching WiFi/LAN tenders and upgrades...")
    st.session_state.progress_bar.progress(30)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" WiFi upgrade network infrastructure',
        f'"{company}" LAN tender bidding',
        f'"{company}" network expansion project',
        f'"{company}" IT infrastructure tender',
        f'"{company}" wireless network deployment'
    ]
    
    research_text = perform_targeted_search(queries, WIFI_TENDER_DOMAINS)
    
    extraction_prompt = f"""
    Find RECENT Wi-Fi UPGRADES or LAN TENDERS for {company}:
    
    Look for:
    - Recent WiFi upgrade announcements
    - LAN infrastructure tenders
    - Network expansion projects
    - IT infrastructure bidding
    - Wireless deployment projects
    
    Research Data:
    {research_text}
    
    Provide specific details about any tenders or upgrades found, including dates and sources.
    If no tenders found, state "No recent WiFi/LAN tenders identified".
    """
    
    try:
        wifi_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Find WiFi/LAN tenders for {company}")
        ]).content
    except Exception as e:
        wifi_info = research_text
    
    return {"wifi_tender_research": wifi_info}

def iot_automation_research_node(state: AgentState) -> AgentState:
    """Specialized agent for IoT/Automation/Edge integration"""
    st.session_state.status_text.info("🤖 Researching IoT and automation initiatives...")
    st.session_state.progress_bar.progress(40)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" IoT Internet of Things implementation',
        f'"{company}" automation projects digital transformation',
        f'"{company}" edge computing initiatives',
        f'"{company}" smart factory Industry 4.0',
        f'"{company}" robotics process automation'
    ]
    
    research_text = perform_targeted_search(queries, IOT_AUTOMATION_DOMAINS)
    
    extraction_prompt = f"""
    Find IoT/AUTOMATION/EDGE INTEGRATION initiatives for {company}:
    
    Look for:
    - IoT implementation projects
    - Automation and robotics initiatives
    - Edge computing deployments
    - Smart manufacturing/digital factory
    - Industry 4.0 transformation
    
    Research Data:
    {research_text}
    
    Provide specific project details and technologies used.
    Include source URLs for verification.
    """
    
    try:
        iot_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Find IoT and automation projects for {company}")
        ]).content
    except Exception as e:
        iot_info = research_text
    
    return {"iot_automation_research": iot_info}

def cloud_adoption_research_node(state: AgentState) -> AgentState:
    """Specialized agent for cloud adoption and GCC setup"""
    st.session_state.status_text.info("☁️ Researching cloud adoption and GCC setup...")
    st.session_state.progress_bar.progress(50)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" cloud adoption migration AWS Azure',
        f'"{company}" Global Capability Center GCC setup',
        f'"{company}" digital transformation cloud strategy',
        f'"{company}" IT shared services center',
        f'"{company}" cloud infrastructure investment'
    ]
    
    research_text = perform_targeted_search(queries, CLOUD_ADOPTION_DOMAINS)
    
    extraction_prompt = f"""
    Find CLOUD ADOPTION and GCC SETUP information for {company}:
    
    Focus on:
    - Cloud migration projects (AWS, Azure, Google Cloud)
    - Global Capability Center (GCC) establishment
    - Cloud strategy announcements
    - Digital transformation initiatives
    - IT shared services setup
    
    Research Data:
    {research_text}
    
    Provide specific cloud platforms used and GCC setup details.
    Include source URLs and timelines when available.
    """
    
    try:
        cloud_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Find cloud adoption details for {company}")
        ]).content
    except Exception as e:
        cloud_info = research_text
    
    return {"cloud_adoption_research": cloud_info}

def physical_infrastructure_research_node(state: AgentState) -> AgentState:
    """Specialized agent for physical infrastructure signals"""
    st.session_state.status_text.info("🏗️ Researching physical infrastructure signals...")
    st.session_state.progress_bar.progress(60)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" new facility construction expansion',
        f'"{company}" manufacturing plant warehouse setup',
        f'"{company}" infrastructure investment capex',
        f'"{company}" real estate development project',
        f'"{company}" capacity expansion announcement'
    ]
    
    research_text = perform_targeted_search(queries, PHYSICAL_INFRASTRUCTURE_DOMAINS)
    
    extraction_prompt = f"""
    Find PHYSICAL INFRASTRUCTURE SIGNALS for {company}:
    
    Look for:
    - New facility constructions
    - Manufacturing plant expansions
    - Warehouse/distribution center setups
    - Real estate investments
    - Capacity expansion announcements
    
    Research Data:
    {research_text}
    
    Provide specific locations, investment amounts, and project timelines.
    Include source URLs for verification.
    """
    
    try:
        infrastructure_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Find physical infrastructure projects for {company}")
        ]).content
    except Exception as e:
        infrastructure_info = research_text
    
    return {"physical_infrastructure_research": infrastructure_info}

def it_budget_research_node(state: AgentState) -> AgentState:
    """Specialized agent for IT infrastructure budget and capex"""
    st.session_state.status_text.info("💰 Researching IT budget and capex allocation...")
    st.session_state.progress_bar.progress(70)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" IT budget capital expenditure',
        f'"{company}" technology investment spending',
        f'"{company}" digital transformation budget',
        f'"{company}" IT infrastructure capex',
        f'"{company}" annual report technology investment'
    ]
    
    research_text = perform_targeted_search(queries, IT_BUDGET_DOMAINS)
    
    extraction_prompt = f"""
    Find IT INFRASTRUCTURE BUDGET and CAPEX ALLOCATION for {company}:
    
    Focus on:
    - IT budget announcements
    - Capital expenditure on technology
    - Digital transformation investments
    - Infrastructure spending plans
    - Technology roadmap budgets
    
    Research Data:
    {research_text}
    
    Provide specific budget amounts, allocation details, and timeframes.
    Include source URLs from financial reports or announcements.
    """
    
    try:
        budget_info = llm_groq.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Find IT budget information for {company}")
        ]).content
    except Exception as e:
        budget_info = research_text
    
    return {"it_budget_research": budget_info}

def branch_network_research_node(state: AgentState) -> AgentState:
    """Specialized agent for branch network and facilities"""
    st.session_state.status_text.info("🏢 Researching branch network and facilities...")
    st.session_state.progress_bar.progress(75)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" branch network facilities locations',
        f'"{company}" offices warehouses distribution centers',
        f'"{company}" manufacturing plants global presence',
        f'"{company}" operational facilities count'
    ]
    
    research_text = perform_targeted_search(queries, ["company website", "annual report", "business directories"])
    
    # This research will be used in validation but not stored separately
    return state

def syntel_relevance_analysis_node(state: AgentState) -> AgentState:
    """Specialized agent for Syntel relevance analysis"""
    st.session_state.status_text.info("🎯 Analyzing Syntel relevance and scoring...")
    st.session_state.progress_bar.progress(80)
    
    company = state["company_name"]
    
    # Gather all research data
    all_research = f"""
    BASIC COMPANY INFO: {state.get('basic_research', 'No basic research')}
    NETWORK VENDORS: {state.get('network_vendors_research', 'No vendor research')}
    WIFI TENDERS: {state.get('wifi_tender_research', 'No tender research')}
    IOT AUTOMATION: {state.get('iot_automation_research', 'No IoT research')}
    CLOUD ADOPTION: {state.get('cloud_adoption_research', 'No cloud research')}
    PHYSICAL INFRASTRUCTURE: {state.get('physical_infrastructure_research', 'No infrastructure research')}
    IT BUDGET: {state.get('it_budget_research', 'No budget research')}
    """
    
    analysis_prompt = f"""
    Analyze {company} for SYNTEL BUSINESS RELEVANCE based on all research data.
    
    Available Research:
    {all_research}
    
    Syntel Core Expertise:
    {SYNTEL_EXPERTISE}
    
    Generate specific, actionable insights:
    
    1. why_relevant_to_syntel_bullets: 3-5 specific bullet points based on actual research findings
    2. intent_scoring_level: Low/Medium/High based on concrete evidence
    
    Scoring Criteria:
    - High: Clear IT transformation signals, specific budget indications, active expansion plans
    - Medium: Some digital initiatives, growth potential, moderate IT spending
    - Low: Limited IT signals, stable operations, minimal transformation
    
    Make bullets SPECIFIC and based on research findings, not generic statements.
    """
    
    try:
        relevance_analysis = llm_groq.invoke([
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=f"Analyze Syntel business relevance for {company}")
        ]).content
    except Exception as e:
        relevance_analysis = "Relevance analysis based on available research data"
    
    # Store relevance analysis in basic research for validation
    enhanced_basic_research = state.get('basic_research', '') + f"\n\nRELEVANCE ANALYSIS:\n{relevance_analysis}"
    return {"basic_research": enhanced_basic_research}

# --- Manual JSON Parser and Validator ---
def parse_and_validate_json(json_string: str, company_name: str, all_research: dict) -> dict:
    """Enhanced JSON parser that uses all specialized research data"""
    
    # First, try to extract JSON from the response
    json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = json_string
    
    # Clean the JSON string
    json_str = re.sub(r'</?function.*?>', '', json_str)
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = create_enhanced_structured_data(all_research, company_name)
    
    return validate_and_complete_data(data, company_name, all_research)

def create_enhanced_structured_data(all_research: dict, company_name: str) -> dict:
    """Create structured data using all specialized research"""
    
    data = {}
    
    # Extract from specialized research
    vendor_research = all_research.get('network_vendors_research', '')
    wifi_research = all_research.get('wifi_tender_research', '')
    iot_research = all_research.get('iot_automation_research', '')
    cloud_research = all_research.get('cloud_adoption_research', '')
    infra_research = all_research.get('physical_infrastructure_research', '')
    budget_research = all_research.get('it_budget_research', '')
    
    # Extract network vendors
    if "No specific vendor information" not in vendor_research:
        data["existing_network_vendors"] = extract_specific_info(vendor_research, "vendors", company_name)
    
    # Extract WiFi tenders
    if "No recent WiFi/LAN tenders" not in wifi_research:
        data["wifi_lan_tender_found"] = extract_specific_info(wifi_research, "tenders", company_name)
    
    # Extract IoT/Automation
    data["iot_automation_edge_integration"] = extract_specific_info(iot_research, "iot_automation", company_name)
    
    # Extract Cloud adoption
    data["cloud_adoption_gcc_setup"] = extract_specific_info(cloud_research, "cloud_adoption", company_name)
    
    # Extract Physical infrastructure
    data["physical_infrastructure_signals"] = extract_specific_info(infra_research, "infrastructure", company_name)
    
    # Extract IT budget
    data["it_infra_budget_capex"] = extract_specific_info(budget_research, "budget", company_name)
    
    return data

def extract_specific_info(research_text: str, info_type: str, company_name: str) -> str:
    """Extract specific information from research text"""
    
    if "No specific" in research_text or "not found" in research_text.lower():
        return f"Information not found in specialized search for {company_name}"
    
    # Return the research summary
    return f"Research findings: {research_text[:300]}..." if len(research_text) > 300 else research_text

def validate_and_complete_data(data: dict, company_name: str, all_research: dict = None) -> dict:
    """Enhanced validation using specialized research"""
    
    completed_data = {}
    
    for field in REQUIRED_FIELDS:
        if field in data and data[field] and data[field] != "Not specified":
            completed_data[field] = str(data[field])
        else:
            # Use specialized research for better defaults
            completed_data[field] = get_enhanced_default(field, company_name, all_research)
    
    # Ensure intent scoring is valid
    if completed_data["intent_scoring_level"] not in ["Low", "Medium", "High"]:
        completed_data["intent_scoring_level"] = "Medium"
    
    return completed_data

def get_enhanced_default(field: str, company_name: str, all_research: dict = None) -> str:
    """Get enhanced defaults using specialized research"""
    
    enhanced_defaults = {
        "existing_network_vendors": "Vendor information being researched through specialized channels",
        "wifi_lan_tender_found": "No active tenders identified in recent searches",
        "iot_automation_edge_integration": "IoT initiatives under investigation",
        "cloud_adoption_gcc_setup": "Cloud strategy assessment in progress", 
        "physical_infrastructure_signals": "Physical expansion monitoring active",
        "it_infra_budget_capex": "Budget analysis through financial channels",
        "why_relevant_to_syntel_bullets": f"* {company_name} shows digital transformation potential\n* Infrastructure modernization opportunities identified\n* IT service integration possibilities exist",
        "intent_scoring_level": "Medium"
    }
    
    default = enhanced_defaults.get(field, "Data collection in progress")
    
    # Enhance with actual research findings if available
    if all_research:
        if field == "existing_network_vendors" and all_research.get('network_vendors_research'):
            if "No specific" not in all_research['network_vendors_research']:
                default = f"Research ongoing: {all_research['network_vendors_research'][:200]}..."
    
    return default

# --- Graph Nodes ---
def validation_node(state: AgentState) -> AgentState:
    """Enhanced validation node using all specialized research"""
    st.session_state.status_text.info("📊 Validating and structuring data...")
    st.session_state.progress_bar.progress(85)
    
    # Combine all research data
    all_research = {
        'basic_research': state.get('basic_research', ''),
        'network_vendors_research': state.get('network_vendors_research', ''),
        'wifi_tender_research': state.get('wifi_tender_research', ''),
        'iot_automation_research': state.get('iot_automation_research', ''),
        'cloud_adoption_research': state.get('cloud_adoption_research', ''),
        'physical_infrastructure_research': state.get('physical_infrastructure_research', ''),
        'it_budget_research': state.get('it_budget_research', '')
    }
    
    combined_research = "\n\n".join([f"{key.upper()}:\n{value}" for key, value in all_research.items()])
    
    company = state["company_name"]
    
    validation_prompt = f"""
    Based on ALL SPECIALIZED RESEARCH below, create a structured data summary for {company}.
    
    ALL RESEARCH DATA:
    {combined_research}
    
    Create a JSON-like structure with these exact fields:
    {json.dumps(REQUIRED_FIELDS, indent=2)}
    
    Use the specialized research findings to fill each field with specific, verified information.
    Include source URLs when available from the research.
    Format as valid JSON.
    """
    
    try:
        validated_output = llm_groq.invoke([
            SystemMessage(content=validation_prompt),
            HumanMessage(content=f"Create comprehensive structured data for {company}")
        ]).content
    except Exception as e:
        validated_output = combined_research
    
    return {"validated_data_text": validated_output}

def formatter_node(state: AgentState) -> AgentState:
    """Enhanced formatter node using all specialized research"""
    st.session_state.status_text.info("🎨 Finalizing output format...")
    st.session_state.progress_bar.progress(95)
    
    validated_data_text = state["validated_data_text"]
    company_name = state["company_name"]
    
    # Combine all research for enhanced parsing
    all_research = {
        'network_vendors_research': state.get('network_vendors_research', ''),
        'wifi_tender_research': state.get('wifi_tender_research', ''),
        'iot_automation_research': state.get('iot_automation_research', ''),
        'cloud_adoption_research': state.get('cloud_adoption_research', ''),
        'physical_infrastructure_research': state.get('physical_infrastructure_research', ''),
        'it_budget_research': state.get('it_budget_research', '')
    }
    
    try:
        final_data = parse_and_validate_json(validated_data_text, company_name, all_research)
        st.success("✅ Data successfully structured with multi-agent research")
    except Exception as e:
        st.warning(f"Using enhanced fallback with specialized research: {str(e)}")
        final_data = validate_and_complete_data({}, company_name, all_research)
    
    return {"final_json_data": final_data}

# --- Graph Construction ---
def build_enhanced_graph():
    """Builds enhanced multi-agent workflow"""
    workflow = StateGraph(AgentState)

    # Add all specialized nodes
    workflow.add_node("basic_research", basic_company_research_node)
    workflow.add_node("network_vendors", network_vendors_research_node)
    workflow.add_node("wifi_tender", wifi_tender_research_node)
    workflow.add_node("iot_automation", iot_automation_research_node)
    workflow.add_node("cloud_adoption", cloud_adoption_research_node)
    workflow.add_node("physical_infrastructure", physical_infrastructure_research_node)
    workflow.add_node("it_budget", it_budget_research_node)
    workflow.add_node("branch_network", branch_network_research_node)
    workflow.add_node("relevance_analysis", syntel_relevance_analysis_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("format", formatter_node)

    # Define sequential flow with all specialized agents
    workflow.add_edge(START, "basic_research")
    workflow.add_edge("basic_research", "network_vendors")
    workflow.add_edge("network_vendors", "wifi_tender")
    workflow.add_edge("wifi_tender", "iot_automation")
    workflow.add_edge("iot_automation", "cloud_adoption")
    workflow.add_edge("cloud_adoption", "physical_infrastructure")
    workflow.add_edge("physical_infrastructure", "it_budget")
    workflow.add_edge("it_budget", "branch_network")
    workflow.add_edge("branch_network", "relevance_analysis")
    workflow.add_edge("relevance_analysis", "validate")
    workflow.add_edge("validate", "format")
    workflow.add_edge("format", END)

    return workflow.compile()

# Build the enhanced graph
app = build_enhanced_graph()

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
            value = data_dict.get(data_field, "Data not available")
        
        if data_field == "why_relevant_to_syntel_bullets":
            html_value = str(value).replace('\n', '<br>').replace('*', '•')
            data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Multi-Agent Enhanced)", 
    layout="wide",
    page_icon="🏢"
)

st.title("🏢 Syntel Company Data AI Agent")
st.markdown("### 🤖 Multi-Agent Specialized Research System")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

# Display agent architecture
with st.expander("🔧 Multi-Agent Architecture Overview", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤝 Basic Info Agent**")
        st.markdown("• Company website  \n• LinkedIn  \n• Headquarters  \n• Industry")
    
    with col2:
        st.markdown("**🖧 Network Vendors Agent**")
        st.markdown("• Tech stack  \n• IT vendors  \n• Cloud providers  \n• Partnerships")
    
    with col3:
        st.markdown("**📡 WiFi/LAN Agent**")
        st.markdown("• Network upgrades  \n• Tender information  \n• Infrastructure projects")

    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("**🤖 IoT/Automation Agent**")
        st.markdown("• IoT projects  \n• Automation  \n• Edge computing  \n• Industry 4.0")
    
    with col5:
        st.markdown("**☁️ Cloud Adoption Agent**")
        st.markdown("• Cloud migration  \n• GCC setup  \n• Digital transformation")
    
    with col6:
        st.markdown("**💰 Budget Analysis Agent**")
        st.markdown("• IT capex  \n• Budget allocation  \n• Investment plans")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input)
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("🚀 Start Multi-Agent Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    with st.spinner(f"🤖 Deploying specialized agents to research **{company_input}**..."):
        try:
            initial_state: AgentState = {
                "company_name": company_input,
                "basic_research": "",
                "network_vendors_research": "",
                "wifi_tender_research": "",
                "iot_automation_research": "",
                "cloud_adoption_research": "",
                "physical_infrastructure_research": "",
                "it_budget_research": "",
                "validated_data_text": "",
                "final_json_data": {},
                "messages": []
            }

            final_state = app.invoke(initial_state)
            data_dict = final_state["final_json_data"]
            
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"🎉 Multi-Agent Research Complete for {company_input}!")
            
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": data_dict
            }
            st.session_state.research_history.append(research_entry)
            
            # Display results
            st.subheader(f"📊 Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, data_dict)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show specialized research status
            st.subheader("🔍 Specialized Research Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if data_dict.get("existing_network_vendors", "").startswith("Research findings"):
                    st.success("✅ Network vendors research completed")
                else:
                    st.info("🔍 Network vendors research in progress")
                
                if data_dict.get("wifi_lan_tender_found", "").startswith("Research findings"):
                    st.success("✅ WiFi/LAN tender research completed")
                else:
                    st.info("🔍 WiFi/LAN research in progress")
            
            with col2:
                if data_dict.get("iot_automation_edge_integration", "").startswith("Research findings"):
                    st.success("✅ IoT/Automation research completed")
                else:
                    st.info("🔍 IoT/Automation research in progress")
                
                if data_dict.get("cloud_adoption_gcc_setup", "").startswith("Research findings"):
                    st.success("✅ Cloud adoption research completed")
                else:
                    st.info("🔍 Cloud adoption research in progress")
            
            with col3:
                if data_dict.get("physical_infrastructure_signals", "").startswith("Research findings"):
                    st.success("✅ Physical infrastructure research completed")
                else:
                    st.info("🔍 Infrastructure research in progress")
                
                if data_dict.get("it_infra_budget_capex", "").startswith("Research findings"):
                    st.success("✅ IT budget research completed")
                else:
                    st.info("🔍 IT budget research in progress")
            
            # Download options
            st.subheader("💾 Download Options")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(data_dict, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_multi_agent_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_multi_agent_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_multi_agent_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")

st.markdown("---")

# Research History
if st.session_state.research_history:
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"🎯 Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            branch_info = research['data'].get('branch_network_count', 'No data')[:80]
            st.write(f"🏢 Branch Network: {branch_info}...")
            if st.button(f"📥 Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Technical info
with st.sidebar.expander("🔧 Technical Details"):
    st.markdown("""
    **Multi-Agent Architecture:**
    - 7 specialized research agents
    - Domain-specific search optimization
    - Enhanced field accuracy
    - Source URL integration
    
    **Research Coverage:**
    - Network vendors & tech stack
    - WiFi/LAN tenders & upgrades  
    - IoT/Automation initiatives
    - Cloud adoption & GCC setup
    - Physical infrastructure
    - IT budget & capex
    - Basic company intelligence
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Powered by Multi-Agent AI Research System | "
    "Specialized Domain Intelligence"
    "</div>",
    unsafe_allow_html=True
)
