import streamlit as st
import pandas as pd
import json
import operator
import re
from typing import TypedDict, Annotated, List, Dict
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
    "crownworldwide.com", "microsoft.com", "sap.com", "oracle.com", "infosys.com",
    "techcircle.in", "vmware.com", "cisco.com", "aws.amazon.com", "azure.microsoft.com"
]

WIFI_TENDER_DOMAINS = [
    "spectra.co", "enterprise.spectra.com", "economictimes.indiatimes.com",
    "tendersinfo.com", "tendertiger.com", "networkworld.com", "gem.gov.in"
]

IOT_AUTOMATION_DOMAINS = [
    "automationworld.com", "iotinsider.com", "economictimes.indiatimes.com",
    "expresscomputer.in", "iotworldtoday.com", "techemerge.org"
]

CLOUD_ADOPTION_DOMAINS = [
    "economictimes.indiatimes.com", "techcircle.in", "cio.economictimes.indiatimes.com",
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com"
]

PHYSICAL_INFRASTRUCTURE_DOMAINS = [
    "constructionweekonline.in", "constructionworld.in", "economictimes.indiatimes.com",
    "businessstandard.com", "moneycontrol.com", "projectsToday.com"
]

IT_BUDGET_DOMAINS = [
    "economictimes.indiatimes.com", "business-standard.com", "moneycontrol.com",
    "livemint.com", "thehindubusinessline.com"
]

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

# --- Enhanced LangGraph State Definition ---
class AgentState(TypedDict):
    """Enhanced state with all research fields and completion tracking"""
    company_name: str
    # Research fields with completion status
    basic_info_complete: bool
    basic_info_data: dict
    
    network_vendors_complete: bool
    network_vendors_data: dict
    
    wifi_tender_complete: bool
    wifi_tender_data: dict
    
    iot_automation_complete: bool
    iot_automation_data: dict
    
    cloud_adoption_complete: bool
    cloud_adoption_data: dict
    
    physical_infrastructure_complete: bool
    physical_infrastructure_data: dict
    
    it_budget_complete: bool
    it_budget_data: dict
    
    # Final output
    all_research_complete: bool
    final_json_data: dict
    research_summary: str

# --- Syntel Core Offerings ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Enhanced Search Utility Function ---
def perform_targeted_search(queries: list, preferred_domains: list = None) -> tuple:
    """Perform targeted search and return both formatted text and raw results"""
    all_results = []
    
    for query in queries:
        try:
            time.sleep(1)  # Rate limiting
            
            if preferred_domains:
                domain_query = f"{query} site:{' OR site:'.join(preferred_domains[:3])}"
                results = search_tool.invoke({"query": domain_query, "max_results": 2})
            else:
                results = search_tool.invoke({"query": query, "max_results": 2})
            
            if results:
                for result in results:
                    content = result.get('content', '')
                    if len(content) > 50:
                        all_results.append({
                            "title": result.get('title', ''),
                            "content": content[:500],
                            "url": result.get('url', '')
                        })
        except Exception as e:
            continue
    
    if not all_results:
        return "No specific information found in targeted search.", []
    
    research_text = "SEARCH RESULTS:\n\n"
    for i, result in enumerate(all_results):
        research_text += f"🔗 Source {i+1}:\n"
        research_text += f"📖 Title: {result['title']}\n"
        research_text += f"🌐 URL: {result['url']}\n"
        research_text += f"📝 Content: {result['content']}\n"
        research_text += "─" * 60 + "\n"
    
    return research_text, all_results

# --- Enhanced Research Agents with Strict Field Mapping ---
def basic_info_agent(state: AgentState) -> AgentState:
    """Enhanced agent for basic company information with strict field mapping"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("🔍 Researching basic company information...")
    progress_placeholder.progress(10)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" official website',
        f'"{company}" LinkedIn company page',
        f'"{company}" headquarters location',
        f'"{company}" industry business',
        f'"{company}" employee count size',
        f'"{company}" revenue financial results'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, ["linkedin.com", "zoominfo.com", "crunchbase.com"])
    
    # Enhanced extraction with strict field mapping
    extraction_prompt = f"""
    Extract DETAILED BASIC COMPANY INFORMATION for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    You MUST extract and map to these EXACT field names:
    - company_website_url: Official website URL
    - linkedin_url: LinkedIn company page URL  
    - industry_category: Specific industry classification
    - employee_count_linkedin: Employee count with source
    - headquarters_location: Full headquarters address/location
    - revenue_source: Revenue streams and business model
    
    Return ONLY a JSON object with these exact field names. For example:
    {{
        "company_website_url": "https://www.snowman.in [Source: search results]",
        "linkedin_url": "https://linkedin.com/company/snowmanlogistics [Source: LinkedIn]",
        "industry_category": "Cold Chain Logistics [Source: company info]",
        "employee_count_linkedin": "1,001-5,000 employees [Source: LinkedIn]",
        "headquarters_location": "Bangalore, Karnataka [Source: company website]",
        "revenue_source": "Cold chain warehousing and logistics services [Source: financial reports]"
    }}
    
    Be specific and include source URLs when available.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a precise data extraction agent. Return only valid JSON with exact field names."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            basic_data = json.loads(json_match.group(0))
        else:
            basic_data = {}
            
    except Exception as e:
        basic_data = {}
    
    # Ensure all basic fields are present with proper fallbacks
    required_basic_fields = {
        "company_website_url": "Information not found in research",
        "linkedin_url": "Information not found in research", 
        "industry_category": "Information not found in research",
        "employee_count_linkedin": "Information not found in research",
        "headquarters_location": "Information not found in research",
        "revenue_source": "Information not found in research"
    }
    
    for field, default in required_basic_fields.items():
        if field not in basic_data or not basic_data[field]:
            basic_data[field] = default
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "basic_info_complete": True,
        "basic_info_data": basic_data
    }

def network_vendors_agent(state: AgentState) -> AgentState:
    """Enhanced agent for existing network vendors and tech stack"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("🖧 Researching network vendors and tech stack...")
    progress_placeholder.progress(20)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" technology stack IT infrastructure',
        f'"{company}" network vendors Cisco VMware',
        f'"{company}" software platforms SAP Oracle',
        f'"{company}" cloud providers AWS Azure',
        f'"{company}" IT partnerships vendors'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, NETWORK_VENDOR_DOMAINS)
    
    extraction_prompt = f"""
    Extract EXISTING NETWORK VENDORS and TECH STACK for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    Extract SPECIFIC information about existing_network_vendors field.
    
    Return ONLY a JSON object with this structure:
    {{
        "existing_network_vendors": "Detailed description of specific vendors, technologies, and platforms used by {company}. Include vendor names like Cisco, VMware, SAP, Oracle, Microsoft, AWS, Azure, Infosys, etc. with source URLs when available."
    }}
    
    Be specific and include actual vendor names, technologies, and source URLs.
    If no specific vendors found, state "No specific vendor information found in research".
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract specific technology vendor information. Return only valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            vendor_data = json.loads(json_match.group(0))
        else:
            vendor_data = {"existing_network_vendors": "No specific vendor information found in research"}
            
    except Exception as e:
        vendor_data = {"existing_network_vendors": "Research encountered an error"}
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "network_vendors_complete": True,
        "network_vendors_data": vendor_data
    }

def wifi_tender_agent(state: AgentState) -> AgentState:
    """Enhanced agent for WiFi/LAN tenders and upgrades"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("📡 Researching WiFi/LAN tenders and upgrades...")
    progress_placeholder.progress(30)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" WiFi upgrade network infrastructure',
        f'"{company}" LAN tender bidding',
        f'"{company}" network expansion project',
        f'"{company}" IT infrastructure tender',
        f'"{company}" wireless network deployment'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, WIFI_TENDER_DOMAINS)
    
    extraction_prompt = f"""
    Find RECENT Wi-Fi UPGRADES or LAN TENDERS for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    Extract SPECIFIC information for wifi_lan_tender_found field.
    
    Return ONLY a JSON object with this structure:
    {{
        "wifi_lan_tender_found": "Specific details about any WiFi upgrades, LAN tenders, or network infrastructure projects with dates, locations, and source URLs. If no tenders found, state 'No recent WiFi/LAN tenders identified in research'."
    }}
    
    Include specific details, dates, locations, and source URLs when available.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You identify network infrastructure tenders and upgrades. Return only valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            wifi_data = json.loads(json_match.group(0))
        else:
            wifi_data = {"wifi_lan_tender_found": "No recent WiFi/LAN tenders identified in research"}
            
    except Exception as e:
        wifi_data = {"wifi_lan_tender_found": "Research encountered an error"}
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "wifi_tender_complete": True,
        "wifi_tender_data": wifi_data
    }

def iot_automation_agent(state: AgentState) -> AgentState:
    """Enhanced agent for IoT/Automation/Edge integration"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("🤖 Researching IoT and automation initiatives...")
    progress_placeholder.progress(40)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" IoT Internet of Things implementation',
        f'"{company}" automation projects digital transformation',
        f'"{company}" edge computing initiatives',
        f'"{company}" smart factory Industry 4.0',
        f'"{company}" robotics process automation'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, IOT_AUTOMATION_DOMAINS)
    
    extraction_prompt = f"""
    Find IoT/AUTOMATION/EDGE INTEGRATION initiatives for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    Extract SPECIFIC information for iot_automation_edge_integration field.
    
    Return ONLY a JSON object with this structure:
    {{
        "iot_automation_edge_integration": "Detailed description of specific IoT projects, automation initiatives, edge computing deployments, robotics, or smart technologies used by {company}. Include technologies, project scope, and source URLs when available."
    }}
    
    Be specific about technologies used, project scope, and include source URLs.
    If no specific initiatives found, state "No specific IoT/automation initiatives found in research".
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract IoT and automation project details. Return only valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            iot_data = json.loads(json_match.group(0))
        else:
            iot_data = {"iot_automation_edge_integration": "No specific IoT/automation initiatives found in research"}
            
    except Exception as e:
        iot_data = {"iot_automation_edge_integration": "Research encountered an error"}
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "iot_automation_complete": True,
        "iot_automation_data": iot_data
    }

def cloud_adoption_agent(state: AgentState) -> AgentState:
    """Enhanced agent for cloud adoption and GCC setup"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("☁️ Researching cloud adoption and GCC setup...")
    progress_placeholder.progress(50)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" cloud adoption migration AWS Azure',
        f'"{company}" Global Capability Center GCC setup',
        f'"{company}" digital transformation cloud strategy',
        f'"{company}" IT shared services center',
        f'"{company}" cloud infrastructure investment'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, CLOUD_ADOPTION_DOMAINS)
    
    extraction_prompt = f"""
    Find CLOUD ADOPTION and GCC SETUP information for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    Extract SPECIFIC information for cloud_adoption_gcc_setup field.
    
    Return ONLY a JSON object with this structure:
    {{
        "cloud_adoption_gcc_setup": "Detailed description of cloud migration projects, GCC establishment, cloud strategy, specific cloud platforms used (AWS, Azure, Google Cloud), and source URLs when available."
    }}
    
    Be specific about cloud platforms, migration status, GCC setup, and include source URLs.
    If no specific cloud adoption found, state "No specific cloud adoption initiatives found in research".
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract cloud adoption and GCC setup details. Return only valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            cloud_data = json.loads(json_match.group(0))
        else:
            cloud_data = {"cloud_adoption_gcc_setup": "No specific cloud adoption initiatives found in research"}
            
    except Exception as e:
        cloud_data = {"cloud_adoption_gcc_setup": "Research encountered an error"}
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "cloud_adoption_complete": True,
        "cloud_adoption_data": cloud_data
    }

def physical_infrastructure_agent(state: AgentState) -> AgentState:
    """Enhanced agent for physical infrastructure signals"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("🏗️ Researching physical infrastructure signals...")
    progress_placeholder.progress(60)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" new facility construction expansion',
        f'"{company}" manufacturing plant warehouse setup',
        f'"{company}" infrastructure investment capex',
        f'"{company}" real estate development project',
        f'"{company}" capacity expansion announcement'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, PHYSICAL_INFRASTRUCTURE_DOMAINS)
    
    extraction_prompt = f"""
    Find PHYSICAL INFRASTRUCTURE SIGNALS for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    Extract SPECIFIC information for these THREE fields:
    - physical_infrastructure_signals: New facilities, expansions, construction projects
    - branch_network_count: Number of facilities, locations, capacity with specific numbers
    - expansion_news_12mo: Recent expansion announcements with dates and details
    
    Return ONLY a JSON object with this EXACT structure:
    {{
        "physical_infrastructure_signals": "Detailed infrastructure projects with locations, capacities, and timelines",
        "branch_network_count": "Specific facility count, locations, and capacity details", 
        "expansion_news_12mo": "Recent expansion announcements with dates, locations, and investment details"
    }}
    
    Be specific about locations, capacities, investment amounts, dates, and include source URLs.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract physical infrastructure and expansion details. Return only valid JSON with exact field names."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            infra_data = json.loads(json_match.group(0))
        else:
            infra_data = {}
            
    except Exception as e:
        infra_data = {}
    
    # Ensure all infrastructure fields are present
    required_infra_fields = {
        "physical_infrastructure_signals": "No specific infrastructure projects found in research",
        "branch_network_count": "Facility information not available in research",
        "expansion_news_12mo": "No recent expansion announcements found in research"
    }
    
    for field, default in required_infra_fields.items():
        if field not in infra_data or not infra_data[field]:
            infra_data[field] = default
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "physical_infrastructure_complete": True,
        "physical_infrastructure_data": infra_data
    }

def it_budget_agent(state: AgentState) -> AgentState:
    """Enhanced agent for IT infrastructure budget and capex"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("💰 Researching IT budget and capex allocation...")
    progress_placeholder.progress(70)
    
    company = state["company_name"]
    
    queries = [
        f'"{company}" IT budget capital expenditure',
        f'"{company}" technology investment spending',
        f'"{company}" digital transformation budget',
        f'"{company}" IT infrastructure capex',
        f'"{company}" annual report technology investment'
    ]
    
    research_text, raw_results = perform_targeted_search(queries, IT_BUDGET_DOMAINS)
    
    extraction_prompt = f"""
    Find IT INFRASTRUCTURE BUDGET and CAPEX ALLOCATION for {company} from the research below.
    
    RESEARCH DATA:
    {research_text}
    
    Extract SPECIFIC information for it_infra_budget_capex field.
    
    Return ONLY a JSON object with this structure:
    {{
        "it_infra_budget_capex": "Specific budget amounts, capex allocations, investment timelines, and source URLs from financial reports or announcements."
    }}
    
    Be specific about amounts, timeframes, and include source URLs.
    If no specific budget information found, state "No specific IT budget information found in research".
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You extract IT budget and capex information. Return only valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            budget_data = json.loads(json_match.group(0))
        else:
            budget_data = {"it_infra_budget_capex": "No specific IT budget information found in research"}
            
    except Exception as e:
        budget_data = {"it_infra_budget_capex": "Research encountered an error"}
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "it_budget_complete": True,
        "it_budget_data": budget_data
    }

def check_all_research_complete(state: AgentState) -> AgentState:
    """Check if all research agents have completed their work"""
    completed_agents = [
        state.get("basic_info_complete", False),
        state.get("network_vendors_complete", False),
        state.get("wifi_tender_complete", False),
        state.get("iot_automation_complete", False),
        state.get("cloud_adoption_complete", False),
        state.get("physical_infrastructure_complete", False),
        state.get("it_budget_complete", False)
    ]
    
    all_complete = all(completed_agents)
    
    return {"all_research_complete": all_complete}

def final_assembly_agent(state: AgentState) -> AgentState:
    """Enhanced assembly agent that ensures all fields are properly mapped"""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("📊 Assembling final report with field validation...")
    progress_placeholder.progress(95)
    
    company = state["company_name"]
    
    # Combine all research data with proper field validation
    all_data = {}
    
    # Define which agent handles which fields
    field_agent_mapping = {
        "basic_info_data": [
            "linkedin_url", "company_website_url", "industry_category",
            "employee_count_linkedin", "headquarters_location", "revenue_source"
        ],
        "network_vendors_data": ["existing_network_vendors"],
        "wifi_tender_data": ["wifi_lan_tender_found"],
        "iot_automation_data": ["iot_automation_edge_integration"],
        "cloud_adoption_data": ["cloud_adoption_gcc_setup"],
        "physical_infrastructure_data": [
            "physical_infrastructure_signals", "branch_network_count", "expansion_news_12mo"
        ],
        "it_budget_data": ["it_infra_budget_capex"]
    }
    
    # Collect data from each agent with field validation
    for agent_key, fields in field_agent_mapping.items():
        agent_data = state.get(agent_key, {})
        for field in fields:
            if field in agent_data and agent_data[field] and agent_data[field] not in [
                "Research in progress", "Research encountered an error", 
                "Information not found in research"
            ]:
                all_data[field] = agent_data[field]
    
    # Ensure all required fields are present with proper fallbacks
    for field in REQUIRED_FIELDS:
        if field not in all_data:
            all_data[field] = "Information not available from comprehensive research"
    
    # Set default values for fields that should always have something
    all_data.setdefault("digital_transformation_initiatives", "Digital transformation initiatives under investigation")
    all_data.setdefault("it_leadership_change", "IT leadership information being researched")
    
    # Generate enhanced relevance analysis
    research_context = f"""
    Company: {company}
    
    Research Findings:
    - Basic Info: {all_data.get('industry_category', 'Not found')}
    - Network Vendors: {all_data.get('existing_network_vendors', 'Not found')}
    - Infrastructure: {all_data.get('branch_network_count', 'Not found')}
    - Expansion: {all_data.get('expansion_news_12mo', 'Not found')}
    - IoT/Automation: {all_data.get('iot_automation_edge_integration', 'Not found')}
    - Cloud: {all_data.get('cloud_adoption_gcc_setup', 'Not found')}
    - Budget: {all_data.get('it_infra_budget_capex', 'Not found')}
    """
    
    relevance_prompt = f"""
    Based on the following research data for {company}, generate SPECIFIC relevance analysis for Syntel:
    
    {research_context}
    
    Syntel Core Expertise:
    {SYNTEL_EXPERTISE}
    
    Generate TWO specific outputs:
    
    1. why_relevant_to_syntel_bullets: 3-5 SPECIFIC, ACTIONABLE bullet points based on ACTUAL research findings
    2. intent_scoring_level: Low, Medium, or High based on CONCRETE evidence from research
    
    Scoring Criteria:
    - High: Clear IT transformation signals, specific infrastructure expansion, budget indications
    - Medium: Some digital initiatives, growth potential, moderate IT signals  
    - Low: Limited IT signals, stable operations, minimal transformation
    
    Make bullets SPECIFIC and based on actual research findings, not generic statements.
    """
    
    try:
        relevance_response = llm_groq.invoke([
            SystemMessage(content="You analyze business relevance for IT services companies. Be specific and evidence-based."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Extract relevance information with better parsing
        lines = relevance_response.split('\n')
        bullets = []
        scoring = "Medium"  # Default
        
        for line in lines:
            line = line.strip()
            if line.startswith('*') or line.startswith('-') or line.startswith('•'):
                bullets.append(line)
            elif 'intent_scoring_level' in line.lower() or 'scoring' in line.lower():
                if 'high' in line.lower():
                    scoring = "High"
                elif 'low' in line.lower():
                    scoring = "Low"
        
        if bullets:
            all_data["why_relevant_to_syntel_bullets"] = "\n".join(bullets[:5])  # Limit to 5 bullets
        else:
            all_data["why_relevant_to_syntel_bullets"] = relevance_response[:500]  # Fallback
        
        all_data["intent_scoring_level"] = scoring
        
    except Exception as e:
        all_data["why_relevant_to_syntel_bullets"] = "Relevance analysis based on comprehensive multi-agent research"
        all_data["intent_scoring_level"] = "Medium"
    
    # Create detailed research summary
    completed_agents = sum([
        state.get("basic_info_complete", False),
        state.get("network_vendors_complete", False),
        state.get("wifi_tender_complete", False),
        state.get("iot_automation_complete", False),
        state.get("cloud_adoption_complete", False),
        state.get("physical_infrastructure_complete", False),
        state.get("it_budget_complete", False)
    ])
    
    research_summary = f"""
    COMPREHENSIVE RESEARCH COMPLETE FOR {company.upper()}
    
    ✅ {completed_agents}/7 Specialized Agents Completed Research
    📊 {sum(1 for f in REQUIRED_FIELDS if all_data.get(f) and 'not available' not in all_data.get(f, '').lower())}/{len(REQUIRED_FIELDS)} Fields Populated
    
    Agent Completion Status:
    {'✅' if state.get('basic_info_complete') else '❌'} Basic Company Information
    {'✅' if state.get('network_vendors_complete') else '❌'} Network Vendors & Tech Stack  
    {'✅' if state.get('wifi_tender_complete') else '❌'} WiFi/LAN Tender Analysis
    {'✅' if state.get('iot_automation_complete') else '❌'} IoT/Automation Initiatives
    {'✅' if state.get('cloud_adoption_complete') else '❌'} Cloud Adoption & GCC Setup
    {'✅' if state.get('physical_infrastructure_complete') else '❌'} Physical Infrastructure Signals
    {'✅' if state.get('it_budget_complete') else '❌'} IT Budget & Capex Analysis
    
    Final Intent Score: {all_data.get('intent_scoring_level', 'Medium')}
    """
    
    status_placeholder.empty()
    progress_placeholder.empty()
    
    return {
        "final_json_data": all_data,
        "research_summary": research_summary,
        "all_research_complete": True
    }

# --- Graph Construction ---
def build_parallel_research_graph():
    """Build graph where all research agents run in parallel"""
    workflow = StateGraph(AgentState)

    # Add all specialized agent nodes
    workflow.add_node("basic_info", basic_info_agent)
    workflow.add_node("network_vendors", network_vendors_agent)
    workflow.add_node("wifi_tender", wifi_tender_agent)
    workflow.add_node("iot_automation", iot_automation_agent)
    workflow.add_node("cloud_adoption", cloud_adoption_agent)
    workflow.add_node("physical_infrastructure", physical_infrastructure_agent)
    workflow.add_node("it_budget", it_budget_agent)
    workflow.add_node("check_completion", check_all_research_complete)
    workflow.add_node("final_assembly", final_assembly_agent)

    # Start all research agents in parallel
    workflow.add_edge(START, "basic_info")
    workflow.add_edge(START, "network_vendors")
    workflow.add_edge(START, "wifi_tender")
    workflow.add_edge(START, "iot_automation")
    workflow.add_edge(START, "cloud_adoption")
    workflow.add_edge(START, "physical_infrastructure")
    workflow.add_edge(START, "it_budget")

    # After each research agent, check if all are complete
    workflow.add_edge("basic_info", "check_completion")
    workflow.add_edge("network_vendors", "check_completion")
    workflow.add_edge("wifi_tender", "check_completion")
    workflow.add_edge("iot_automation", "check_completion")
    workflow.add_edge("cloud_adoption", "check_completion")
    workflow.add_edge("physical_infrastructure", "check_completion")
    workflow.add_edge("it_budget", "check_completion")

    # Route based on completion status
    def route_after_completion(state: AgentState):
        if state.get("all_research_complete", False):
            return "final_assembly"
        else:
            return "check_completion"

    workflow.add_conditional_edges(
        "check_completion",
        route_after_completion,
        {
            "final_assembly": "final_assembly",
            "check_completion": "check_completion",
        }
    )

    workflow.add_edge("final_assembly", END)

    return workflow.compile()

# Build the graph
app = build_parallel_research_graph()

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
            value = data_dict.get(data_field, "Research complete - detailed information not found")
        
        # Format bullet points for better display
        if data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str):
                html_value = value.replace('\n', '<br>').replace('*', '•').replace('-', '•')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            data_list.append({"Column Header": display_col, "Value": str(value)})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Enhanced Multi-Agent)",
    layout="wide",
    page_icon="🏢"
)

st.title("🏢 Syntel Company Data AI Agent")
st.markdown("### 🤖 Enhanced Multi-Agent Research System")

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"

# Display enhanced agent architecture
with st.expander("🔧 Enhanced Multi-Agent Architecture", expanded=True):
    st.markdown("""
    **🚀 7 Enhanced Specialized Agents:**
    - **🔍 Basic Info Agent**: Company website, LinkedIn, headquarters, industry, revenue
    - **🖧 Network Vendors Agent**: Tech stack, IT vendors, cloud providers with source links  
    - **📡 WiFi/LAN Agent**: Network upgrades, tender information with specific details
    - **🤖 IoT/Automation Agent**: IoT projects, automation initiatives with technologies
    - **☁️ Cloud Adoption Agent**: Cloud migration, GCC setup with platform details
    - **🏗️ Physical Infrastructure Agent**: Facility expansions, construction with locations
    - **💰 IT Budget Agent**: Capex allocation, budget analysis with amounts
    
    **🛡️ Enhanced Features:**
    - ✅ Strict field mapping and validation
    - ✅ Source URL inclusion in all fields
    - ✅ No "Research in progress" in final output
    - ✅ Comprehensive error handling
    - ✅ Field completion tracking
    """)

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input)
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("🚀 Start Enhanced Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # Create progress and status display
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Show initial status
    status_text.info("🤖 Deploying 7 enhanced specialized agents for comprehensive research...")
    progress_bar.progress(5)
    
    with st.spinner(f"Researching **{company_input}** with enhanced multi-agent system..."):
        try:
            # Initialize state
            initial_state: AgentState = {
                "company_name": company_input,
                "basic_info_complete": False,
                "basic_info_data": {},
                "network_vendors_complete": False,
                "network_vendors_data": {},
                "wifi_tender_complete": False,
                "wifi_tender_data": {},
                "iot_automation_complete": False,
                "iot_automation_data": {},
                "cloud_adoption_complete": False,
                "cloud_adoption_data": {},
                "physical_infrastructure_complete": False,
                "physical_infrastructure_data": {},
                "it_budget_complete": False,
                "it_budget_data": {},
                "all_research_complete": False,
                "final_json_data": {},
                "research_summary": ""
            }

            # Run the graph
            final_state = app.invoke(initial_state)
            data_dict = final_state["final_json_data"]
            
            # Update final progress
            progress_bar.progress(100)
            status_text.success(f"🎉 Enhanced Research Complete for {company_input}!")
            
            # Store in history
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": data_dict
            }
            st.session_state.research_history.append(research_entry)
            
            # Display completion message
            st.balloons()
            st.success("✅ All 7 enhanced agents have completed their research!")
            
            # Display final results
            st.subheader(f"📊 Comprehensive Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, data_dict)
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show enhanced research summary
            with st.expander("🔍 Detailed Research Summary", expanded=True):
                st.markdown(final_state.get("research_summary", "Research completed successfully."))
                
                # Show detailed field completion status
                st.subheader("📈 Field Completion Analytics")
                completed_fields = sum(1 for field in REQUIRED_FIELDS if data_dict.get(field) and 'not available' not in data_dict.get(field, '').lower())
                total_fields = len(REQUIRED_FIELDS)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{total_fields}")
                with col2:
                    st.metric("Completion Rate", f"{(completed_fields/total_fields)*100:.1f}%")
                with col3:
                    st.metric("Intent Score", data_dict.get("intent_scoring_level", "Medium"))
            
            # Download options
            st.subheader("💾 Download Complete Report")
            
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
                     file_name=f"{company_input.replace(' ', '_')}_enhanced_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_enhanced_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_enhanced_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            progress_bar.progress(100)
            status_text.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits or network issues. Please try again in a few moments.")

# Research History
if st.session_state.research_history:
    st.sidebar.header("📚 Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"🎯 Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            completed_fields = sum(1 for field in REQUIRED_FIELDS if research['data'].get(field) and 'not available' not in research['data'].get(field, '').lower())
            st.write(f"📊 Fields Completed: {completed_fields}/{len(REQUIRED_FIELDS)}")
            if st.button(f"📥 Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Powered by Enhanced Multi-Agent AI Research System | "
    "Strict Field Mapping | Source URL Integration | No Partial Outputs"
    "</div>",
    unsafe_allow_html=True
)
