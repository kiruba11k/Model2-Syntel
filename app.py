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
    "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
    "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
    "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
    "physical_infrastructure_signals", "it_infra_budget_capex", "core_intent_analysis",
    "why_relevant_to_syntel_bullets", "intent_scoring_level"
]

# --- Core Functions ---

def clean_and_format_url(url: str) -> str:
    if not url or url == "N/A":
        return "N/A"
    if url.startswith('//'):
        url = 'https:' + url
    elif not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.replace(' ', '').strip()

def generate_dynamic_search_queries(company_name: str, field_name: str) -> List[str]:
    field_queries = {
        "branch_network_count": [
            f'"{company_name}" branch network facilities locations count',
            f'"{company_name}" warehouse facility pallet capacity'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion news 2024 2025 new facilities',
            f'"{company_name}" new warehouse construction'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT initiatives'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT infrastructure leadership'
        ],
        "existing_network_vendors": [
            f'"{company_name}" network infrastructure vendors Cisco HPE',
            f'"{company_name}" IT technology stack'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN tender network upgrade'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation implementation',
            f'"{company_name}" robotics automation technology'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption AWS Azure GCC',
            f'"{company_name}" global capability center'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility expansion',
            f'"{company_name}" warehouse development'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget capex investment',
            f'"{company_name}" technology spending'
        ]
    }
    return field_queries.get(field_name, [f'"{company_name}" {field_name}'])

def dynamic_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    queries = generate_dynamic_search_queries(company_name, field_name)
    all_results = []
    
    for query in queries[:2]:
        try:
            time.sleep(0.5)
            results = search_tool.invoke({
                "query": query, 
                "max_results": 3,
                "search_depth": "advanced"
            })
            
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        if len(content) > 50:
                            all_results.append({
                                "title": result.get('title', ''),
                                "content": content[:800],
                                "url": result.get('url', ''),
                                "field": field_name,
                                "query": query
                            })
        except Exception:
            continue
    return all_results

def get_detailed_extraction_prompt(company_name: str, field_name: str, research_context: str, source_urls: List[str]) -> str:
    
    # Format source URLs for the prompt
    sources_text = ", ".join(source_urls) if source_urls else "No specific URLs found"
    
    field_specific_instructions = {
        "branch_network_count": "Extract ONLY facility counts, warehouse numbers, pallet capacity explicitly mentioned.",
        "expansion_news_12mo": "Extract ONLY recent expansion announcements with locations and dates explicitly mentioned.",
        "digital_transformation_initiatives": "Extract ONLY specific digital projects, systems, or platforms explicitly mentioned.",
        "it_leadership_change": "Extract ONLY CIO/CTO/Head Infra changes with names and dates explicitly mentioned.",
        "existing_network_vendors": "Extract ONLY network hardware vendors (Cisco, HPE, etc.) or tech stack explicitly mentioned.",
        "wifi_lan_tender_found": "Extract ONLY specific WiFi/LAN tenders or upgrade projects explicitly mentioned.",
        "iot_automation_edge_integration": "Extract ONLY IoT, automation, or edge computing projects explicitly mentioned.",
        "cloud_adoption_gcc_setup": "Extract ONLY cloud providers or GCC setup plans explicitly mentioned.",
        "physical_infrastructure_signals": "Extract ONLY new construction, facility expansions with details explicitly mentioned.",
        "it_infra_budget_capex": "Extract ONLY IT budget figures, capex amounts, or investment numbers explicitly mentioned."
    }
    
    field_instruction = field_specific_instructions.get(field_name, "Extract ONLY factual information explicitly mentioned.")
    
    prompt = f"""
    RESEARCH TASK: Extract information about {field_name} for {company_name}
    
    SOURCE CONTENT:
    {research_context}
    
    AVAILABLE SOURCES: {sources_text}
    
    STRICT INSTRUCTIONS:
    1. {field_instruction}
    2. USE ONLY information explicitly provided in the source content above
    3. DO NOT use any prior knowledge or make assumptions
    4. DO NOT invent, estimate, or calculate any numbers
    5. If information is NOT found in the source content, output ONLY: "N/A"
    6. If information is found, provide a CONCISE 1-2 sentence summary
    7. ALWAYS include the relevant source URLs at the end
    
    OUTPUT FORMAT:
    [Concise factual summary] [Sources: url1, url2]
    
    If no information found:
    N/A
    
    EXTRACTION RESULT:
    """
    
    return prompt

def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    
    if not search_results:
        return "N/A"
    
    # Build research context with clear source attribution
    research_context = ""
    for i, result in enumerate(search_results[:3]):
        content = result['content'][:500]  # Limited content to prevent context overflow
        research_context += f"SOURCE {i+1} - {result.get('title', 'No Title')}:\n{content}\n\n"
    
    # Extract and validate URLs
    unique_urls = []
    for result in search_results:
        url = result.get('url', '')
        if url and url.startswith(('http://', 'https://')) and url not in unique_urls:
            unique_urls.append(url)
    
    source_urls = unique_urls[:2]  # Use max 2 most relevant URLs
    
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context, source_urls)
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="""You are a strict factual research analyst. Follow these rules:
            CRITICAL RULES:
            1. EXTRACT ONLY information explicitly stated in the provided source content
            2. DO NOT use any external knowledge or make assumptions
            3. If information is not explicitly found, output ONLY "N/A"
            4. DO NOT invent numbers, names, or facts
            5. Be concise and factual
            6. Include actual source URLs only if information is found"""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Strict validation for hallucination prevention
        response_lower = response.lower()
        if (not response or 
            response == "N/A" or
            len(response) < 5 or
            'not found' in response_lower or
            'no information' in response_lower or
            'unable to' in response_lower or
            'cannot provide' in response_lower or
            'based on my knowledge' in response_lower):
            return "N/A"
        
        # Clean response
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure proper source formatting
        if response != "N/A" and source_urls:
            # Check if sources are already included
            if not any(url in response for url in source_urls):
                source_text = f" [Sources: {', '.join(source_urls)}]"
                response += source_text
        
        # Final length management without mid-sentence cuts
        if len(response) > 300:
            # Find a clean cutoff point
            if '.' in response[:280]:
                cutoff = response[:280].rfind('.') + 1
                response = response[:cutoff]
            elif ' ' in response[:280]:
                cutoff = response[:280].rfind(' ') 
                response = response[:cutoff]
        
        return response[:350]  # Strict limit
            
    except Exception as e:
        return "N/A"

def analyze_core_intent_article(article_url: str, company_name: str) -> str:
    """
    Analyze the core intent article with robust hallucination prevention
    """
    if not article_url or not article_url.startswith(('http://', 'https://')):
        return "N/A"
    
    try:
        # Use Tavily to fetch the specific article content
        search_results = search_tool.invoke({
            "query": f'"{company_name}"',
            "max_results": 5,
            "include_raw_content": True,
            "search_depth": "advanced"
        })
        
        article_content = ""
        relevant_urls = []
        
        # Find the most relevant content matching the article URL
        for result in search_results:
            if isinstance(result, dict):
                url = result.get('url', '')
                content = result.get('content', '') or result.get('snippet', '')
                
                # Prioritize content from the provided article URL
                if article_url in url or url in article_url:
                    article_content = content[:2000] if content else ""
                    relevant_urls.append(url)
                    break
                elif content and len(content) > 100:
                    # Use other relevant content as fallback
                    article_content = content[:1500]
                    relevant_urls.append(url)
        
        if not article_content:
            return "N/A"
        
        # Use the original article URL as primary source
        primary_source = article_url if article_url not in relevant_urls else relevant_urls[0]
        
        prompt = f"""
        ANALYZE THIS COMPANY CONTENT FOR CORE BUSINESS INTENT:
        
        CONTENT:
        {article_content}
        
        STRICT REQUIREMENTS:
        1. Extract ONLY the main strategic business objective mentioned
        2. Focus on expansion, technology adoption, or infrastructure plans
        3. Use ONLY information explicitly stated in the content
        4. If no clear strategic intent is stated, output "N/A"
        5. Be extremely concise (1 sentence maximum)
        6. DO NOT invent or assume any strategic plans
        
        OUTPUT FORMAT:
        [One sentence about core business intent] [Source: {primary_source}]
        
        If no clear intent found:
        N/A
        
        ANALYSIS:
        """
        
        response = llm_groq.invoke([
            SystemMessage(content="""You analyze business strategy. Extract ONLY explicitly stated strategic intent.
            CRITICAL: If the content doesn't clearly state strategic objectives, output 'N/A'"""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Validate response
        if (not response or 
            response == "N/A" or 
            len(response) < 20 or
            'not mentioned' in response.lower() or
            'no clear' in response.lower()):
            return "N/A"
            
        return response
        
    except Exception as e:
        return "N/A"

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str, core_intent_analysis: str) -> tuple:
    """
    Generates relevance analysis with STRICT core intent integration and hallucination prevention
    """
    
    # Build factual context from available data only
    factual_context = []
    
    # Only include fields with actual data (not N/A)
    for field in ["expansion_news_12mo", "iot_automation_edge_integration", 
                  "branch_network_count", "digital_transformation_initiatives",
                  "physical_infrastructure_signals"]:
        value = company_data.get(field, "N/A")
        if value != "N/A":
            # Remove source URLs for cleaner context
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            if clean_value and len(clean_value) > 10:
                factual_context.append(f"{field}: {clean_value}")
    
    data_context = " | ".join(factual_context) if factual_context else "Limited specific data available"
    
    # Determine if core intent should influence scoring
    core_intent_weight = "HIGH" if core_intent_analysis != "N/A" else "LOW"
    
    prompt = f"""
    ASSESS RELEVANCE TO SYNTEL (Wi-Fi & Network Integration for Indian enterprises)
    
    COMPANY: {company_name}
    
    AVAILABLE FACTS:
    {data_context}
    
    CORE INTENT ANALYSIS:
    {core_intent_analysis}
    (Core Intent Weight: {core_intent_weight})
    
    SYNTEL FOCUS AREAS:
    - Warehouses, Manufacturing, Logistics facilities
    - Network infrastructure for new expansions
    - IoT/Automation implementations requiring WiFi
    - Digital transformation projects
    
    SCORING CRITERIA:
    HIGH: Expansion signals + IoT/Digital projects + Clear network needs
    MEDIUM: Some expansion or technology signals
    LOW: Limited relevant signals
    
    OUTPUT FORMAT:
    CompanyName\t• Bullet1 • Bullet2 • Bullet3\tScore
    
    BULLET REQUIREMENTS:
    - MUST reference specific available facts
    - MUST connect to network/WiFi infrastructure needs
    - If core intent available, MUST reference it in first bullet
    
    RELEVANCE ASSESSMENT:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="""You assess company relevance for network infrastructure projects.
            OUTPUT MUST BE: CompanyName\t• Fact1 • Fact2 • Fact3\tHigh/Medium/Low
            Use ONLY the provided facts. Connect facts to network needs."""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Parse the response
        parts = response.split('\t')
        if len(parts) >= 3:
            relevance_text = parts[1].strip() if len(parts) > 1 else ""
            score = parts[2].strip() if len(parts) > 2 else "Medium"
            
            # Clean and format bullets
            bullets = [b.strip() for b in relevance_text.split('•') if b.strip()]
            formatted_bullets = []
            
            for bullet in bullets[:3]:
                if len(bullet) > 15:  # Only include substantial bullets
                    formatted_bullets.append(f"• {bullet}")
            
            # Ensure we have 3 bullets
            while len(formatted_bullets) < 3:
                if core_intent_analysis != "N/A" and len(formatted_bullets) == 0:
                    formatted_bullets.append("• Core strategic initiatives indicate infrastructure needs")
                elif len(formatted_bullets) == 1:
                    formatted_bullets.append("• Operations in sectors requiring network coverage")
                else:
                    formatted_bullets.append("• Company scale suggests IT infrastructure requirements")
            
            return "\n".join(formatted_bullets[:3]), score
        
        # Fallback with core intent consideration
        fallback_bullets = []
        if core_intent_analysis != "N/A":
            fallback_bullets.append("• Strategic initiatives require network infrastructure support")
        if company_data.get('expansion_news_12mo') != "N/A":
            fallback_bullets.append("• Expansion plans create immediate network deployment needs")
        if company_data.get('iot_automation_edge_integration') != "N/A":
            fallback_bullets.append("• Automation projects require reliable wireless connectivity")
        
        while len(fallback_bullets) < 3:
            fallback_bullets.append("• Target industry alignment with Syntel's focus areas")
        
        return "\n".join(fallback_bullets[:3]), "Medium"

    except Exception:
        # Basic fallback
        fallback_bullets = [
            "• Potential for network infrastructure projects",
            "• Industry sector alignment with Syntel's focus",
            "• Growth indicators suggest IT investment needs"
        ]
        return "\n".join(fallback_bullets), "Medium"

def dynamic_research_company_intelligence(company_name: str, article_url: str = None) -> Dict[str, Any]:
    """Main function to conduct comprehensive company research with hallucination prevention"""
    
    company_data = {}
    
    total_fields = len(REQUIRED_FIELDS) - 3
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Analyze core intent article with priority
    if article_url:
        status_text.info("🔍 Analyzing core intent article...")
        core_intent = analyze_core_intent_article(article_url, company_name)
        company_data["core_intent_analysis"] = core_intent
        st.write(f"Core Intent Result: {core_intent}")  # Debug output
        progress_bar.progress(20)
    else:
        company_data["core_intent_analysis"] = "N/A"
    
    # Step 2: Research other fields
    research_fields = [f for f in REQUIRED_FIELDS if f not in ["core_intent_analysis", "why_relevant_to_syntel_bullets", "intent_scoring_level"]]
    
    for i, field in enumerate(research_fields):
        progress = 20 + (i / len(research_fields)) * 60
        progress_bar.progress(int(progress))
        status_text.info(f"🔍 Researching {field.replace('_', ' ').title()}...")
        
        try:
            search_results = dynamic_search_for_field(company_name, field)
            field_data = dynamic_extract_field_with_sources(company_name, field, search_results)
            company_data[field] = field_data
            time.sleep(0.5)
            
        except Exception:
            company_data[field] = "N/A"
            continue
    
    # Step 3: Relevance analysis with core intent integration
    status_text.info("🎯 Conducting relevance analysis...")
    progress_bar.progress(85)
    
    try:
        relevance_bullets, intent_score = syntel_relevance_analysis_v2(
            company_data, company_name, company_data.get("core_intent_analysis", "N/A")
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception as e:
        company_data["why_relevant_to_syntel_bullets"] = "• Analysis based on available company data"
        company_data["intent_scoring_level"] = "Medium"
    
    progress_bar.progress(100)
    status_text.success("✅ Research complete!")
    
    return company_data

def format_clean_display(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean display format"""
    
    display_data = {
        "Company Name": company_input,
        "Branch Network / Facilities Count": data_dict.get("branch_network_count", "N/A"),
        "Expansion News (Last 12 Months)": data_dict.get("expansion_news_12mo", "N/A"),
        "Digital Transformation Initiatives": data_dict.get("digital_transformation_initiatives", "N/A"),
        "IT Infrastructure Leadership Change": data_dict.get("it_leadership_change", "N/A"),
        "Existing Network Vendors / Tech Stack": data_dict.get("existing_network_vendors", "N/A"),
        "Recent Wi-Fi Upgrade or LAN Tender Found": data_dict.get("wifi_lan_tender_found", "N/A"),
        "IoT / Automation / Edge Integration": data_dict.get("iot_automation_edge_integration", "N/A"),
        "Cloud Adoption / GCC Setup": data_dict.get("cloud_adoption_gcc_setup", "N/A"),
        "Physical Infrastructure Signals": data_dict.get("physical_infrastructure_signals", "N/A"),
        "IT Infra Budget / Capex Allocation": data_dict.get("it_infra_budget_capex", "N/A"),
        "Core Intent Analysis": data_dict.get("core_intent_analysis", "N/A"),
        "Why Relevant to Syntel": data_dict.get("why_relevant_to_syntel_bullets", "N/A"),
        "Intent Scoring": data_dict.get("intent_scoring_level", "N/A")
    }
    
    # Create DataFrame with one row
    df = pd.DataFrame([display_data])
    return df

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Company_Intel')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def create_tsv_data(df: pd.DataFrame) -> str:
    """Convert DataFrame to TSV string"""
    return df.to_csv(sep='\t', index=False)

# --- Streamlit UI ---
if __name__ == "__main__":
    st.set_page_config(page_title="Company Intelligence", layout="wide")
    st.title("🚀 Company Intelligence Generator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Configuration")
        company_name = st.text_input(
            "Company Name:", 
            value="",
            placeholder="e.g., Snowman Logistics"
        )
        
        article_url = st.text_input(
            "Core Intent Article URL:",
            value="",
            placeholder="Paste the article link that prompted this research"
        )
        
        trigger_search = st.button("🚀 Run Comprehensive Research", type="primary")

    # Initialize session state
    if 'company_name_to_search' not in st.session_state:
        st.session_state['company_name_to_search'] = None
    if 'company_data' not in st.session_state:
        st.session_state['company_data'] = None
    if 'article_url' not in st.session_state:
        st.session_state['article_url'] = None

    if trigger_search and company_name:
        st.session_state['company_name_to_search'] = company_name
        st.session_state['article_url'] = article_url
        st.session_state['company_data'] = None

    # Execute research
    if st.session_state['company_name_to_search'] and st.session_state['company_data'] is None:
        with st.spinner(f"Researching {st.session_state['company_name_to_search']}..."):
            company_data = dynamic_research_company_intelligence(
                st.session_state['company_name_to_search'], 
                st.session_state['article_url']
            )  
            st.session_state['company_data'] = company_data

    # Display results
    if 'company_data' in st.session_state and st.session_state['company_data']:
        current_company = st.session_state['company_name_to_search']
        
        with col2:
            st.subheader(f"📊 Intelligence Report: {current_company}")
            
            # Display clean dataframe
            df_display = format_clean_display(current_company, st.session_state['company_data'])
            st.dataframe(df_display, use_container_width=True, height=500)
            
            # Export options
            st.subheader("📥 Export Options")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            
            with exp_col1:
                excel_data = to_excel(df_display)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"{current_company}_Intelligence.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with exp_col2:
                tsv_data = create_tsv_data(df_display)
                st.download_button(
                    label="📄 Download TSV", 
                    data=tsv_data,
                    file_name=f"{current_company}_Intelligence.tsv",
                    mime="text/tab-separated-values",
                    use_container_width=True
                )
            
            # TSV preview
            st.subheader("📋 TSV Data")
            st.code(create_tsv_data(df_display), language="text")
