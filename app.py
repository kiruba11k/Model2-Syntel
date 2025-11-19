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
            f'"{company_name}" branch network facilities locations count 2024',
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
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        if len(content) > 50:
                            all_results.append({
                                "title": result.get('title', ''),
                                "content": content[:600],
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
    
    base_prompt = f"""
    Extract CONCISE information about {field_name} for {company_name}.
    
    RESEARCH CONTEXT:
    {research_context}
    
    AVAILABLE SOURCES: {sources_text}
    
    REQUIREMENTS:
    1. Extract ONLY factual information explicitly mentioned in the research
    2. If no specific information found, output ONLY "N/A"
    3. DO NOT invent, estimate, or assume any information
    4. DO NOT include explanations or apologies
    5. Keep response under 200 characters
    6. Format: [Brief factual summary] [Sources: url1, url2]
    
    EXTRACTED INFORMATION:
    """
    
    return base_prompt

def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    
    if not search_results:
        return "N/A"
    
    # Build research context
    research_context = ""
    for i, result in enumerate(search_results[:3]):
        research_context += f"Source {i+1}: {result['content'][:300]}\n\n"
    
    # Extract actual URLs from search results
    unique_urls = []
    for result in search_results:
        url = result.get('url', '')
        if url and url not in unique_urls and 'http' in url:
            unique_urls.append(url)
    
    source_urls = unique_urls[:2]  # Use max 2 URLs
    
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context, source_urls)
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="""You are a factual research analyst. Extract ONLY information explicitly mentioned in the research.
            CRITICAL RULES:
            1. If information is not found, output ONLY "N/A"
            2. DO NOT use placeholder URLs like url1, url2
            3. DO NOT truncate text in the middle
            4. Keep response very concise (max 200 characters)
            5. Always include actual source URLs at the end"""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Validate response
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', ''] or 
            len(response) < 5 or
            'unable to access' in response.lower() or
            'cannot provide' in response.lower()):
            return "N/A"
        
        # Clean response
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure URLs are included properly
        if source_urls and "http" not in response:
            source_text = f" [Sources: {', '.join(source_urls)}]"
            response += source_text
        elif not source_urls and "http" not in response and response != "N/A":
            response += " [Sources: Not specified]"
        
        # Final validation to prevent mid-sentence cuts
        if len(response) > 250:
            # Find a clean cutoff point
            if '.' in response[:240]:
                cutoff = response[:240].rfind('.') + 1
                response = response[:cutoff]
            elif ' ' in response[:240]:
                cutoff = response[:240].rfind(' ') 
                response = response[:cutoff]
            response = response[:250]
            
        return response
            
    except Exception as e:
        return "N/A"

def analyze_core_intent_article(article_url: str, company_name: str) -> str:
    """
    Analyze the core intent article provided by the user
    """
    if not article_url:
        return "N/A"
    
    try:
        # Use Tavily to search for the specific article
        search_results = search_tool.invoke({
            "query": f'"{company_name}" "{article_url}"',
            "max_results": 3,
            "include_raw_content": True
        })
        
        article_content = ""
        actual_article_url = article_url
        
        # Extract content from search results
        if search_results and isinstance(search_results, list):
            for result in search_results:
                if isinstance(result, dict):
                    content = result.get('content', '') or result.get('snippet', '')
                    url = result.get('url', '')
                    if content and len(content) > 100:
                        article_content = content[:1500]
                        if url:
                            actual_article_url = url
                        break
        
        if not article_content:
            # If specific article not found, search for general news
            search_results = search_tool.invoke({
                "query": f'"{company_name}" recent news strategic expansion',
                "max_results": 2
            })
            
            article_content = "Recent company news: "
            for result in search_results[:2]:
                if isinstance(result, dict):
                    content = result.get('content', '') or result.get('snippet', '')
                    if content:
                        article_content += content[:500] + " "
        
        if not article_content or len(article_content) < 50:
            return "N/A"
        
        prompt = f"""
        Analyze this content about {company_name} and extract the core business intent in ONE concise sentence.
        
        CONTENT: {article_content}
        
        Output ONLY: [One sentence about strategic intent] [Source: {actual_article_url}]
        
        If no clear intent found, output: N/A
        """
        
        response = llm_groq.invoke([
            SystemMessage(content="Extract core business intent in one sentence. If unclear, output N/A."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        if not response or response == "N/A" or len(response) < 20:
            return "N/A"
            
        return response
        
    except Exception as e:
        return "N/A"

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str, core_intent_analysis: str) -> tuple:
    """
    Generates concise relevance analysis with core intent integration
    """
    
    # Prepare data context - only use fields with actual data
    context_parts = []
    for field in ["expansion_news_12mo", "iot_automation_edge_integration", "branch_network_count", "digital_transformation_initiatives"]:
        value = company_data.get(field, "N/A")
        if value != "N/A":
            # Remove source URLs for context
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            context_parts.append(f"{field}: {clean_value}")
    
    data_context = " | ".join(context_parts) if context_parts else "Limited company data available"

    relevance_prompt = f"""
    Company: {company_name}
    Data: {data_context}
    Core Intent: {core_intent_analysis}
    
    Assess relevance to Syntel (Wi-Fi/Network integration for warehouses, manufacturing, logistics).
    Output format: Company\t• Point1 • Point2 • Point3\tScore
    
    Score: High/Medium/Low
    Be specific about network/infrastructure needs.
    """

    try:
        response = llm_groq.invoke([
            SystemMessage(content="Output: CompanyName\t• Bullet1 • Bullet2 • Bullet3\tHigh/Medium/Low"),
            HumanMessage(content=relevance_prompt)
        ]).content.strip()
        
        parts = response.split('\t')
        if len(parts) >= 3:
            relevance_text = parts[1].strip()
            score = parts[2].strip()
            
            # Clean and format bullets
            bullets = [b.strip() for b in relevance_text.split('•') if b.strip()]
            formatted_bullets = "\n".join([f"• {bullet}" for bullet in bullets[:3] if len(bullet) > 10])
            
            if not formatted_bullets:
                formatted_bullets = "• Expansion requires network infrastructure\n• Operations in target sector\n• Strategic initiatives need IT support"
            
            return formatted_bullets, score
        
        return "• Strategic relevance for network infrastructure\n• Target industry alignment\n• Growth signals IT opportunities", "Medium"

    except Exception:
        return "• Strategic relevance for network infrastructure\n• Target industry alignment\n• Growth signals IT opportunities", "Medium"

def dynamic_research_company_intelligence(company_name: str, article_url: str = None) -> Dict[str, Any]:
    """Main function to conduct comprehensive company research"""
    
    company_data = {}
    
    total_fields = len(REQUIRED_FIELDS) - 3
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analyze core intent article first
    if article_url:
        status_text.info(f"🔍 Analyzing core intent article...")
        core_intent = analyze_core_intent_article(article_url, company_name)
        company_data["core_intent_analysis"] = core_intent
        progress_bar.progress(15)
    else:
        company_data["core_intent_analysis"] = "N/A"
    
    # Research other fields
    research_fields = [f for f in REQUIRED_FIELDS if f not in ["core_intent_analysis", "why_relevant_to_syntel_bullets", "intent_scoring_level"]]
    
    for i, field in enumerate(research_fields):
        progress = 15 + (i / len(research_fields)) * 70
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
    
    status_text.info("🎯 Conducting relevance analysis...")
    progress_bar.progress(90)
    
    try:
        relevance_bullets, intent_score = syntel_relevance_analysis_v2(
            company_data, company_name, company_data.get("core_intent_analysis", "N/A")
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception:
        company_data["why_relevant_to_syntel_bullets"] = "• Strategic relevance for network infrastructure"
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
            placeholder="Paste article link for core intent analysis"
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
