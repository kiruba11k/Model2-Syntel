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
    
    for query in queries[:2]:  # Reduced to 2 queries per field
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
                                "content": content[:600],  # Reduced content length
                                "url": result.get('url', ''),
                                "field": field_name,
                                "query": query
                            })
        except Exception:
            continue
    return all_results

def get_detailed_extraction_prompt(company_name: str, field_name: str, research_context: str) -> str:
    
    prompts = {
        "branch_network_count": f"""
        Extract CONCISE summary about branch network/facilities for {company_name}. 
        Include: total facilities count, key locations, pallet capacity.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "expansion_news_12mo": f"""
        Extract CONCISE recent expansion news for {company_name} (last 12 months).
        Include: locations, capacities, timelines.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "digital_transformation_initiatives": f"""
        Extract CONCISE digital transformation projects for {company_name}.
        Include: specific systems/platforms implemented.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "it_leadership_change": f"""
        Extract CONCISE IT leadership changes for {company_name}.
        Include: names, positions, change dates if available.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "existing_network_vendors": f"""
        Extract CONCISE network vendors/tech stack for {company_name}.
        Include: specific vendor names mentioned.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "wifi_lan_tender_found": f"""
        Extract CONCISE WiFi/LAN tender info for {company_name}.
        Include: specific projects, upgrade details.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "iot_automation_edge_integration": f"""
        Extract CONCISE IoT/Automation implementations for {company_name}.
        Include: specific technologies, use cases.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "cloud_adoption_gcc_setup": f"""
        Extract CONCISE cloud adoption/GCC setup for {company_name}.
        Include: cloud providers, GCC plans if mentioned.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "physical_infrastructure_signals": f"""
        Extract CONCISE physical infrastructure developments for {company_name}.
        Include: new facilities, locations, key features.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """,
        
        "it_infra_budget_capex": f"""
        Extract CONCISE IT budget/capex info for {company_name}.
        Include: budget figures, investment focus areas.
        RESEARCH: {research_context}
        FORMAT: [Summary] [Sources: url1, url2]
        OUTPUT:
        """
    }
    
    return prompts.get(field_name, f"""
    Extract CONCISE information about {field_name} for {company_name}.
    RESEARCH: {research_context}
    FORMAT: [Summary] [Sources: url1, url2]
    OUTPUT:
    """)

def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    
    if not search_results:
        return "N/A"
    
    # Build concise research context
    research_context = f"Research for {company_name}:\n"
    for i, result in enumerate(search_results[:3]):  # Use only top 3 results
        research_context += f"Source {i+1}: {result['content'][:300]}\n"
    
    unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:2]
    prompt = get_detailed_extraction_prompt(company_name, field_name, research_context)
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content=f"""You are a concise research analyst. Extract ONLY key facts in 1-2 sentences maximum.
            **CRITICAL: Format as: [Brief Summary] [Sources: url1, url2]**
            Be extremely concise and factual. No explanations."""),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', ''] or 
            len(response) < 10):
            return "N/A"
        
        # Ensure sources are included
        if unique_urls and "http" not in response:
            source_text = f" [Sources: {', '.join(unique_urls)}]"
            response += source_text
        elif "http" not in response:
            response += " [Sources: Not specified]"
            
        return response[:400]  # Strict length limit
            
    except Exception as e:
        return "N/A"

def analyze_core_intent_article(article_url: str, company_name: str) -> str:
    """
    Analyze the core intent article provided by the user
    """
    if not article_url or article_url == "N/A":
        return "N/A - No article URL provided"
    
    try:
        # Get content from the specific URL
        search_results = search_tool.invoke({
            "query": f"site:{article_url}",
            "max_results": 1,
            "include_raw_content": True
        })
        
        article_content = ""
        if search_results and isinstance(search_results, list):
            for result in search_results:
                if isinstance(result, dict):
                    content = result.get('content', '') or result.get('snippet', '')
                    if content:
                        article_content = content[:1000]
                        break
        
        prompt = f"""
        Analyze this article about {company_name} and extract core business intent in ONE sentence.
        ARTICLE: {article_content}
        Output: [One-sentence analysis] [Article: {article_url}]
        """
        
        response = llm_groq.invoke([
            SystemMessage(content="You are a strategic analyst. Extract the core business intent in one concise sentence."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        return response if response else f"N/A - Could not analyze [Article: {article_url}]"
        
    except Exception as e:
        return f"N/A - Error analyzing article [URL: {article_url}]"

def syntel_relevance_analysis_v2(company_data: Dict, company_name: str, core_intent_analysis: str) -> tuple:
    """
    Generates concise relevance analysis with core intent integration
    """
    
    # Prepare concise data context
    context_lines = []
    key_fields = ["expansion_news_12mo", "iot_automation_edge_integration", "digital_transformation_initiatives", "branch_network_count"]
    
    for field in key_fields:
        value = company_data.get(field, "N/A")
        if value != "N/A":
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            context_lines.append(f"{field}: {clean_value[:100]}")

    data_context = "\n".join(context_lines)

    relevance_prompt = f"""
    Analyze relevance to Syntel's Wi-Fi & Network Integration focus.
    
    COMPANY DATA: {data_context}
    CORE INTENT: {core_intent_analysis}
    
    Output format: Company Name\tWhy Relevant to Syntel\tIntent Score
    
    Why Relevant: 3 VERY SHORT bullet points (• Bullet1 • Bullet2 • Bullet3)
    Intent Score: High/Medium/Low
    """

    try:
        response = llm_groq.invoke([
            SystemMessage(content="Output ONLY in this exact format: Company Name\t• Point1 • Point2 • Point3\tHigh/Medium/Low"),
            HumanMessage(content=relevance_prompt)
        ]).content.strip()
        
        parts = response.split('\t')
        if len(parts) == 3:
            company, relevance_text, score = parts
            
            # Clean and format bullets
            bullets = [bullet.strip() for bullet in relevance_text.split('•') if bullet.strip()]
            formatted_bullets = "\n".join([f"• {bullet}" for bullet in bullets[:3]])
            
            return formatted_bullets, score.strip()
        
        # Fallback
        fallback_bullets = [
            "• Expansion plans require network infrastructure",
            "• Operations align with Syntel's target sectors", 
            "• Core intent indicates technology needs"
        ]
        return "\n".join(fallback_bullets), "Medium"

    except Exception:
        fallback_bullets = [
            "• Strategic initiatives need network support",
            "• Company in target industry segment",
            "• Growth signals infrastructure opportunities"
        ]
        return "\n".join(fallback_bullets), "Medium"

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
        company_data["core_intent_analysis"] = "N/A - No article URL provided"
    
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
            
            time.sleep(0.5)  # Reduced delay
            
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

def format_concise_display(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean, concise display format"""
    
    # Define the display structure with concise summaries
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
            placeholder="Paste article link that prompted this research"
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
            
            # Display concise dataframe
            df_display = format_concise_display(current_company, st.session_state['company_data'])
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
            
            with exp_col3:
                st.download_button(
                    label="📋 Copy TSV Data",
                    data=create_tsv_data(df_display),
                    file_name=f"{current_company}_data.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # TSV preview
            st.subheader("📋 TSV Preview")
            st.code(create_tsv_data(df_display), language="text")
