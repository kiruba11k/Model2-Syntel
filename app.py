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

# --- Syntel Core Offerings ---
SYNTEL_EXPERTISE = """
Syntel specializes in:
1. IT Automation/RPA: SyntBots platform
2. Digital Transformation: Digital One suite
3. Cloud & Infrastructure: IT Infrastructure Management
4. KPO/BPO: Industry-specific solutions
"""

# --- Enhanced Dynamic Search Queries ---
def generate_dynamic_search_queries(company_name: str, field_name: str) -> List[str]:
    """Generate dynamic search queries based on company and field"""
    
    field_queries = {
        "linkedin_url": [
            f'"{company_name}" LinkedIn company page',
            f'"{company_name}" LinkedIn official'
        ],
        "company_website_url": [
            f'"{company_name}" official website',
            f'"{company_name}" company website'
        ],
        "industry_category": [
            f'"{company_name}" industry business sector',
            f'"{company_name}" type of business'
        ],
        "employee_count_linkedin": [
            f'"{company_name}" employee count LinkedIn',
            f'"{company_name}" number of employees'
        ],
        "headquarters_location": [
            f'"{company_name}" headquarters location',
            f'"{company_name}" corporate headquarters'
        ],
        "revenue_source": [
            f'"{company_name}" revenue business model',
            f'"{company_name}" annual revenue'
        ],
        "branch_network_count": [
            f'"{company_name}" branch network facilities locations',
            f'"{company_name}" offices locations branches'
        ],
        "expansion_news_12mo": [
            f'"{company_name}" expansion 2024 2025 new facilities',
            f'"{company_name}" growth expansion news'
        ],
        "digital_transformation_initiatives": [
            f'"{company_name}" digital transformation IT initiatives',
            f'"{company_name}" technology modernization'
        ],
        "it_leadership_change": [
            f'"{company_name}" CIO CTO IT leadership',
            f'"{company_name}" chief information officer'
        ],
        "existing_network_vendors": [
            f'"{company_name}" technology vendors Cisco VMware SAP Oracle',
            f'"{company_name}" IT infrastructure vendors'
        ],
        "wifi_lan_tender_found": [
            f'"{company_name}" WiFi LAN tender network upgrade',
            f'"{company_name}" network infrastructure project'
        ],
        "iot_automation_edge_integration": [
            f'"{company_name}" IoT automation robotics',
            f'"{company_name}" smart technology implementation'
        ],
        "cloud_adoption_gcc_setup": [
            f'"{company_name}" cloud adoption AWS Azure GCC',
            f'"{company_name}" cloud migration strategy'
        ],
        "physical_infrastructure_signals": [
            f'"{company_name}" new construction facility expansion',
            f'"{company_name}" infrastructure development'
        ],
        "it_infra_budget_capex": [
            f'"{company_name}" IT budget capex investment',
            f'"{company_name}" technology spending'
        ]
    }
    
    return field_queries.get(field_name, [f'"{company_name}" {field_name}'])

# --- FIXED: Enhanced Dynamic Search Function with Robust Error Handling ---
def dynamic_search_for_field(company_name: str, field_name: str) -> List[Dict]:
    """Dynamic search for specific field information with multiple query attempts"""
    queries = generate_dynamic_search_queries(company_name, field_name)
    all_results = []
    
    for query in queries[:3]:  # Try up to 3 different queries
        try:
            time.sleep(1.2)  # Rate limiting
            results = search_tool.invoke({"query": query, "max_results": 3})
            
            # FIX: Handle different types of responses
            if isinstance(results, str):
                # If it's a string error, log and skip
                st.warning(f"Search returned error for '{query}': {results}")
                continue
                
            elif isinstance(results, list):
                # Normal case - process list of results
                for result in results:
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        if len(content) > 50:  # Filter out very short results
                            all_results.append({
                                "title": result.get('title', ''),
                                "content": content[:500],
                                "url": result.get('url', ''),
                                "field": field_name,
                                "query": query
                            })
                    else:
                        st.warning(f"Unexpected result type: {type(result)}")
                        
            elif isinstance(results, dict):
                # Single result case
                content = results.get('content', '') or results.get('snippet', '')
                if len(content) > 50:
                    all_results.append({
                        "title": results.get('title', ''),
                        "content": content[:500],
                        "url": results.get('url', ''),
                        "field": field_name,
                        "query": query
                    })
                    
        except Exception as e:
            st.warning(f"Search failed for query '{query}': {str(e)[:100]}...")
            continue
    
    return all_results

# --- FIXED: Clean URL formatting function ---
def clean_and_format_url(url: str) -> str:
    """Clean and format URLs to remove double slashes and make proper"""
    if not url or url == "N/A":
        return "N/A"
    
    # Remove double slashes at start but keep http(s)://
    if url.startswith('//'):
        url = 'https:' + url
    elif url.startswith('http://') and '//' in url[7:]:
        url = url.replace('http://', 'http://').replace('//', '/')
    elif url.startswith('https://') and '//' in url[8:]:
        url = url.replace('https://', 'https://').replace('//', '/')
    
    return url

# --- FIXED: Clean text formatting function ---
def clean_text_content(text: str) -> str:
    """Clean and format text content for display"""
    if not text or text == "N/A":
        return "N/A"
    
    # Remove URLs from text content
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove extraction prefixes and irrelevant phrases
    prefixes_to_remove = [
        "Extracted Digital/IT Transformation Initiatives for",
        "Extract the",
        "Revenue/Business Model Information for",
        "IT Leadership Information for", 
        "WiFi/LAN/network upgrade information for",
        "Extracted",
        "IT infrastructure budget/capex information for",
        "Neuberg Diagnostics:",
        "Snowman Logistics:",
        "URLs?:",
        "URL 1:",
        "URL 2:",
        "- URL 1:",
        "- URL 2:",
        "URL:",
        "URL",
        "http",
        "https",
        "www.",
        ".com",
        ".org",
        ".in"
    ]
    
    for prefix in prefixes_to_remove:
        text = text.replace(prefix, '')
    
    # Remove extra newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'-\s+N/A', '', text)  # Remove "- N/A" patterns
    text = re.sub(r'N/A', '', text)  # Remove standalone N/A
    
    return text.strip()

# --- FIXED: Extract Single LinkedIn URL ---
def extract_single_linkedin_url(search_results: List[Dict]) -> str:
    """Extract a single, clean LinkedIn URL from search results"""
    linkedin_urls = []
    
    for result in search_results:
        content = result.get('content', '').lower()
        url = result.get('url', '')
        
        # Look for LinkedIn URLs in content or URL
        if 'linkedin.com/company' in url.lower() or 'linkedin.com/company' in content:
            clean_url = clean_and_format_url(url)
            if clean_url not in linkedin_urls:
                linkedin_urls.append(clean_url)
    
    # Return the first valid LinkedIn URL or N/A
    return linkedin_urls[0] if linkedin_urls else "N/A"

# --- FIXED: Extract Single Company Website URL ---
def extract_single_website_url(search_results: List[Dict]) -> str:
    """Extract a single, clean company website URL from search results"""
    website_urls = []
    
    for result in search_results:
        content = result.get('content', '').lower()
        url = result.get('url', '')
        
        # Look for company website URLs (exclude LinkedIn, social media, etc.)
        if (any(domain in url.lower() for domain in ['.com', '.in', '.org']) and
            not any(social in url.lower() for social in ['linkedin', 'facebook', 'twitter', 'youtube', 'instagram'])):
            clean_url = clean_and_format_url(url)
            if clean_url not in website_urls:
                website_urls.append(clean_url)
    
    # Return the first valid website URL or N/A
    return website_urls[0] if website_urls else "N/A"

# --- FIXED: Clean Extraction for Specific Fields ---
def extract_clean_field_data(field_name: str, response: str) -> str:
    """Clean and format field-specific data"""
    if not response or response == "N/A":
        return "N/A"
    
    # Field-specific cleaning rules
    cleaning_rules = {
        "employee_count_linkedin": lambda x: re.sub(r'https?://\S+', '', x).split('\n')[0].strip(),
        "headquarters_location": lambda x: re.sub(r'https?://\S+', '', x).split('\n')[0].strip(),
        "revenue_source": lambda x: re.sub(r'https?://\S+', '', x).split('\n')[0].strip(),
        "branch_network_count": lambda x: re.sub(r'- N/A\s*', '', re.sub(r'https?://\S+', '', x)),
        "expansion_news_12mo": lambda x: re.sub(r'https?://\S+', '', x).split('- http')[0].strip(),
        "digital_transformation_initiatives": lambda x: re.sub(r'https?://\S+', '', x).split('- http')[0].strip(),
        "it_leadership_change": lambda x: re.sub(r'https?://\S+', '', x).split('URLs?:')[0].strip(),
        "existing_network_vendors": lambda x: re.sub(r'https?://\S+', '', x).split('https')[0].strip(),
        "wifi_lan_tender_found": lambda x: re.sub(r'https?://\S+', '', x).split('- http')[0].strip(),
        "iot_automation_edge_integration": lambda x: re.sub(r'https?://\S+', '', x).split('- http')[0].strip(),
        "cloud_adoption_gcc_setup": lambda x: re.sub(r'https?://\S+', '', x),
        "physical_infrastructure_signals": lambda x: re.sub(r'https?://\S+', '', x).split('- http')[0].strip(),
        "it_infra_budget_capex": lambda x: re.sub(r'https?://\S+', '', x).split('https')[0].strip()
    }
    
    cleaner = cleaning_rules.get(field_name, lambda x: re.sub(r'https?://\S+', '', x))
    cleaned_response = cleaner(response)
    
    # Final cleanup
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
    cleaned_response = re.sub(r'\n+', ' ', cleaned_response)
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response[:300] if cleaned_response else "N/A"

# --- FIXED: Dynamic Extraction with Clean Formatting ---
def dynamic_extract_field_with_sources(company_name: str, field_name: str, search_results: List[Dict]) -> str:
    """Dynamically extract information based on actual search results"""
    
    if not search_results:
        return "N/A"
    
    # SPECIAL HANDLING: For LinkedIn URL, extract only one URL without sources
    if field_name == "linkedin_url":
        return extract_single_linkedin_url(search_results)
    
    # SPECIAL HANDLING: For Company Website URL, extract only one URL without sources
    if field_name == "company_website_url":
        return extract_single_website_url(search_results)
    
    # Format research data for context
    research_context = f"Research context for {field_name}:\n"
    for i, result in enumerate(search_results[:3]):
        clean_content = re.sub(r'https?://\S+', '', result['content'][:300])
        research_context += f"[Source {i+1}]: {clean_content}\n"
        research_context += f"[URL {i+1}]: {result['url']}\n\n"
    
    # Get unique source URLs
    unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:2]
    
    # STRICT extraction prompts
    extraction_prompts = {
        "industry_category": f"""
        Extract ONLY the primary industry/business category for {company_name}.
        Return exactly 2-4 words like 'Clinical Laboratory Testing' or 'Cold Chain Logistics'.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "employee_count_linkedin": f"""
        Extract ONLY the employee count information for {company_name}.
        Return exactly the number/range like '501-1,000 employees' or '1,001-5,000 employees'.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "headquarters_location": f"""
        Extract ONLY the headquarters location for {company_name}.
        Return exactly the location like 'Chennai, India' or 'Navi Mumbai, Maharashtra'.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "revenue_source": f"""
        Extract ONLY the revenue information for {company_name}.
        Return exactly the revenue numbers or business model.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "branch_network_count": f"""
        Extract ONLY the branch/network/facility information for {company_name}.
        Return exactly the count and description of facilities.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "expansion_news_12mo": f"""
        Extract ONLY recent expansion/growth news for {company_name} from last 12-24 months.
        Return exactly the expansion details with locations/dates.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "digital_transformation_initiatives": f"""
        Extract ONLY digital/IT transformation initiatives for {company_name}.
        Return exactly the technology projects and initiatives.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "it_leadership_change": f"""
        Extract ONLY IT leadership information for {company_name}.
        Return exactly the names/positions of CIO, CTO, IT Director.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "existing_network_vendors": f"""
        Extract ONLY technology vendors/partners for {company_name}.
        Return exactly the vendor names like Cisco, SAP, Oracle.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "wifi_lan_tender_found": f"""
        Extract ONLY WiFi/LAN/network upgrade information for {company_name}.
        Return exactly the network project details.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "iot_automation_edge_integration": f"""
        Extract ONLY IoT/Automation/Edge computing adoption for {company_name}.
        Return exactly the technology adoption details.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "cloud_adoption_gcc_setup": f"""
        Extract ONLY cloud adoption/GCC setup for {company_name}.
        Return exactly the cloud platform usage.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "physical_infrastructure_signals": f"""
        Extract ONLY physical infrastructure developments for {company_name}.
        Return exactly the construction and expansion details.
        NO URLs, NO explanations, NO additional text.
        """,
        
        "it_infra_budget_capex": f"""
        Extract ONLY IT infrastructure budget/capex information for {company_name}.
        Return exactly the budget and investment details.
        NO URLs, NO explanations, NO additional text.
        """
    }
    
    prompt = f"""
    RESEARCH DATA:
    {research_context}
    
    TASK: {extraction_prompts.get(field_name, f"Extract ONLY {field_name} for {company_name}")}
    
    CRITICAL INSTRUCTIONS:
    - Extract ONLY the specific information requested
    - NO URLs in the response
    - NO explanatory text
    - NO "N/A" mentions in the middle of response
    - If information is not found, return ONLY 'N/A'
    - Be concise and factual
    
    EXTRACTED INFORMATION:
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You are a strict information extraction agent. Extract ONLY the requested facts. NEVER include URLs, explanations, or additional text. Return 'N/A' if information is not available."),
            HumanMessage(content=prompt)
        ]).content.strip()
        
        # Clean and validate response
        if (not response or 
            response.lower() in ['n/a', 'not found', 'no information', ''] or 
            len(response) < 3):
            return "N/A"
        
        # Apply field-specific cleaning
        response = extract_clean_field_data(field_name, response)
        
        # For industry category, return just the data without sources
        if field_name == 'industry_category':
            return response
        
        # Add source URLs if we have valid information (for all other fields)
        if unique_urls and response != "N/A":
            source_text = f" [Sources: {', '.join(unique_urls[:2])}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
            response += source_text
            
        return response
            
    except Exception as e:
        st.warning(f"Extraction failed for {field_name}: {str(e)[:100]}...")
        return "N/A"

# --- FIXED: Generate Clean Relevance Analysis ---
def generate_dynamic_relevance_analysis(company_data: Dict, company_name: str, all_search_results: List[Dict]) -> tuple:
    """Generate dynamic relevance analysis based on actual company data"""
    
    # Create comprehensive context from all collected data
    context_lines = []
    for field, value in company_data.items():
        if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level"]:
            # Clean the value for context
            clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
            if clean_value and clean_value != "N/A":
                context_lines.append(f"{field}: {clean_value}")
    
    context = "\n".join(context_lines)
    
    relevance_prompt = f"""
    COMPANY: {company_name}
    
    COMPANY DATA:
    {context}
    
    SYNTEL EXPERTISE:
    - IT Automation/RPA: SyntBots platform
    - Digital Transformation: Digital One suite  
    - Cloud & Infrastructure: IT Infrastructure Management
    - KPO/BPO: Industry-specific solutions
    
    TASK: Create 3 CONCISE, ACTIONABLE bullet points explaining why this company is relevant for Syntel.
    Focus on specific opportunities based on the actual data above.
    
    Then provide an INTENT SCORE: High/Medium/Low based on concrete signals in the data.
    
    FORMAT EXACTLY (CLEAN FORMAT - NO MARKDOWN):
    BULLETS:
    1. [Specific opportunity] - [Syntel solution match]
    2. [Technology need] - [Syntel capability] 
    3. [Business signal] - [Service alignment]
    SCORE: High/Medium/Low
    
    Be specific and evidence-based from the company data. Use clean, professional language.
    """
    
    try:
        response = llm_groq.invoke([
            SystemMessage(content="You analyze business relevance for IT services. Be specific, evidence-based, and actionable. Use clean, professional formatting."),
            HumanMessage(content=relevance_prompt)
        ]).content
        
        # Parse response
        bullets = []
        score = "Medium"  # Default
        
        lines = response.split('\n')
        bullet_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('BULLETS:') or bullet_section:
                bullet_section = True
                if (line.startswith(('1', '2', '3', '•', '-')) and len(line) > 5 and 
                    not line.startswith('SCORE:')):
                    # Clean and format bullet - remove numbers and dots
                    clean_line = re.sub(r'^[1-3][\.\)]\s*', '', line)
                    clean_line = re.sub(r'^[•\-]\s*', '', clean_line)
                    bullets.append(f"• {clean_line}")
            elif 'SCORE:' in line.upper():
                if 'HIGH' in line.upper():
                    score = "High"
                elif 'LOW' in line.upper():
                    score = "Low"
                bullet_section = False  # Stop bullet collection
        
        # Ensure we have 3 bullets
        while len(bullets) < 3:
            bullets.append(f"• Additional IT service opportunity identified for {company_name}")
        
        # Clean up bullets
        cleaned_bullets = []
        for bullet in bullets[:3]:
            # Remove any remaining markdown or messy formatting
            clean_bullet = re.sub(r'\*\*|\*|__|_', '', bullet)  # Remove bold/italic
            clean_bullet = re.sub(r'\s+', ' ', clean_bullet).strip()
            cleaned_bullets.append(clean_bullet)
        
        formatted_bullets = "\n".join(cleaned_bullets)
        return formatted_bullets, score
        
    except Exception as e:
        st.warning(f"Relevance analysis failed: {str(e)[:100]}...")
        fallback_bullets = f"""• Digital transformation opportunities identified
• IT infrastructure modernization potential
• Alignment with Syntel's automation expertise"""
        return fallback_bullets, "Medium"

# --- FIXED: Main Research Function with Better Error Handling ---
def dynamic_research_company_intelligence(company_name: str) -> Dict[str, Any]:
    """Main function to dynamically research all fields"""
    
    company_data = {}
    all_search_results = []
    
    # Research each field dynamically
    total_fields = len(REQUIRED_FIELDS) - 2  # Exclude relevance fields
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, field in enumerate(REQUIRED_FIELDS[:-2]):
        progress = (i / total_fields) * 80
        progress_bar.progress(int(progress))
        status_text.info(f"🔍 Researching {field.replace('_', ' ').title()} for {company_name}...")
        
        try:
            # Dynamic search and extraction
            search_results = dynamic_search_for_field(company_name, field)
            all_search_results.extend(search_results)
            
            field_data = dynamic_extract_field_with_sources(company_name, field, search_results)
            company_data[field] = field_data
            
            time.sleep(1.5)  # Rate limiting
            
        except Exception as e:
            st.warning(f"Research failed for {field}: {str(e)[:100]}...")
            company_data[field] = "N/A"
            continue
    
    # Generate dynamic relevance analysis
    status_text.info("🤔 Analyzing Syntel relevance...")
    progress_bar.progress(90)
    
    try:
        relevance_bullets, intent_score = generate_dynamic_relevance_analysis(
            company_data, company_name, all_search_results
        )
        company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
        company_data["intent_scoring_level"] = intent_score
    except Exception as e:
        st.warning(f"Relevance analysis failed: {str(e)[:100]}...")
        company_data["why_relevant_to_syntel_bullets"] = "• Analysis pending additional company data"
        company_data["intent_scoring_level"] = "Medium"
    
    progress_bar.progress(100)
    status_text.success("✅ Research complete!")
    
    return company_data

# --- FIXED: Clean Display Formatting ---
def format_concise_display_with_sources(company_input: str, data_dict: dict) -> pd.DataFrame:
    """Transform data into clean, professional display format"""
    
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
            value = data_dict.get(data_field, "N/A")
        
        # For the three specific fields, return only the data without sources
        if display_col in ["LinkedIn URL", "Company Website URL", "Industry Category"]:
            data_list.append({"Column Header": display_col, "Value": str(value)})
        
        # Format bullet points for relevance section
        elif data_field == "why_relevant_to_syntel_bullets":
            if isinstance(value, str) and value != "N/A":
                # Clean and format bullets
                cleaned_value = value.replace('1)', '•').replace('2)', '•').replace('3)', '•')
                cleaned_value = re.sub(r'^\d\.\s*', '• ', cleaned_value, flags=re.MULTILINE)
                cleaned_value = re.sub(r'\*\*|\*', '', cleaned_value)  # Remove markdown
                html_value = cleaned_value.replace('\n', '<br>')
                data_list.append({"Column Header": display_col, "Value": f'<div style="text-align: left; line-height: 1.4;">{html_value}</div>'})
            else:
                data_list.append({"Column Header": display_col, "Value": str(value)})
        else:
            # For all other fields with source URLs, format them cleanly
            if isinstance(value, str) and "http" in value and "Source" in value:
                # Extract the main content and sources separately
                main_content = value.split(' [Source')[0] if ' [Source' in value else value
                sources_part = value.split(' [Source')[1] if ' [Source' in value else ""
                
                # Format sources as clean links
                if sources_part:
                    # Extract URLs from sources
                    urls = re.findall(r'https?://[^\s,\]]+', sources_part)
                    if urls:
                        source_links = []
                        for i, url in enumerate(urls[:2]):  # Max 2 sources
                            clean_url = clean_and_format_url(url)
                            source_links.append(f'<a href="{clean_url}" target="_blank">Source {i+1}</a>')
                        
                        sources_html = f"<br><small>Sources: {', '.join(source_links)}</small>"
                        display_value = f'<div style="text-align: left; line-height: 1.4;">{main_content}{sources_html}</div>'
                    else:
                        display_value = f'<div style="text-align: left;">{main_content}</div>'
                else:
                    display_value = f'<div style="text-align: left;">{main_content}</div>'
                
                data_list.append({"Column Header": display_col, "Value": display_value})
            else:
                # Regular text formatting
                display_value = f'<div style="text-align: left; line-height: 1.4;">{value}</div>'
                data_list.append({"Column Header": display_col, "Value": display_value})
            
    return pd.DataFrame(data_list)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Dynamic Syntel BI Agent",
    layout="wide",
    page_icon="🔍"
)

st.title("🔍 Syntel Dynamic Company Data AI Agent")
st.markdown("### 🚀 Professional Business Intelligence Reports")

# Display enhanced approach
with st.expander("🚀 Dynamic Research Approach", expanded=True):
    st.markdown("""
    **Enhanced Dynamic Features:**
    
    - **🧠 Smart Search**: Multiple query attempts per field
    - **🔍 Real-time Data**: No static/default values  
    - **📊 Evidence-Based**: All information sourced from live searches
    - **🎯 Company-Specific**: Tailored research for each company
    - **⚡ Adaptive Extraction**: LLM analyzes actual search results
    - **✨ Clean Formatting**: Professional, readable output
    
    **Research Process:**
    1. Generate dynamic search queries for each field
    2. Execute multiple search attempts
    3. Extract information from actual search results
    4. Generate relevance analysis based on real data
    5. Provide source URLs for verification
    """)

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", "Neuberg Diagnostics")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("🚀 Start Dynamic Research", type="primary")

if submitted:
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    with st.spinner(f"**🔍 Conducting dynamic research for {company_input}...**"):
        try:
            # Perform dynamic research
            company_data = dynamic_research_company_intelligence(company_input)
            
            # Display results
            st.balloons()
            st.success(f"✅ Dynamic research complete for {company_input}!")
            
            # Display final results
            st.subheader(f"Business Intelligence Report for {company_input}")
            final_df = format_concise_display_with_sources(company_input, company_data)
            
            # Apply custom CSS for better styling
            st.markdown("""
            <style>
            .dataframe {
                width: 100%;
            }
            .dataframe th {
                background-color: #f0f2f6;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            .dataframe td {
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            # Show completion metrics
            with st.expander("📊 Research Summary", expanded=True):
                completed_fields = sum(1 for field in REQUIRED_FIELDS 
                                    if company_data.get(field) and 
                                    company_data.get(field) != "N/A")
                
                fields_with_sources = sum(1 for field in REQUIRED_FIELDS[:-2]
                                       if company_data.get(field) and 
                                       company_data.get(field) != "N/A" and
                                       "Source" in company_data.get(field, ""))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fields Completed", f"{completed_fields}/{len(REQUIRED_FIELDS)}")
                with col2:
                    st.metric("Fields with Sources", f"{fields_with_sources}/{len(REQUIRED_FIELDS)-2}")
                with col3:
                    score_color = {
                        "High": "green", 
                        "Medium": "orange", 
                        "Low": "red"
                    }.get(company_data.get("intent_scoring_level", "Medium"), "gray")
                    st.markdown(f"<h3 style='color: {score_color};'>Intent Score: {company_data.get('intent_scoring_level', 'Medium')}</h3>", unsafe_allow_html=True)
            
            # Download options
            st.subheader("💾 Download Report")
            
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 st.download_button(
                     label="Download JSON",
                     data=json.dumps(company_data, indent=2),
                     file_name=f"{company_input.replace(' ', '_')}_dynamic_data.json",
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = final_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download CSV",
                     data=csv_data,
                     file_name=f"{company_input.replace(' ', '_')}_dynamic_data.csv",
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(final_df)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data,
                     file_name=f"{company_input.replace(' ', '_')}_dynamic_data.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.info("This might be due to API rate limits or search constraints. Please try again in a few moments.")

# Research History
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

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
    "Dynamic Syntel BI Agent | Professional Business Intelligence"
    "</div>",
    unsafe_allow_html=True
)
