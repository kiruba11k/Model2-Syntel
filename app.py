import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any, Tuple
from io import BytesIO
from datetime import datetime
import time
import logging

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration & Setup ---
class Config:
    """Configuration management"""
    REQUIRED_FIELDS = [
        "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
        "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
        "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
        "physical_infrastructure_signals", "it_infra_budget_capex", "core_intent_analysis",
        "why_relevant_to_syntel_bullets", "intent_scoring_level"
    ]
    
    FIELD_DISPLAY_NAMES = {
        "branch_network_count": "Branch Network / Facilities Count",
        "expansion_news_12mo": "Expansion News (Last 12 Months)",
        "digital_transformation_initiatives": "Digital Transformation Initiatives",
        "it_leadership_change": "IT Infrastructure Leadership Change",
        "existing_network_vendors": "Existing Network Vendors / Tech Stack",
        "wifi_lan_tender_found": "Recent Wi-Fi Upgrade or LAN Tender Found",
        "iot_automation_edge_integration": "IoT / Automation / Edge Integration Mentioned",
        "cloud_adoption_gcc_setup": "Cloud Adoption / GCC Setup",
        "physical_infrastructure_signals": "Physical Infrastructure Signals",
        "it_infra_budget_capex": "IT Infra Budget / Capex Allocation",
        "core_intent_analysis": "Core Intent Analysis",
        "why_relevant_to_syntel_bullets": "Why Relevant to Syntel",
        "intent_scoring_level": "Intent Scoring"
    }

# --- Utility Functions ---
def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

logger = setup_logging()

def clean_and_format_url(url: str) -> str:
    """Clean and format URLs"""
    if not url or url == "N/A":
        return "N/A"
    
    if url.startswith('//'):
        url = 'https:' + url
    elif not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    return url.replace(' ', '').strip()

def validate_company_name(company_name: str) -> Tuple[bool, str]:
    """Validate company name input"""
    if not company_name or len(company_name.strip()) < 2:
        return False, "Company name must be at least 2 characters long"
    
    if len(company_name.strip()) > 100:
        return False, "Company name too long (max 100 characters)"
    
    # Basic sanitization check
    if re.search(r'[<>{}[\]]', company_name):
        return False, "Company name contains invalid characters"
    
    return True, ""

# --- LLM Service Class ---
class LLMService:
    """LLM service management"""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.initialize_services()
    
    def initialize_services(self):
        """Initialize LLM and search services"""
        try:
            TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
            GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

            if not GROQ_API_KEY or not TAVILY_API_KEY:
                st.error("ERROR: Both GROQ_API_KEY and TAVILY_API_KEY must be set in Streamlit secrets.")
                st.stop()

            self.llm = ChatGroq(
                model="llama-3.1-8b-instant", 
                groq_api_key=GROQ_API_KEY,
                temperature=0
            )
            self.search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
            
            st.success("✅ Services initialized: Groq (Llama 3.1 8B) + Tavily Search")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize services: {e}")
            logger.error(f"Service initialization failed: {e}")
            st.stop()

# --- Search Service Class ---
class SearchService:
    """Search service management"""
    
    def __init__(self, search_tool):
        self.search_tool = search_tool
        self.search_queries = {
            "branch_network_count": [
                '"{company}" branch network facilities locations count',
                '"{company}" warehouse facility count pallet capacity 2024 2025',
                '"{company}" manufacturing plants offices locations'
            ],
            "expansion_news_12mo": [
                '"{company}" expansion news 2024 2025 new facilities',
                '"{company}" new warehouse construction Q3 Q4 2024 2025',
                '"{company}" investment expansion project 2024'
            ],
            "digital_transformation_initiatives": [
                '"{company}" digital transformation IT initiatives 2024',
                '"{company}" Industry 4.0 technology adoption'
            ],
            "it_leadership_change": [
                '"{company}" CIO CTO IT infrastructure leadership 2024',
                '"{company}" IT leadership team executives'
            ],
            "existing_network_vendors": [
                '"{company}" network infrastructure vendors Cisco HPE Aruba',
                '"{company}" IT technology stack network equipment',
                '"{company}" technology partners vendors'
            ],
            "wifi_lan_tender_found": [
                '"{company}" WiFi LAN tender network upgrade 2024',
                '"{company}" wireless network infrastructure project',
                '"{company}" network modernization initiative'
            ],
            "iot_automation_edge_integration": [
                '"{company}" IoT automation robotics implementation',
                '"{company}" smart manufacturing Industry 4.0',
                '"{company}" automation technology projects'
            ],
            "cloud_adoption_gcc_setup": [
                '"{company}" cloud adoption AWS Azure GCC setup',
                '"{company}" cloud migration digital infrastructure'
            ],
            "physical_infrastructure_signals": [
                '"{company}" new construction facility expansion',
                '"{company}" infrastructure development projects'
            ],
            "it_infra_budget_capex": [
                '"{company}" IT budget capex investment technology spending',
                '"{company}" capital expenditure technology infrastructure'
            ]
        }
    
    def generate_search_queries(self, company_name: str, field_name: str) -> List[str]:
        """Generate search queries for a specific field"""
        queries = self.search_queries.get(field_name, [f'"{company_name}" {field_name}'])
        return [query.format(company=company_name) for query in queries[:3]]
    
    def search_for_field(self, company_name: str, field_name: str) -> List[Dict]:
        """Search for information about a specific field"""
        queries = self.generate_search_queries(company_name, field_name)
        all_results = []
        
        for query in queries:
            try:
                time.sleep(1.0)  # Increased rate limiting for reliability
                results = self.search_tool.invoke({"query": query, "max_results": 4})
                
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            content = result.get('content', '') or result.get('snippet', '')
                            if len(content) > 30:  # Reduced minimum content length
                                all_results.append({
                                    "title": result.get('title', 'No Title'),
                                    "content": content[:1000],  # Increased content length
                                    "url": clean_and_format_url(result.get('url', '')),
                                    "field": field_name,
                                    "query": query
                                })
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        return all_results

# --- Data Extraction Class ---
class DataExtractor:
    """Data extraction and processing"""
    
    def __init__(self, llm_service):
        self.llm = llm_service
    
    def get_extraction_prompt(self, company_name: str, field_name: str, research_context: str) -> str:
        """Get the appropriate extraction prompt for a field"""
        
        prompts = {
            "branch_network_count": f"""
            Extract ONLY factual numbers and counts about {company_name}'s facilities, branches, plants, offices, or locations.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY numbers explicitly mentioned (e.g., "15 offices", "12 plants", "17 facilities")
            - Include location details if provided with numbers
            - DO NOT calculate, estimate, or infer any numbers
            - If no specific counts found, return exactly "N/A"
            - Format as bullet points if multiple numbers found
            - Start directly with the extracted numbers
            
            EXTRACTED DATA:
            """,
            
            "expansion_news_12mo": f"""
            Extract ONLY specific expansion announcements, investments, or new projects for {company_name} from the last 12-24 months.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicit expansion news with details
            - Include investment amounts, locations, dates if mentioned
            - Focus on recent announcements (2023-2025)
            - DO NOT infer expansions
            - If no expansion news found, return exactly "N/A"
            - Format as numbered list if multiple expansions
            - Start directly with the extracted news
            
            EXTRACTED EXPANSION NEWS:
            """,
            
            "digital_transformation_initiatives": f"""
            Extract ONLY specific digital transformation projects, technologies, or initiatives for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicitly mentioned technologies or projects
            - Include specific system names if provided (e.g., SAP, IoT, cloud platforms)
            - Focus on recent implementations (2023-2025)
            - DO NOT infer digital initiatives
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple initiatives
            - Start directly with the extracted initiatives
            
            EXTRACTED DIGITAL INITIATIVES:
            """,
            
            "it_leadership_change": f"""
            Extract ONLY specific IT leadership changes, appointments, or executive team members for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicitly mentioned leadership changes or appointments
            - Include names and positions if provided
            - Focus on CIO, CTO, IT Head, Infrastructure leadership
            - DO NOT infer leadership changes
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple changes
            - Start directly with the extracted information
            
            EXTRACTED LEADERSHIP INFORMATION:
            """,
            
            "existing_network_vendors": f"""
            Extract ONLY specific technology vendors, partners, or platforms mentioned for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicitly mentioned vendor names
            - Include technology platforms if specified
            - Focus on network, infrastructure, cloud vendors
            - DO NOT infer vendors based on industry
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple vendors
            - Start directly with the extracted vendors
            
            EXTRACTED VENDORS:
            """,
            
            "wifi_lan_tender_found": f"""
            Extract ONLY specific information about WiFi, LAN, network upgrades, or technology tenders for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicit mentions of network upgrades, WiFi projects, LAN tenders
            - Include specific project details if provided
            - Focus on recent projects (2023-2025)
            - DO NOT assume network upgrades
            - If none found, return exactly "N/A"
            - Start directly with the extracted information
            
            EXTRACTED NETWORK PROJECTS:
            """,
            
            "iot_automation_edge_integration": f"""
            Extract ONLY specific IoT, automation, robotics, or smart technology projects for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicit mentions of IoT, automation, robotics, smart technologies
            - Include specific project details if provided
            - Focus on manufacturing, logistics, operational technologies
            - DO NOT infer IoT usage
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple projects
            - Start directly with the extracted projects
            
            EXTRACTED IOT/AUTOMATION PROJECTS:
            """,
            
            "cloud_adoption_gcc_setup": f"""
            Extract ONLY specific cloud adoption, GCC setup, or cloud migration details for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicit mentions of cloud platforms, GCC, cloud migration
            - Include specific cloud providers if mentioned (AWS, Azure, etc.)
            - Focus on recent cloud initiatives
            - DO NOT assume cloud adoption
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple details
            - Start directly with the extracted information
            
            EXTRACTED CLOUD/GCC INFORMATION:
            """,
            
            "physical_infrastructure_signals": f"""
            Extract ONLY specific physical infrastructure developments, construction, or facility projects for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicit mentions of construction, new facilities, infrastructure projects
            - Include locations and investment details if provided
            - Focus on recent developments (2023-2025)
            - DO NOT infer infrastructure projects
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple projects
            - Start directly with the extracted projects
            
            EXTRACTED INFRASTRUCTURE PROJECTS:
            """,
            
            "it_infra_budget_capex": f"""
            Extract ONLY specific IT infrastructure budget, capex, or technology investment numbers for {company_name}.
            
            RESEARCH DATA: {research_context}
            
            CRITICAL RULES:
            - Extract ONLY explicit budget figures, capex amounts, investment numbers
            - Include timeframes if provided (e.g., FY2024, FY2025)
            - Focus on IT, technology, digital infrastructure spending
            - DO NOT estimate or calculate budgets
            - If none found, return exactly "N/A"
            - Format as numbered list if multiple figures
            - Start directly with the extracted numbers
            
            EXTRACTED BUDGET INFORMATION:
            """
        }
        
        return prompts.get(field_name, f"""
        Extract factual information about {field_name} for {company_name}.
        
        RESEARCH DATA: {research_context}
        
        RULES:
        - Extract ONLY explicit information from the research
        - Be concise and factual
        - Return exactly "N/A" if no information found
        - Start directly with the extracted data
        
        EXTRACTED INFORMATION:
        """)
    
    def extract_field_data(self, company_name: str, field_name: str, search_results: List[Dict]) -> str:
        """Extract field data from search results"""
        
        if not search_results:
            return "N/A"
        
        # Build research context with better source tracking
        research_context = f"Research data for {company_name} - {field_name}:\n\n"
        source_mapping = {}
        
        for i, result in enumerate(search_results[:6]):  # Increased from 4 to 6
            source_key = f"Source {i+1}"
            source_mapping[source_key] = result['url']
            research_context += f"{source_key} - {result.get('title', 'No Title')}:\n"
            research_context += f"CONTENT: {result['content']}\n\n" 
        
        unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:3]
        
        try:
            prompt = self.get_extraction_prompt(company_name, field_name, research_context)
            
            response = self.llm.invoke([
                SystemMessage(content="""You are a precise data extraction assistant. 
                Extract ONLY information explicitly stated in the research data. 
                If information is not found, respond with exactly "N/A".
                Do not use any prior knowledge or make assumptions.
                Be factual and concise. Start your response directly with the extracted data."""),
                HumanMessage(content=prompt)
            ]).content.strip()
            
            # Enhanced validation for hallucination prevention
            if self._is_invalid_response(response):
                return "N/A"
            
            # Clean response more aggressively
            response = self._clean_response(response)
            
            # Add sources if data found - ensure URLs are preserved
            if response != "N/A" and unique_urls:
                # Use shorter source notation to save space
                if len(unique_urls) == 1:
                    source_text = f" [Source: {unique_urls[0]}]"
                else:
                    # Truncate long URLs but keep them functional
                    shortened_urls = []
                    for url in unique_urls[:2]:
                        if len(url) > 50:
                            # Keep the domain and key parts
                            domain = re.findall(r'https?://([^/]+)', url)
                            if domain:
                                shortened_url = f"{domain[0]}/..."
                                shortened_urls.append(shortened_url)
                            else:
                                shortened_urls.append(url[:50] + "...")
                        else:
                            shortened_urls.append(url)
                    source_text = f" [Sources: {', '.join(shortened_urls)}]"
                
                response += source_text
            
            return response[:800]  # Increased limit to preserve sources
            
        except Exception as e:
            logger.error(f"Extraction failed for {field_name}: {e}")
            return "N/A"
    
    def _is_invalid_response(self, response: str) -> bool:
        """Check if response is invalid or hallucinated"""
        if not response or len(response) < 5:
            return True
        
        invalid_indicators = [
            'n/a', 'not found', 'no information', 
            'information not available', 'not mentioned',
            'no specific', 'unable to find', 'could not find',
            'based on my knowledge', 'i cannot', 'i don\'t know',
            'the research data does not', 'the provided research'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in invalid_indicators)
    
    def _clean_response(self, response: str) -> str:
        """Clean up response text while preserving structure"""
        # Remove introductory phrases more aggressively
        clean_up_phrases = [
            r'^\s*Based on (the|this|the provided).*?[:\-]\s*',
            r'^\s*Here is.*?[:\-]\s*',
            r'^\s*Extracted.*?[:\-]\s*',
            r'^\s*The.*?for.*?is[:\-]\s*',
            r'^\s*According to.*?[:\-]\s*',
            r'^\s*From the research.*?[:\-]\s*',
        ]
        
        for phrase in clean_up_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up bullet points and numbering
        response = re.sub(r'^\s*[\d•\-]\s*', '', response, flags=re.MULTILINE)
        
        # Final cleaning - preserve newlines for readability
        response = re.sub(r'\n\s*\n', '\n', response)  # Reduce multiple newlines
        response = re.sub(r'[ \t]+', ' ', response)     # Clean up spaces
        response = response.replace("**", "").replace("*", "")
        response = response.strip()
        
        return response

# --- Core Intent Analysis ---
class CoreIntentAnalyzer:
    """Core intent analysis functionality"""
    
    def __init__(self, llm_service, search_tool):
        self.llm = llm_service
        self.search_tool = search_tool
    
    def analyze_article(self, article_url: str, company_name: str) -> str:
        """Analyze core intent from article"""
        if not article_url or article_url == "N/A":
            return "N/A - No article URL provided"
        
        try:
            # Enhanced article content extraction
            article_content = ""
            
            # Try multiple approaches to get content
            try:
                # Approach 1: Direct URL search
                search_results = self.search_tool.invoke({
                    "query": f'"{company_name}" "{article_url}"',
                    "max_results": 2
                })
                
                if search_results and isinstance(search_results, list):
                    for result in search_results:
                        content = result.get('content', '') or result.get('snippet', '')
                        if content and len(content) > 100:
                            article_content = content[:2000]
                            break
            except:
                pass
            
            # Approach 2: General company news as fallback
            if not article_content:
                try:
                    search_results = self.search_tool.invoke({
                        "query": f'"{company_name}" recent news 2024 2025',
                        "max_results": 3
                    })
                    
                    article_content = f"Recent news about {company_name}:\n\n"
                    for i, result in enumerate(search_results[:2]):
                        if isinstance(result, dict):
                            content = result.get('content', '') or result.get('snippet', '')
                            if content:
                                article_content += f"News {i+1}: {content}\n\n"
                except:
                    pass
            
            if not article_content:
                return f"N/A - Could not retrieve article content [URL: {article_url}]"
            
            # Enhanced analysis prompt
            prompt = f"""
            Analyze the strategic business intent from this content about {company_name}:
            
            CONTENT: {article_content}
            
            Provide a CONCISE analysis focusing on:
            
            1. **Core Business Objective**: What is the main strategic move or goal?
            2. **Technology Implications**: What infrastructure/technology needs does this create?
            3. **Network Requirements**: What does this mean for network/WiFi infrastructure?
            
            Be specific and actionable. Focus on what the company is actually doing, not general industry trends.
            """
            
            response = self.llm.invoke([
                SystemMessage(content="You are a strategic business analyst. Provide concise, actionable analysis focusing on specific business moves and their technology implications."),
                HumanMessage(content=prompt)
            ]).content.strip()
            
            if response and len(response) > 50:
                return f"{response} [Article: {article_url}]"
            else:
                return f"N/A - Insufficient analysis generated [URL: {article_url}]"
            
        except Exception as e:
            logger.error(f"Core intent analysis failed: {e}")
            return f"N/A - Analysis error: {str(e)} [URL: {article_url}]"

# --- Relevance Analysis ---
class RelevanceAnalyzer:
    """Strategic relevance analysis"""
    
    def __init__(self, llm_service):
        self.llm = llm_service
    
    def analyze_relevance(self, company_data: Dict, company_name: str, core_intent: str) -> Tuple[str, str]:
        """Analyze relevance to Syntel and generate intent score"""
        
        # Prepare data context with better formatting
        context_lines = []
        for field, value in company_data.items():
            if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level", "core_intent_analysis"]:
                # Clean value but preserve key information
                clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
                if clean_value and len(clean_value) > 10:  # Only include substantial data
                    context_lines.append(f"• {field.replace('_', ' ').title()}: {clean_value}")
        
        data_context = "\n".join(context_lines) if context_lines else "• Limited specific data available"

        # Enhanced prompt with better formatting
        prompt = f"""
        Analyze {company_name} for Syntel's Wi-Fi & Network Integration services.

        **SYNTEL TARGET PROFILE:**
        - Industries: Manufacturing, Warehousing, Logistics, Healthcare, Education, IT/ITES
        - Key Signals: Expansion, Digital Transformation, Network Upgrades, IoT/Automation
        - Services: WiFi deployment, Network integration, Multi-vendor implementation

        **COMPANY DATA:**
        {data_context}

        **CORE STRATEGIC INTENT:**
        {core_intent}

        **TASK:**
        1. Generate 3 SPECIFIC bullet points "Why Relevant to Syntel"
        2. Assign Intent Score: High/Medium/Low
        3. Focus on concrete opportunities, not generalities

        **OUTPUT FORMAT (TSV - Tab Separated):**
        Company<TAB>Bullet Points<TAB>Score

        **BULLET POINT RULES:**
        - Start each with "• " 
        - Separate bullets with newline (\\n)
        - Be specific about opportunities
        - No markdown, no excessive formatting
        - Maximum 2 lines per bullet

        **SCORING CRITERIA:**
        - High: Clear expansion + technology signals
        - Medium: Some relevant signals present  
        - Low: Minimal relevant signals
        """

        try:
            response = self.llm.invoke([
                SystemMessage(content="""You are a pragmatic GTM analyst. 
                Output ONLY in the exact TSV format requested. 
                Focus on specific, actionable opportunities.
                Use clear, professional language without exaggeration."""),
                HumanMessage(content=prompt)
            ]).content.strip()

            # More robust parsing
            lines = response.split('\n')
            for line in lines:
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        _, relevance_text, score = parts[0], parts[1], parts[2]
                        
                        # Clean and format bullets
                        bullets = []
                        for bullet_line in relevance_text.split('\n'):
                            clean_bullet = re.sub(r'^[•\-\s]*', '• ', bullet_line.strip())
                            clean_bullet = re.sub(r'\*\*|\*|__|_', '', clean_bullet)
                            if clean_bullet and len(clean_bullet) > 10:  # Minimum meaningful length
                                bullets.append(clean_bullet)
                        
                        # Ensure we have exactly 3 quality bullets
                        while len(bullets) < 3:
                            new_bullet = self._get_fallback_bullet(len(bullets), company_data, core_intent)
                            if new_bullet not in bullets:  # Avoid duplicates
                                bullets.append(new_bullet)
                        
                        formatted_bullets = "\n".join(bullets[:3])
                        clean_score = self._clean_score(score)
                        return formatted_bullets, clean_score

            # If TSV parsing fails, try other formats
            return self._parse_alternative_format(response, company_data, core_intent)

        except Exception as e:
            logger.error(f"Relevance analysis failed: {e}")
            return self._get_fallback_analysis(company_data, core_intent)

    def _clean_score(self, score: str) -> str:
        """Clean and validate score"""
        score_clean = re.sub(r'[^a-zA-Z]', '', score.lower())
        if 'high' in score_clean:
            return "High"
        elif 'medium' in score_clean:
            return "Medium"
        elif 'low' in score_clean:
            return "Low"
        else:
            return "Medium"  # Default

    def _parse_alternative_format(self, response: str, company_data: Dict, core_intent: str) -> Tuple[str, str]:
        """Parse alternative response formats"""
        try:
            # Look for bullet points and score in various formats
            bullets = []
            lines = response.split('\n')
            score = "Medium"
            
            for line in lines:
                line_clean = line.strip()
                # Look for bullet points
                if line_clean.startswith(('•', '-', '*')) and len(line_clean) > 10:
                    clean_bullet = re.sub(r'^[•\-\*\s]*', '• ', line_clean)
                    bullets.append(clean_bullet)
                # Look for score
                elif 'high' in line_clean.lower():
                    score = "High"
                elif 'low' in line_clean.lower():
                    score = "Low"
            
            # Use what we found, fallback if needed
            if bullets:
                while len(bullets) < 3:
                    bullets.append(self._get_fallback_bullet(len(bullets), company_data, core_intent))
                return "\n".join(bullets[:3]), score
            else:
                return self._get_fallback_analysis(company_data, core_intent)
                
        except:
            return self._get_fallback_analysis(company_data, core_intent)

    def _get_fallback_bullet(self, index: int, company_data: Dict, core_intent: str) -> str:
        """Get context-aware fallback bullet points"""
        fallbacks = [
            "• Operations in target manufacturing/logistics sector align with Syntel's network expertise",
            "• Infrastructure scale indicates need for professional network deployment services", 
            "• Digital initiatives create opportunities for network modernization projects"
        ]
        
        # Customize based on available data
        if index == 0 and company_data.get('expansion_news_12mo') != "N/A":
            return "• Recent expansion activities indicate immediate network infrastructure requirements"
        elif index == 1 and company_data.get('iot_automation_edge_integration') != "N/A":
            return "• IoT/Automation projects require robust, high-performance network infrastructure"
        elif index == 2 and "N/A" not in core_intent:
            return "• Strategic initiatives suggest need for scalable network solutions to support growth"
        
        return fallbacks[index] if index < len(fallbacks) else fallbacks[-1]

    def _get_fallback_analysis(self, company_data: Dict, core_intent: str) -> Tuple[str, str]:
        """Generate comprehensive fallback analysis"""
        bullets = []
        
        # Analyze available data for best bullets
        expansion_signals = company_data.get('expansion_news_12mo') not in ["N/A", ""]
        tech_signals = company_data.get('digital_transformation_initiatives') not in ["N/A", ""]
        iot_signals = company_data.get('iot_automation_edge_integration') not in ["N/A", ""]
        intent_signals = "N/A" not in core_intent
        
        if expansion_signals:
            bullets.append("• Expansion projects create immediate opportunities for network infrastructure deployment")
        else:
            bullets.append("• Manufacturing/logistics operations require reliable wide-area network coverage")
        
        if tech_signals or iot_signals:
            bullets.append("• Digital transformation initiatives indicate need for modern network infrastructure")
        else:
            bullets.append("• Scale of operations suggests potential for network optimization and upgrades")
        
        if intent_signals:
            bullets.append("• Strategic growth plans align with Syntel's expertise in scalable network solutions")
        else:
            bullets.append("• Industry position indicates ongoing network infrastructure requirements")

        # Determine score based on signals
        strong_signals = sum([expansion_signals, tech_signals, iot_signals, intent_signals])
        if strong_signals >= 3:
            score = "High"
        elif strong_signals >= 2:
            score = "Medium" 
        else:
            score = "Low"

        return "\n".join(bullets[:3]), score

# --- Main Research Engine ---
class CompanyResearchEngine:
    """Main research engine coordinating all services"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.search_service = SearchService(self.llm_service.search_tool)
        self.data_extractor = DataExtractor(self.llm_service.llm)
        self.intent_analyzer = CoreIntentAnalyzer(self.llm_service.llm, self.llm_service.search_tool)
        self.relevance_analyzer = RelevanceAnalyzer(self.llm_service.llm)
        self.config = Config()
    
    def research_company(self, company_name: str, article_url: str = None) -> Dict[str, Any]:
        """Conduct comprehensive company research with better error handling"""
        
        company_data = {}
        research_fields = [f for f in self.config.REQUIRED_FIELDS 
                         if f not in ["core_intent_analysis", "why_relevant_to_syntel_bullets", "intent_scoring_level"]]
        
        total_steps = len(research_fields) + 2  # +2 for core intent and relevance analysis
        current_step = 0
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Core Intent Analysis
            if article_url:
                current_step += 1
                progress = int((current_step / total_steps) * 100)
                progress_bar.progress(progress)
                status_text.info("🔍 Analyzing core intent article...")
                
                company_data["core_intent_analysis"] = self.intent_analyzer.analyze_article(article_url, company_name)
            else:
                company_data["core_intent_analysis"] = "N/A - No article URL provided"
            
            # Step 2: Field Research with better error handling
            for i, field in enumerate(research_fields):
                current_step += 1
                progress = int((current_step / total_steps) * 100)
                progress_bar.progress(progress)
                status_text.info(f"🔎 Researching {field.replace('_', ' ').title()}...")
                
                try:
                    # Search with retry logic
                    search_results = self._search_with_retry(company_name, field)
                    field_data = self.data_extractor.extract_field_data(company_name, field, search_results)
                    company_data[field] = field_data
                    
                    # Dynamic sleep based on field complexity
                    sleep_time = 1.2 if field in ['branch_network_count', 'expansion_news_12mo'] else 0.8
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Research failed for {field}: {e}")
                    company_data[field] = "N/A"
                    continue
            
            # Step 3: Relevance Analysis
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar.progress(progress)
            status_text.info("📊 Conducting strategic relevance analysis...")
            
            relevance_bullets, intent_score = self.relevance_analyzer.analyze_relevance(
                company_data, company_name, company_data.get("core_intent_analysis", "N/A")
            )
            company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
            company_data["intent_scoring_level"] = intent_score
            
            progress_bar.progress(100)
            status_text.success("✅ Research complete!")
            
            return company_data
            
        except Exception as e:
            logger.error(f"Research engine failed: {e}")
            status_text.error(f"❌ Research failed: {e}")
            # Return partial data if available
            return company_data
    
    def _search_with_retry(self, company_name: str, field: str, max_retries: int = 2) -> List[Dict]:
        """Search with retry logic for reliability"""
        for attempt in range(max_retries + 1):
            try:
                results = self.search_service.search_for_field(company_name, field)
                if results:  # Return if we got some results
                    return results
                elif attempt < max_retries:
                    time.sleep(2.0)  # Longer delay before retry
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Search attempt {attempt + 1} failed for {field}: {e}")
                    time.sleep(2.0)
                else:
                    logger.error(f"All search attempts failed for {field}: {e}")
        
        return []  # Return empty list if all attempts fail

# --- Output Formatter ---
class OutputFormatter:
    """Format and display results"""
    
    def __init__(self):
        self.config = Config()
    
    def format_horizontal_display(self, company_name: str, data_dict: dict) -> pd.DataFrame:
        """Transform data into clean horizontal display format"""
        
        row_data = {"Company Name": company_name}
        
        for data_field, display_name in self.config.FIELD_DISPLAY_NAMES.items():
            value = data_dict.get(data_field, "N/A")
            # Ensure values are strings and handle long text
            if isinstance(value, str) and len(value) > 1000:
                value = value[:1000] + "... [truncated]"
            row_data[display_name] = value
        
        return pd.DataFrame([row_data])
    
    def create_excel_download(self, df: pd.DataFrame) -> BytesIO:
        """Create Excel file for download"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Company_Intel')
            # Auto-adjust column widths
            worksheet = writer.sheets['Company_Intel']
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, min(max_len, 50))
        return output.getvalue()

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Company Intelligence Generator", layout="wide")
    st.title("🏢 Dynamic Company Intelligence Generator")
    
    # Initialize services
    research_engine = CompanyResearchEngine()
    formatter = OutputFormatter()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    company_name = st.sidebar.text_input(
        "Enter Company Name:",
        value="",
        placeholder="e.g., Snowman Logistics",
        key="company_input"
    )
    
    article_url = st.sidebar.text_input(
        "Core Intent Article URL (Optional):",
        value="",
        placeholder="Paste article link for context",
        key="article_url_input"
    )
    
    # Initialize session state
    if 'research_data' not in st.session_state:
        st.session_state.research_data = None
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None
    
    # Research trigger
    if st.sidebar.button("🚀 Run Comprehensive Research", type="primary"):
        if not company_name:
            st.sidebar.error("Please enter a company name")
        else:
            # Validate company name
            is_valid, validation_msg = validate_company_name(company_name)
            if not is_valid:
                st.sidebar.error(validation_msg)
            else:
                st.session_state.current_company = company_name
                st.session_state.research_data = None
                
                with st.spinner(f"Starting comprehensive research for **{company_name}**..."):
                    try:
                        research_data = research_engine.research_company(company_name, article_url)
                        st.session_state.research_data = research_data
                    except Exception as e:
                        st.error(f"Research failed: {e}")
                        st.session_state.research_data = {"error": str(e)}
    
    # Display results
    if st.session_state.research_data and st.session_state.current_company:
        if "error" in st.session_state.research_data:
            st.error(f"Research failed: {st.session_state.research_data['error']}")
        else:
            st.header(f"📊 Intelligence Report: {st.session_state.current_company}")
            
            # Display dataframe
            df_display = formatter.format_horizontal_display(
                st.session_state.current_company, 
                st.session_state.research_data
            )
            
            st.dataframe(df_display, use_container_width=True, height=400)
            
            # Download section
            st.subheader("📥 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                excel_data = formatter.create_excel_download(df_display)
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name=f"{st.session_state.current_company}_Intelligence_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # JSON download option
                json_data = json.dumps(st.session_state.research_data, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"{st.session_state.current_company}_Intelligence_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    # Instructions
    with st.expander(" How to use this tool"):
        st.markdown("""
        **Instructions:**
        1. **Enter Company Name**: Provide the full company name for accurate research
        2. **Article URL (Optional)**: Add a relevant article for core intent analysis
        3. **Run Research**: Click the button to start comprehensive intelligence gathering
        4. **Review Results**: Check the generated intelligence report
        5. **Download**: Export results in Excel or JSON format

        **Research Includes:**
        - Company facilities and expansion news
        - Digital transformation initiatives  
        - IT leadership and vendor information
        - IoT/Automation projects
        - Cloud adoption and infrastructure
        - Strategic relevance analysis

        **Troubleshooting:**
        - If research stops, check your API quotas
        - For missing sources, try running research again
        - Ensure company names are specific and accurate
        """)

if __name__ == "__main__":
    main()
