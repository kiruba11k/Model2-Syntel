import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional, Tuple
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
                '"{company}" warehouse facility count pallet capacity 2024 2025'
            ],
            "expansion_news_12mo": [
                '"{company}" expansion news 2024 2025 new facilities',
                '"{company}" new warehouse construction Q3 Q4 2024 2025'
            ],
            "digital_transformation_initiatives": [
                '"{company}" digital transformation IT initiatives 2024'
            ],
            "it_leadership_change": [
                '"{company}" CIO CTO IT infrastructure leadership 2024'
            ],
            "existing_network_vendors": [
                '"{company}" network infrastructure vendors Cisco HPE Aruba',
                '"{company}" IT technology stack network equipment'
            ],
            "wifi_lan_tender_found": [
                '"{company}" WiFi LAN tender network upgrade 2024'
            ],
            "iot_automation_edge_integration": [
                '"{company}" IoT automation robotics implementation'
            ],
            "cloud_adoption_gcc_setup": [
                '"{company}" cloud adoption AWS Azure GCC setup'
            ],
            "physical_infrastructure_signals": [
                '"{company}" new construction facility expansion'
            ],
            "it_infra_budget_capex": [
                '"{company}" IT budget capex investment technology spending'
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
                time.sleep(0.6)  # Rate limiting
                results = self.search_tool.invoke({"query": query, "max_results": 3})
                
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            content = result.get('content', '') or result.get('snippet', '')
                            if len(content) > 50:  # Filter out very short content
                                all_results.append({
                                    "title": result.get('title', ''),
                                    "content": content[:800],  # Limit content length
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
            Extract factual numbers about {company_name}'s branch network or facilities count.
            
            RESEARCH: {research_context}
            
            RULES:
            - Extract ONLY explicit numbers mentioned
            - Return "N/A" if no specific count found
            - No estimations or calculations
            - Start with extracted data
            
            EXTRACTED:
            """,
            
            "expansion_news_12mo": f"""
            Extract specific expansion announcements for {company_name} from the last 12 months.
            
            RESEARCH: {research_context}
            
            RULES:
            - Extract ONLY explicit expansion news with dates/locations
            - Return "N/A" if no expansion news found
            - No inferences
            - Start with extracted data
            
            EXTRACTED:
            """,
            
            "digital_transformation_initiatives": f"""
            Extract specific digital transformation initiatives for {company_name}.
            
            RESEARCH: {research_context}
            
            RULES:
            - Extract ONLY explicitly mentioned initiatives
            - Include specific technologies if named
            - Return "N/A" if none found
            - Start with extracted data
            
            EXTRACTED:
            """
        }
        
        # Default prompt for other fields
        return prompts.get(field_name, f"""
        Extract factual information about {field_name} for {company_name}.
        
        RESEARCH: {research_context}
        
        RULES:
        - Extract ONLY explicit information
        - Be concise and factual
        - Return "N/A" if no information found
        - Start with extracted data
        
        EXTRACTED:
        """)
    
    def extract_field_data(self, company_name: str, field_name: str, search_results: List[Dict]) -> str:
        """Extract field data from search results"""
        
        if not search_results:
            return "N/A"
        
        # Build research context
        research_context = f"Research for {company_name} - {field_name}:\n"
        for i, result in enumerate(search_results[:4]):
            research_context += f"Source {i+1}: {result['content']}\n\n"
        
        unique_urls = list(set([result['url'] for result in search_results if result.get('url')]))[:2]
        
        try:
            prompt = self.get_extraction_prompt(company_name, field_name, research_context)
            
            response = self.llm.invoke([
                SystemMessage(content="""You are a precise data extraction assistant. 
                Extract ONLY information explicitly stated in the research. 
                If information is not found, respond with exactly "N/A".
                Be factual and concise."""),
                HumanMessage(content=prompt)
            ]).content.strip()
            
            # Validate response
            if self._is_invalid_response(response):
                return "N/A"
            
            # Clean response
            response = self._clean_response(response)
            
            # Add sources if data found
            if response != "N/A" and unique_urls:
                source_text = f" [Sources: {', '.join(unique_urls)}]" if len(unique_urls) > 1 else f" [Source: {unique_urls[0]}]"
                response += source_text
            
            return response[:500]  # Limit response length
            
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
            'no specific', 'unable to find', 'could not find'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in invalid_indicators)
    
    def _clean_response(self, response: str) -> str:
        """Clean up response text"""
        # Remove introductory phrases
        clean_up_phrases = [
            r'^\s*Based on (the|this).*:', 
            r'^\s*Here is.*:', 
            r'^\s*Extracted.*:', 
            r'^\s*The.*for.*is:',
            r'^\s*\*\s*', 
            r'^\s*-\s*', 
            r'^\s*\d+\.\s*', 
        ]
        
        for phrase in clean_up_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE).strip()
        
        # Final cleaning
        response = re.sub(r'\n+', ' ', response).strip() 
        response = re.sub(r'\s+', ' ', response)
        response = response.replace("**", "").replace("*", "")
        
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
            # Try to get content from the specific URL
            search_results = self.search_tool.invoke({
                "query": f"site:{article_url}",
                "max_results": 1
            })
            
            article_content = ""
            if search_results and isinstance(search_results, list):
                for result in search_results:
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        if content:
                            article_content = content[:1500]
                            break
            
            if not article_content:
                # Fallback to general search
                search_results = self.search_tool.invoke({
                    "query": f'"{company_name}" recent news',
                    "max_results": 2
                })
                
                article_content = "Recent news context:\n"
                for i, result in enumerate(search_results[:2]):
                    if isinstance(result, dict):
                        content = result.get('content', '') or result.get('snippet', '')
                        article_content += f"News {i+1}: {content}\n\n"
            
            prompt = f"""
            Analyze this content about {company_name} and identify the core business intent:
            
            CONTENT: {article_content}
            
            Focus on:
            1. Main business objective/strategic move
            2. Implied technology/infrastructure needs
            3. Network/infrastructure requirements
            
            Provide a concise analysis focusing on strategic intent.
            """
            
            response = self.llm.invoke([
                SystemMessage(content="You are a strategic business analyst."),
                HumanMessage(content=prompt)
            ]).content.strip()
            
            return f"{response} [Article: {article_url}]" if response else f"N/A - Could not analyze [URL: {article_url}]"
            
        except Exception as e:
            logger.error(f"Core intent analysis failed: {e}")
            return f"N/A - Analysis error [URL: {article_url}]"

# --- Relevance Analysis ---
class RelevanceAnalyzer:
    """Strategic relevance analysis"""
    
    def __init__(self, llm_service):
        self.llm = llm_service
    
    def analyze_relevance(self, company_data: Dict, company_name: str, core_intent: str) -> Tuple[str, str]:
        """Analyze relevance to Syntel and generate intent score"""
        
        # Prepare data context
        context_lines = []
        for field, value in company_data.items():
            if value and value != "N/A" and field not in ["why_relevant_to_syntel_bullets", "intent_scoring_level", "core_intent_analysis"]:
                clean_value = re.sub(r'\[Sources?:[^\]]+\]', '', value).strip()
                if clean_value:
                    context_lines.append(f"{field.replace('_', ' ').title()}: {clean_value}")
        
        data_context = "\n".join(context_lines) if context_lines else "Limited data available"

        prompt = f"""
        Analyze {company_name}'s relevance to Syntel's Wi-Fi & Network Integration GTM.

        SYNTEL FOCUS:
        - Industries: Ports, Stadiums, Education, Manufacturing, Healthcare, Hospitality, Warehouses, BFSI, IT/ITES, GCCs
        - Key Signals: Expansion, Digital Transformation, Wi-Fi/LAN upgrades, IoT/Automation, Leadership changes
        - Offerings: Wi-Fi deployments, Network integration, Multi-vendor implementation

        COMPANY DATA:
        {data_context}

        CORE INTENT:
        {core_intent}

        TASK:
        1. Generate 3 concise bullet points "Why Relevant to Syntel"
        2. Assign Intent Score (High/Medium/Low)
        3. Integrate core intent insights

        OUTPUT FORMAT (TSV):
        Company Name<TAB>Why Relevant to Syntel<TAB>Intent Score

        RULES:
        - Bullets must start with "• " and be separated by newlines
        - Be specific and actionable
        - No markdown formatting
        """

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a GTM analyst. Output ONLY in the specified TSV format."),
                HumanMessage(content=prompt)
            ]).content.strip()

            # Parse response
            parts = response.split('\t')
            if len(parts) == 3:
                _, relevance_text, score = parts
                
                # Clean and format bullets
                bullets = []
                for line in relevance_text.split('\n'):
                    clean_line = re.sub(r'^[•\-\s]*', '• ', line.strip())
                    clean_line = re.sub(r'\*\*|\*|__|_', '', clean_line)
                    if clean_line and len(clean_line) > 5:
                        bullets.append(clean_line)
                
                # Ensure we have exactly 3 bullets
                while len(bullets) < 3:
                    bullets.append(self._get_fallback_bullet(len(bullets), company_data, core_intent))
                
                formatted_bullets = "\n".join(bullets[:3])
                return formatted_bullets, score.strip()

            raise ValueError("Invalid response format")

        except Exception as e:
            logger.error(f"Relevance analysis failed: {e}")
            return self._get_fallback_analysis(company_data, core_intent)

    def _get_fallback_bullet(self, index: int, company_data: Dict, core_intent: str) -> str:
        """Get fallback bullet points"""
        fallbacks = [
            "• Operations in target sector align with Syntel's network expertise",
            "• Infrastructure scale suggests need for professional network services",
            "• Digital initiatives indicate potential for network modernization"
        ]
        return fallbacks[index] if index < len(fallbacks) else fallbacks[-1]

    def _get_fallback_analysis(self, company_data: Dict, core_intent: str) -> Tuple[str, str]:
        """Generate fallback relevance analysis"""
        bullets = []
        
        # Bullet 1: Core intent or general
        if "N/A" not in core_intent:
            bullets.append("• Strategic initiatives indicate network infrastructure requirements")
        else:
            bullets.append("• Company operates in sectors requiring robust network solutions")
        
        # Bullet 2: Expansion signals
        if company_data.get('expansion_news_12mo') not in ["N/A", ""]:
            bullets.append("• Expansion activities create immediate network deployment opportunities")
        else:
            bullets.append("• Scale of operations suggests network infrastructure needs")
        
        # Bullet 3: Technology signals
        if company_data.get('iot_automation_edge_integration') not in ["N/A", ""]:
            bullets.append("• IoT/Automation initiatives require high-performance network infrastructure")
        else:
            bullets.append("• Potential for network upgrades and modernization projects")

        return "\n".join(bullets), "Medium"

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
        """Conduct comprehensive company research"""
        
        company_data = {}
        total_fields = len(self.config.REQUIRED_FIELDS) - 3  # Exclude analysis fields
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Core Intent Analysis
        if article_url:
            status_text.info("🔍 Analyzing core intent article...")
            company_data["core_intent_analysis"] = self.intent_analyzer.analyze_article(article_url, company_name)
            progress_bar.progress(10)
        else:
            company_data["core_intent_analysis"] = "N/A - No article URL provided"
        
        # Step 2: Field Research
        research_fields = [f for f in self.config.REQUIRED_FIELDS 
                         if f not in ["core_intent_analysis", "why_relevant_to_syntel_bullets", "intent_scoring_level"]]
        
        for i, field in enumerate(research_fields):
            progress = 10 + (i / len(research_fields)) * 70
            progress_bar.progress(int(progress))
            status_text.info(f"🔎 Researching {field.replace('_', ' ').title()}...")
            
            try:
                search_results = self.search_service.search_for_field(company_name, field)
                field_data = self.data_extractor.extract_field_data(company_name, field, search_results)
                company_data[field] = field_data
                
                time.sleep(0.8)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Research failed for {field}: {e}")
                company_data[field] = "N/A"
                continue
        
        # Step 3: Relevance Analysis
        status_text.info("📊 Conducting strategic relevance analysis...")
        progress_bar.progress(90)
        
        try:
            relevance_bullets, intent_score = self.relevance_analyzer.analyze_relevance(
                company_data, company_name, company_data.get("core_intent_analysis", "N/A")
            )
            company_data["why_relevant_to_syntel_bullets"] = relevance_bullets
            company_data["intent_scoring_level"] = intent_score
        except Exception as e:
            logger.error(f"Relevance analysis failed: {e}")
            company_data["why_relevant_to_syntel_bullets"] = "• Analysis incomplete - review data manually"
            company_data["intent_scoring_level"] = "Medium"
        
        progress_bar.progress(100)
        status_text.success("✅ Research complete!")
        
        return company_data

# --- Output Formatter ---
class OutputFormatter:
    """Format and display results"""
    
    def __init__(self):
        self.config = Config()
    
    def format_horizontal_display(self, company_name: str, data_dict: dict) -> pd.DataFrame:
        """Transform data into clean horizontal display format"""
        
        row_data = {"Company Name": company_name}
        
        for data_field, display_name in self.config.FIELD_DISPLAY_NAMES.items():
            row_data[display_name] = data_dict.get(data_field, "N/A")
        
        return pd.DataFrame([row_data])
    
    def create_excel_download(self, df: pd.DataFrame) -> BytesIO:
        """Create Excel file for download"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Company_Intel')
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
                    research_data = research_engine.research_company(company_name, article_url)
                    st.session_state.research_data = research_data
    
    # Display results
    if st.session_state.research_data and st.session_state.current_company:
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
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
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
        """)

if __name__ == "__main__":
    main()
