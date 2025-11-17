# --- Core Functions (Changes made in get_detailed_extraction_prompt) ---

def get_detailed_extraction_prompt(company_name: str, field_name: str, research_context: str) -> str:
    """Get detailed extraction prompts for each field with strict single-fact requirements"""
    
    # ... (Other field prompts remain the same for conciseness) ...
    
    # MODIFIED PROMPT FOR HIGH ACCURACY (Branch Network / Facilities Count)
    prompts = {
        # ... (other prompts) ...
        
        "branch_network_count": f"""
        Analyze the research data and extract ONLY the **latest, consolidated total** of physical facilities/warehouses/locations for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Output ONE single, consolidated number and the associated capacity/location count (e.g., '44 warehouses across 21 cities with 1,54,330 pallets' or '542 locations').
        - **Prioritize the most recent (2025/2026) figure over older ones.**
        - Start directly with the extracted data.
        
        EXTRACTED NETWORK COUNT:
        """,
        
        # MODIFIED PROMPT FOR HIGH ACCURACY (Expansion News (Last 12 Months))
        "expansion_news_12mo": f"""
        Extract ONLY the most recent and significant expansion news for {company_name} from the **last 12-24 months (2024 and 2025/2026)**. Consolidate new facilities, geographic expansions, and fleet/capacity additions.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List specific new facilities, their capacity (if available), and the operational/announced dates (e.g., 'New Processing Lab in Abhiramapuram, Chennai (Oct 2025)').
        - **Focus strictly on announced/completed projects between late 2024 and Q4 2025.**
        - Start directly with the extracted data.
        
        EXTRACTED EXPANSION NEWS:
        """,
        
        # ... (other prompts) ...
        
        "physical_infrastructure_signals": f"""
        Extract ONLY the key physical infrastructure developments for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - List new construction projects and facility expansions (e.g., 'Ranchi Integrated Diagnostics Centre, JV with Star Imaging in Maharashtra').
        - Start directly with the extracted data.
        
        EXTRACTED INFRASTRUCTURE DEVELOPMENTS:
        """,
        
        "it_infra_budget_capex": f"""
        Extract ONLY the specific IT infrastructure budget and capital expenditure information for {company_name}.
        
        RESEARCH DATA:
        {research_context}
        
        REQUIREMENTS:
        - Provide specific budget figures, timeframes, or investment focus areas (e.g., 'No figures found, focus on digital transformation and expansion').
        - Start directly with the extracted data.
        
        EXTRACTED IT BUDGET INFORMATION:
        """
    }
    
    return prompts.get(field_name, f"""
    Extract ONLY the comprehensive, short, and correct information about {field_name} for {company_name}.
    
    RESEARCH DATA:
    {research_context}
    
    REQUIREMENTS:
    - Output must be short, factual, and extremely concise.
    - Start directly with the extracted data.
    
    EXTRACTED INFORMATION:
    """)


# --- Streamlit UI (Execution Block - Remains the same) ---
if __name__ == "__main__":
    # ... (The rest of the Streamlit code remains unchanged) ...
    # ...
