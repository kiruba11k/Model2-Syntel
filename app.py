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
    # --- CRITICAL MODIFICATION: Auto-Execution Setup ---
    # Set a default company for the initial run
    DEFAULT_COMPANY = "Snowman Logistics"
    
    st.title("🤖 Dynamic Company Intelligence Generator")
    st.sidebar.header("Configuration")
    
    # Input field
    company_name = st.sidebar.text_input("Enter Company Name to Research:", DEFAULT_COMPANY)
    
    # Initialize session state variables on the very first run
    if 'company_name' not in st.session_state:
        st.session_state['company_name'] = DEFAULT_COMPANY
        st.session_state['company_data'] = None

    # Determine if a search needs to be triggered
    trigger_search = st.sidebar.button("Run Comprehensive Research")
    
    # Condition to trigger search:
    # 1. Manual button click, OR
    # 2. Company name in the text input has changed since the last run, AND 
    # 3. We have a valid company name
    if trigger_search or (company_name != st.session_state.get('company_name', DEFAULT_COMPANY) and company_name):
        st.session_state['company_name'] = company_name
        # Clear data to force a rerun
        st.session_state['company_data'] = None

    # Auto-run if data is not present in the session state
    if st.session_state['company_data'] is None and st.session_state.get('company_name'):
        
        with st.spinner(f"Starting comprehensive research for **{st.session_state['company_name']}**..."):
            # This call now works because the function is defined above
            company_data = dynamic_research_company_intelligence(st.session_state['company_name']) 
            st.session_state['company_data'] = company_data
            
        st.success(f"Research for **{st.session_state['company_name']}** completed successfully.")

    # Display results
    if 'company_data' in st.session_state and st.session_state['company_data']:
        st.header(f"📊 Extracted Intelligence: {st.session_state['company_name']}")
        
        # Format the data into a clean DataFrame
        df_display = format_concise_display_with_sources(
            st.session_state['company_name'], 
            st.session_state['company_data']
        )
        
        # Display as a table, hiding the index
        st.dataframe(df_display.set_index('Column Header'), use_container_width=True)

        # Download button
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=True, sheet_name='Company_Intel')
            writer.close()
            processed_data = output.getvalue()
            return processed_data

        excel_data = to_excel(df_display.set_index('Column Header'))
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{st.session_state['company_name']}_Intelligence_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )    # ... (The rest of the Streamlit code remains unchanged) ...
    # ...
