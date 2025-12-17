# streamlit run dataset_explorer_v14.py
import streamlit as st
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import openai
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import math

# =============================================================================
# 1. APP CONFIGURATION & STYLING
# =============================================================================

st.set_page_config(
    page_title="Brightspace Data Universe",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# Configure structured logging
logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress insecure request warnings for D2L scrapers
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Apply "Pro" UI CSS
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #1E232B;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] { color: #8B949E; }
    div[data-testid="stMetricValue"] { color: #58A6FF; font-size: 24px; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0E1117;
        border-radius: 4px;
        padding: 8px 16px;
        color: #C9D1D9;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #238636;
        color: white;
        border-color: #30363D;
    }
    
    /* Code Blocks */
    .stCode { font-family: 'Fira Code', monospace; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SESSION STATE MANAGEMENT
# =============================================================================

def init_session_state():
    """Initializes Streamlit session state variables safely."""
    defaults = {
        'authenticated': False,
        'auth_error': False,
        'messages': [],
        'view_mode': 'Universe Map',
        'scrape_status': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# 3. AUTHENTICATION LOGIC
# =============================================================================

def perform_login():
    """Verifies password against Streamlit secrets or allows Dev mode."""
    pwd_secret = st.secrets.get("app_password")
    
    # Dev Mode: If no secret is configured, allow access
    if not pwd_secret:
        logger.warning("No password configured. Allowing open access.")
        st.session_state['authenticated'] = True
        return

    # Production Mode: Check input
    if st.session_state.get("password_input") == pwd_secret:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
    else:
        st.session_state['auth_error'] = True
        st.session_state['authenticated'] = False

def logout():
    """Clears authentication state."""
    st.session_state['authenticated'] = False
    st.session_state['password_input'] = ""
    st.session_state['messages'] = [] # Clear chat on logout

# =============================================================================
# 4. DATA LAYER (SCRAPER & STORAGE)
# =============================================================================

DEFAULT_URLS = """
https://community.d2l.com/brightspace/kb/articles/4752-accommodations-data-sets
https://community.d2l.com/brightspace/kb/articles/4712-activity-feed-data-sets
https://community.d2l.com/brightspace/kb/articles/4723-announcements-data-sets
https://community.d2l.com/brightspace/kb/articles/4767-assignments-data-sets
https://community.d2l.com/brightspace/kb/articles/4519-attendance-data-sets
https://community.d2l.com/brightspace/kb/articles/4520-awards-data-sets
https://community.d2l.com/brightspace/kb/articles/4521-calendar-data-sets
https://community.d2l.com/brightspace/kb/articles/4523-checklist-data-sets
https://community.d2l.com/brightspace/kb/articles/4754-competency-data-sets
https://community.d2l.com/brightspace/kb/articles/4713-content-data-sets
https://community.d2l.com/brightspace/kb/articles/22812-content-service-data-sets
https://community.d2l.com/brightspace/kb/articles/26020-continuous-professional-development-cpd-data-sets
https://community.d2l.com/brightspace/kb/articles/4725-course-copy-data-sets
https://community.d2l.com/brightspace/kb/articles/4524-course-publisher-data-sets
https://community.d2l.com/brightspace/kb/articles/26161-creator-data-sets
https://community.d2l.com/brightspace/kb/articles/4525-discussions-data-sets
https://community.d2l.com/brightspace/kb/articles/4526-exemptions-data-sets
https://community.d2l.com/brightspace/kb/articles/4527-grades-data-sets
https://community.d2l.com/brightspace/kb/articles/4528-intelligent-agents-data-sets
https://community.d2l.com/brightspace/kb/articles/5782-jit-provisioning-data-sets
https://community.d2l.com/brightspace/kb/articles/4714-local-authentication-data-sets
https://community.d2l.com/brightspace/kb/articles/4727-lti-data-sets
https://community.d2l.com/brightspace/kb/articles/4529-organizational-units-data-sets
https://community.d2l.com/brightspace/kb/articles/4796-outcomes-data-sets
https://community.d2l.com/brightspace/kb/articles/4530-portfolio-data-sets
https://community.d2l.com/brightspace/kb/articles/4531-questions-data-sets
https://community.d2l.com/brightspace/kb/articles/4532-quizzes-data-sets
https://community.d2l.com/brightspace/kb/articles/4533-release-conditions-data-sets
https://community.d2l.com/brightspace/kb/articles/33182-reoffer-course-data-sets
https://community.d2l.com/brightspace/kb/articles/4534-role-details-data-sets
https://community.d2l.com/brightspace/kb/articles/4535-rubrics-data-sets
https://community.d2l.com/brightspace/kb/articles/4536-scorm-data-sets
https://community.d2l.com/brightspace/kb/articles/4537-sessions-and-system-access-data-sets
https://community.d2l.com/brightspace/kb/articles/19147-sis-course-merge-data-sets
https://community.d2l.com/brightspace/kb/articles/33427-source-course-deploy-data-sets
https://community.d2l.com/brightspace/kb/articles/4538-surveys-data-sets
https://community.d2l.com/brightspace/kb/articles/4540-tools-data-sets
https://community.d2l.com/brightspace/kb/articles/4740-users-data-sets
https://community.d2l.com/brightspace/kb/articles/4541-virtual-classroom-data-sets
""".strip()

def scrape_table(url: str, category_name: str) -> List[Dict]:
    """
    Parses a D2L Knowledge Base page to extract dataset definitions.
    Returns a list of dictionaries representing columns.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            logger.warning(f"Status {response.status_code} for {url}")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        current_dataset = category_name
        
        # Logic: Headers (H2/H3) denote the Dataset Name, following Table is schema
        elements = soup.find_all(['h2', 'h3', 'table'])
        for element in elements:
            if element.name in ['h2', 'h3']:
                text = element.text.strip()
                if len(text) > 3: 
                    current_dataset = text.lower()
                    
            elif element.name == 'table':
                # Normalize headers
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                
                # Validation: ensure this is a metadata table
                if not table_headers or not any(x in table_headers for x in ['type', 'description', 'data_type']):
                    continue
                
                # Extract Rows
                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) < len(table_headers): 
                        continue
                    
                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_): 
                            entry[header] = columns_[i].text.strip()
                    
                    # Normalize Keys
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    clean_entry = {header_map.get(k, k): v for k, v in entry.items()}
                    
                    if 'column_name' in clean_entry and clean_entry['column_name']:
                        clean_entry['dataset_name'] = current_dataset
                        clean_entry['category'] = category_name
                        clean_entry['url'] = url
                        data.append(clean_entry)
                        
        return data
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return []

def scrape_and_save(urls: List[str]) -> pd.DataFrame:
    """
    Orchestrates the scraping process using ThreadPoolExecutor.
    Saves the result to 'dataset_metadata.csv'.
    """
    all_data = []
    progress_bar = st.progress(0, "Initializing Scraper...")
    
    # Helper to clean URLs
    def extract_category(url):
        filename = os.path.basename(url).split('?')[0]
        clean_name = re.sub(r'^\d+\s*', '', filename)
        return clean_name.replace('-data-sets', '').replace('-', ' ').lower()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        args = [(url, extract_category(url)) for url in urls]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        
        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception as e:
                logger.error(f"Thread error: {e}")
            
            progress_bar.progress((i + 1) / len(urls), f"Scraping {i+1}/{len(urls)}...")
            
    progress_bar.empty()

    if not all_data:
        st.error("Scraper returned no data. Check URLs.")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.fillna('')
    
    # Clean up Text
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    
    # Logic Flags for Joins
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    
    # Persist
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads the CSV from disk if it exists and is valid."""
    if os.path.exists('dataset_metadata.csv') and os.path.getsize('dataset_metadata.csv') > 10:
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()

@st.cache_data
def get_possible_joins(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all possible JOIN conditions based on PK/FK naming."""
    if df.empty: return pd.DataFrame()
    
    pks = df[df['is_primary_key'] == True]
    fks = df[df['is_foreign_key'] == True]
    
    if pks.empty or fks.empty: return pd.DataFrame()
    
    # Merge where Foreign Key Column Name matches Primary Key Column Name
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    
    # Exclude self-joins (joining a table to itself)
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    return joins

# =============================================================================
# 5. VISUALIZATION ENGINE (ORBITAL MAP)
# =============================================================================

@st.cache_data
def create_orbital_map(df: pd.DataFrame, target_node: str = None) -> go.Figure:
    """
    Generates the 'Solar System' map with Deterministic Geometry.
    Categories are Suns. Datasets are Planets.
    Uses strict deterministic math (Orbit) instead of physics (Spring).
    """
    if df.empty: return go.Figure()
    
    # 1. Prepare Data
    categories = sorted(df['category'].unique())
    datasets = df[['dataset_name', 'category', 'description']].drop_duplicates('dataset_name')
    
    # 2. Define Layout Physics (Orbit)
    pos = {}
    center_x, center_y = 0, 0
    orbit_radius_cat = 20 # Distance of Category from Center
    
    # Place Categories in a large ring
    cat_step = 2 * math.pi / len(categories) if categories else 1
    
    # Trace Containers
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    node_line_width, node_line_color = [], []
    cat_x, cat_y, cat_text = [], [], [] # Category Labels
    
    # 3. Determine Highlights (HUD Logic)
    active_edges = []
    active_neighbors = set()
    
    if target_node:
        joins = get_possible_joins(df)
        
        # Find neighbors
        out_ = joins[joins['dataset_name_fk'] == target_node]
        for _, r in out_.iterrows():
            active_edges.append((target_node, r['dataset_name_pk'], r['column_name']))
            active_neighbors.add(r['dataset_name_pk'])
            
        in_ = joins[joins['dataset_name_pk'] == target_node]
        for _, r in in_.iterrows():
            active_edges.append((r['dataset_name_fk'], target_node, r['column_name']))
            active_neighbors.add(r['dataset_name_fk'])

    # 4. Build Nodes
    for i, cat in enumerate(categories):
        # Category Position
        angle = i * cat_step
        cx = center_x + orbit_radius_cat * math.cos(angle)
        cy = center_y + orbit_radius_cat * math.sin(angle)
        pos[cat] = (cx, cy)
        
        # Add Category Node
        node_x.append(cx); node_y.append(cy)
        node_text.append(f"Category: {cat}")
        
        # Visuals: Categories are Gold Suns
        is_dim = (target_node is not None)
        node_color.append('rgba(255, 215, 0, 0.2)' if is_dim else 'rgba(255, 215, 0, 1)')
        node_size.append(35)
        node_line_width.append(0)
        node_line_color.append('rgba(0,0,0,0)')
        
        # Add Category Label
        cat_x.append(cx); cat_y.append(cy + 3); cat_text.append(cat)

        # Dataset Positions (Orbiting the Category)
        cat_ds = datasets[datasets['category'] == cat]
        ds_count = len(cat_ds)
        if ds_count > 0:
            ds_radius = 4 # Distance from Category
            ds_step = 2 * math.pi / ds_count
            
            for j, (_, row) in enumerate(cat_ds.iterrows()):
                ds_name = row['dataset_name']
                ds_angle = j * ds_step
                dx = cx + ds_radius * math.cos(ds_angle)
                dy = cy + ds_radius * math.sin(ds_angle)
                pos[ds_name] = (dx, dy)
                
                # Add Dataset Node
                node_x.append(dx); node_y.append(dy)
                
                # Visual Logic for Dataset
                if target_node:
                    if ds_name == target_node:
                        # TARGET
                        node_color.append('#00FF00') # Bright Green
                        node_size.append(20)
                        node_line_width.append(2); node_line_color.append('white')
                    elif ds_name in active_neighbors:
                        # NEIGHBOR
                        node_color.append('#00CCFF') # Blue
                        node_size.append(15)
                        node_line_width.append(1); node_line_color.append('white')
                    else:
                        # INACTIVE
                        node_color.append('rgba(50,50,50,0.3)')
                        node_size.append(8)
                        node_line_width.append(0); node_line_color.append('rgba(0,0,0,0)')
                else:
                    # DEFAULT STATE
                    node_color.append('#00CCFF')
                    node_size.append(10)
                    node_line_width.append(1); node_line_color.append('rgba(255,255,255,0.3)')

                desc_short = str(row['description'])[:80] + "..."
                node_text.append(f"<b>{ds_name}</b><br>{desc_short}")

    # 5. Build Edges (Lines)
    edge_x, edge_y = [], []
    label_x, label_y, label_text = [], [], []

    for s, t, k in active_edges:
        if s in pos and t in pos:
            x0, y0 = pos[s]
            x1, y1 = pos[t]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            label_x.append((x0+x1)/2); label_y.append((y0+y1)/2); label_text.append(k)

    # 6. Create Traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines', 
        line=dict(width=2, color='#00FF00'), 
        hoverinfo='none'
    )
    
    label_trace = go.Scatter(
        x=label_x, y=label_y, mode='text', text=label_text,
        textfont=dict(color='#00FF00', size=11, family="monospace", weight="bold"),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_text,
        marker=dict(
            color=node_color, 
            size=node_size, 
            line=dict(width=node_line_width, color=node_line_color)
        )
    )
    
    cat_label_trace = go.Scatter(
        x=cat_x, y=cat_y, mode='text', text=cat_text,
        textfont=dict(color='gold', size=10), hoverinfo='none'
    )

    fig = go.Figure(
        data=[edge_trace, label_trace, node_trace, cat_label_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=750
        )
    )
    return fig

# =============================================================================
# 6. SQL BUILDER ENGINE
# =============================================================================

def generate_manual_sql(selected_datasets: List[str], df: pd.DataFrame) -> str:
    """
    Generates a deterministic SQL JOIN query based on the graph relationships.
    Uses a 'Greedy' approach: connect each new table to the existing joined cluster.
    """
    if len(selected_datasets) < 2:
        return "-- Please select at least 2 datasets to generate a JOIN."
    
    # 1. Build the Full Connection Graph
    G_full = nx.Graph()
    joins = get_possible_joins(df)
    for _, r in joins.iterrows():
        # Store the Join Key as an edge attribute
        G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    # 2. Initialize Query
    base_table = selected_datasets[0]
    # Create simple aliases (t1, t2, etc.)
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}
    
    sql_lines = [f"SELECT TOP 100", f"    {aliases[base_table]}.*"]
    sql_lines.append(f"FROM {base_table} {aliases[base_table]}")
    
    # Set of tables already added to the query
    joined_tables = {base_table}
    
    # 3. Iterate through remaining tables
    remaining_tables = selected_datasets[1:]
    
    for current_table in remaining_tables:
        found_connection = False
        
        # Check if current_table connects to ANY table already in the query
        for existing_table in joined_tables:
            if G_full.has_edge(current_table, existing_table):
                key = G_full[current_table][existing_table]['key']
                
                sql_lines.append(
                    f"LEFT JOIN {current_table} {aliases[current_table]} "
                    f"ON {aliases[existing_table]}.{key} = {aliases[current_table]}.{key}"
                )
                
                joined_tables.add(current_table)
                found_connection = True
                break
        
        if not found_connection:
            # Fallback for unconnected tables
            sql_lines.append(
                f"CROSS JOIN {current_table} {aliases[current_table]} "
                f"-- ⚠️ No direct relationship found in metadata"
            )
            joined_tables.add(current_table)
            
    return "\n".join(sql_lines)

# =============================================================================
# 7. VIEW CONTROLLERS (MODULAR UI)
# =============================================================================

def render_sidebar(df: pd.DataFrame) -> str:
    """Renders the sidebar navigation and admin tools."""
    with st.sidebar:
        st.title("🌌 Data Universe")
        
        view = st.radio(
            "Navigation", 
            ["Universe Map", "Schema Explorer", "AI Architect"], 
            label_visibility="collapsed"
        )
        st.divider()
        
        if not df.empty:
            st.caption(f"Loaded {df['dataset_name'].nunique()} Datasets")
        
        # Scraper Accordion
        with st.expander("⚙️ Admin / Update Data"):
            txt = st.text_area("URLs", value=DEFAULT_URLS, height=100)
            if st.button("Run Scraper"):
                urls = [u.strip() for u in txt.split('\n') if u.startswith('http')]
                scrape_and_save(urls)
                load_data.clear()
                st.rerun()
        
        # Authentication Status
        if st.session_state['authenticated']:
            st.success("Authenticated")
            if st.button("Logout"): 
                logout()
                st.rerun()
        else:
            st.text_input(
                "Password", 
                type="password", 
                key="password_input", 
                on_change=perform_login
            )
            
    return view

def render_map_view(df: pd.DataFrame):
    """Renders the Interactive Galaxy Map."""
    st.subheader("Interactive Data Map")
    
    # Control Row
    col_search, col_stats = st.columns([3, 1])
    with col_search:
        all_ds = ["None"] + sorted(df['dataset_name'].unique())
        target = st.selectbox("🎯 Target Dataset (Select to trace connections)", all_ds)
        target_val = None if target == "None" else target
    
    with col_stats:
        if target_val:
            row_cnt = len(df[df['dataset_name'] == target_val])
            st.metric("Column Count", row_cnt)
        else:
            st.metric("Total Datasets", df['dataset_name'].nunique())

    # Map & Details Row
    c1, c2 = st.columns([3, 1])
    
    with c1:
        # Uses ORBITAL MAP (Solar System) which user preferred over Spring Physics
        fig = create_orbital_map(df, target_val)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        if target_val:
            st.markdown(f"### {target_val}")
            ds_data = df[df['dataset_name'] == target_val].iloc[0]
            st.caption(f"Category: {ds_data['category']}")
            
            if ds_data['url']: 
                st.link_button("View Documentation", ds_data['url'])
            
            # Show PK/FKs
            subset = df[df['dataset_name'] == target_val]
            pks = subset[subset['is_primary_key']]['column_name'].tolist()
            fks = subset[subset['is_foreign_key']]['column_name'].tolist()
            
            if pks: st.markdown(f"🔑 **PK:** {', '.join(pks)}")
            if fks: st.markdown(f"🔗 **FK:** {', '.join(fks)}")
            
            with st.expander("Full Schema", expanded=True):
                st.dataframe(
                    subset[['column_name', 'data_type']], 
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("Select a dataset on the map or dropdown to view its connections and schema.")

def render_schema_view(df: pd.DataFrame):
    """Renders the Schema Search and Manual SQL Builder."""
    st.subheader("🔎 Schema Search & SQL Builder")
    
    c_search, c_sel = st.columns(2)
    with c_search:
        search = st.text_input("Find Column", placeholder="e.g. OrgUnitId")
        if search:
            hits = df[df['column_name'].str.contains(search, case=False)]
            if not hits.empty:
                st.dataframe(
                    hits[['dataset_name', 'column_name', 'data_type', 'category']], 
                    use_container_width=True, 
                    height=200
                )
            else:
                st.warning("No matches found.")
                
    with c_sel:
        sel_ds = st.multiselect(
            "Select Datasets for SQL Builder", 
            sorted(df['dataset_name'].unique())
        )

    if sel_ds:
        st.divider()
        c_schema, c_sql = st.columns(2)
        
        with c_schema:
            st.markdown("#### 📋 Field Comparison")
            sub = df[df['dataset_name'].isin(sel_ds)]
            st.dataframe(
                sub[['dataset_name', 'column_name', 'data_type', 'is_primary_key', 'is_foreign_key']], 
                use_container_width=True
            )
        
        with c_sql:
            st.markdown("#### ⚡ Generated SQL")
            sql_code = generate_manual_sql(sel_ds, df)
            st.code(sql_code, language="sql")

def render_ai_view(df: pd.DataFrame):
    """Renders the AI Chat Interface."""
    st.subheader("🤖 Data Architect Assistant")
    
    if not st.session_state['authenticated']:
        st.warning("🔒 Login required to use AI features. Please enter password in sidebar.")
        st.stop()
    
    c_set, c_chat = st.columns([1, 3])
    
    with c_set:
        st.markdown("#### AI Settings")
        prov = st.selectbox("Provider", ["OpenAI", "xAI"])
        
        # API Key Input
        key_input = st.text_input("API Key", type="password")
        
        # Prefer secret, fallback to input
        if prov == "OpenAI":
            api_key = st.secrets.get("openai_api_key") or key_input
        else:
            api_key = st.secrets.get("xai_api_key") or key_input
            
        use_full = st.checkbox("Use Full Database Context", value=True, help="Sends entire schema to AI (more expensive, smarter).")
        
        if st.button("Clear Chat"): 
            st.session_state.messages = []
            st.rerun()

    with c_chat:
        # Display History
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): 
                st.markdown(m["content"])
                
        if prompt := st.chat_input("Ask a question (e.g., 'How do I link Grades to Users?')..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            if api_key:
                try:
                    # Build Context
                    if use_full:
                        # Optimized summary string to save tokens
                        ctx = df.groupby('dataset_name').apply(
                            lambda x: f"{x.name}: {','.join(x['column_name'])}"
                        ).str.cat(sep="\n")
                        ctx_head = "FULL DATABASE SCHEMA"
                    else:
                        ctx = df.head(50).to_csv()
                        ctx_head = "PARTIAL SAMPLE"

                    # Provider Logic
                    if prov == "xAI":
                        client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                        model = "grok-2-1212"
                    else:
                        client = openai.OpenAI(api_key=api_key)
                        model = "gpt-4o"
                    
                    # Call API
                    with st.spinner(f"Consulting {model}..."):
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": f"You are a Brightspace SQL Expert. Context ({ctx_head}):\n{ctx[:60000]}"},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        reply = resp.choices[0].message.content
                    
                    # Save & Display
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"AI Error: {str(e)}")
            else:
                st.error("Please provide an API Key in the settings or secrets.toml.")

# =============================================================================
# 8. MAIN ORCHESTRATOR
# =============================================================================

def main():
    df = load_data()
    
    view = render_sidebar(df)
    
    if df.empty:
        st.title("Welcome")
        st.info("No data found. Please use the Sidebar to scrape the KB articles.")
        return

    # View Router
    if view == "Universe Map":
        render_map_view(df)
    elif view == "Schema Explorer":
        render_schema_view(df)
    elif view == "AI Architect":
        render_ai_view(df)

if __name__ == "__main__":
    main()
