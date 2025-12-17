# streamlit run dataset_explorer_v9.py
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
from typing import List, Dict, Tuple, Optional, Set
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

# Configure Logging
logging.basicConfig(
    filename='scraper.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Custom CSS for Professional UI
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
    """Initializes Streamlit session state variables if they don't exist."""
    defaults = {
        'authenticated': False,
        'auth_error': False,
        'messages': [],
        'view_mode': 'Universe Map'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# 3. AUTHENTICATION LOGIC
# =============================================================================

def perform_login():
    """Verifies password against Streamlit secrets."""
    pwd_secret = st.secrets.get("app_password")
    
    # If no secret is configured, allow access (Dev Mode)
    if not pwd_secret:
        st.session_state['authenticated'] = True
        return

    # Check input
    if st.session_state.get("password_input") == pwd_secret:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
    else:
        st.session_state['auth_error'] = True
        st.session_state['authenticated'] = False

def logout():
    """Resets authentication state."""
    st.session_state['authenticated'] = False
    st.session_state['password_input'] = ""

# =============================================================================
# 4. SCRAPER LOGIC (ROBUST)
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
    """Scrapes a single D2L KB page for dataset tables."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        current_dataset = category_name
        
        # Iterate through headers and tables to map Datasets
        elements = soup.find_all(['h2', 'h3', 'table'])
        for element in elements:
            if element.name in ['h2', 'h3']:
                text = element.text.strip()
                if len(text) > 3: 
                    current_dataset = text.lower()
                    
            elif element.name == 'table':
                # Extract headers
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
                    
                    # Normalize Keys (handle variations in D2L docs)
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    clean_entry = {}
                    for k, v in entry.items():
                        clean_key = header_map.get(k, k)
                        clean_entry[clean_key] = v
                    
                    if 'column_name' in clean_entry and clean_entry['column_name']:
                        clean_entry['dataset_name'] = current_dataset
                        clean_entry['category'] = category_name
                        clean_entry['url'] = url
                        data.append(clean_entry)
                        
        return data
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return []

def scrape_and_save(urls: List[str]) -> pd.DataFrame:
    """Manages the ThreadPool for scraping and saves result to CSV."""
    all_data = []
    progress_bar = st.progress(0, "Initializing Scraper...")
    
    # Helper to guess category from URL
    def extract_category(url):
        filename = os.path.basename(url).split('?')[0]
        clean_name = re.sub(r'^\d+\s*', '', filename)
        return clean_name.replace('-data-sets', '').replace('-', ' ').lower()
    
    # Concurrent Scraping
    with ThreadPoolExecutor(max_workers=8) as executor:
        args = [(url, extract_category(url)) for url in urls]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        
        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception as e:
                logging.error(f"Thread error: {e}")
            
            progress_bar.progress((i + 1) / len(urls), f"Scraping {i+1}/{len(urls)}...")
            
    progress_bar.empty()

    if not all_data:
        return pd.DataFrame()

    # Post-Processing
    df = pd.DataFrame(all_data)
    df = df.fillna('')
    
    # Formatting
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    
    # Logic Flags
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads the cached CSV data."""
    if os.path.exists('dataset_metadata.csv') and os.path.getsize('dataset_metadata.csv') > 10:
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()

@st.cache_data
def get_joins(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all possible PK/FK relationships."""
    if df.empty: return pd.DataFrame()
    
    pks = df[df['is_primary_key'] == True]
    fks = df[df['is_foreign_key'] == True]
    
    if pks.empty or fks.empty: return pd.DataFrame()
    
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    # Filter self-joins
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    return joins

# =============================================================================
# 5. VISUALIZATION ENGINE (HYBRID MAP)
# =============================================================================

@st.cache_data
def create_hybrid_map(df: pd.DataFrame, target_node: str = None) -> go.Figure:
    """
    Generates the Interactive Universe Map using Plotly.
    Uses 'Clustered Spring Layout' for organic node placement.
    Uses 'HUD Logic' to only show lines when a Target is selected.
    """
    if df.empty: return go.Figure()
    
    # 1. Build NetworkX Graph for Layout Calculation
    G = nx.Graph()
    
    # Aggregate data to get 1 node per dataset
    dataset_info = df.groupby('dataset_name').agg({
        'category': 'first',
        'column_name': 'count'
    }).reset_index()
    
    for _, row in dataset_info.iterrows():
        G.add_node(row['dataset_name'], category=row['category'], size=row['column_name'])

    # 2. Physics: Clustered Spring Layout
    # This creates the "Cloud" look where datasets hover near their categories
    categories = sorted(list(set(nx.get_node_attributes(G, 'category').values())))
    
    # Generate colors
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1
    cat_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
    
    # Seed category positions in a circle
    n_cats = len(categories)
    cat_pos = {}
    radius = 4
    for i, cat in enumerate(categories):
        angle = 2 * np.pi * i / n_cats
        cat_pos[cat] = (np.cos(angle) * radius, np.sin(angle) * radius)
        
    # Initial node positions (jittered around category center)
    initial_pos = {}
    for node in G.nodes():
        cat = G.nodes[node]['category']
        base_x, base_y = cat_pos[cat]
        initial_pos[node] = (
            base_x + np.random.uniform(-0.8, 0.8), 
            base_y + np.random.uniform(-0.8, 0.8)
        )

    # Run Physics Simulation
    pos = nx.spring_layout(G, k=1.2, iterations=80, pos=initial_pos, seed=42)

    # 3. Determine Active Connections (HUD Logic)
    active_edges = []
    active_neighbors = set()
    
    if target_node:
        joins = get_joins(df)
        
        # Outgoing connections (Target -> Others)
        out_ = joins[joins['dataset_name_fk'] == target_node]
        for _, r in out_.iterrows():
            active_edges.append((target_node, r['dataset_name_pk'], r['column_name']))
            active_neighbors.add(r['dataset_name_pk'])
            
        # Incoming connections (Others -> Target)
        in_ = joins[joins['dataset_name_pk'] == target_node]
        for _, r in in_.iterrows():
            active_edges.append((r['dataset_name_fk'], target_node, r['column_name']))
            active_neighbors.add(r['dataset_name_fk'])

    # 4. Draw Edges (Lines)
    edge_x, edge_y = [], []
    label_x, label_y, label_text = [], [], []
    
    for s, t, k in active_edges:
        if s in pos and t in pos:
            x0, y0 = pos[s]
            x1, y1 = pos[t]
            
            # Line coordinates
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Label coordinates (Midpoint)
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            label_text.append(k)

    # Trace: Edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#58A6FF'), # Bright Blue
        hoverinfo='none'
    )
    
    # Trace: Edge Labels (Key names)
    label_trace = go.Scatter(
        x=label_x, y=label_y,
        mode='text',
        text=label_text,
        textfont=dict(color='#58A6FF', size=11, family="monospace", weight="bold"),
        hoverinfo='none'
    )

    # 5. Draw Nodes (Dots)
    node_x, node_y, node_text, node_color, node_size, node_line = [], [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        cat = G.nodes[node]['category']
        node_x.append(x)
        node_y.append(y)
        
        # Visual Logic
        base_color = cat_colors.get(cat, '#888')
        
        if target_node:
            # Focused Mode
            if node == target_node:
                node_color.append('#58A6FF') # Highlight Target
                node_size.append(25)
                node_line.append(dict(width=3, color='white'))
            elif node in active_neighbors:
                node_color.append(base_color) # Keep neighbor color
                node_size.append(18)
                node_line.append(dict(width=2, color='white'))
            else:
                # Dim everyone else
                node_color.append('rgba(50,50,50,0.3)')
                node_size.append(8)
                node_line.append(dict(width=0, color='black'))
        else:
            # Default Mode (All visible)
            node_color.append(base_color)
            node_size.append(10 + min(G.nodes[node]['size'] * 0.2, 10))
            node_line.append(dict(width=1, color='rgba(255,255,255,0.5)'))
        
        # Tooltip Text
        counts = dataset_info[dataset_info['dataset_name'] == node].iloc[0]
        node_text.append(f"<b>{node}</b><br>Category: {cat}<br>Cols: {counts['column_name']}")

    # Trace: Nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            color=node_color, 
            size=node_size, 
            line=node_line[0] if len(node_line)==1 else node_line
        )
    )

    # 6. Final Layout
    fig = go.Figure(
        data=[edge_trace, label_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=700
        )
    )
    return fig

# =============================================================================
# 6. SQL BUILDER ENGINE
# =============================================================================

def generate_manual_sql(selected_datasets: List[str], df: pd.DataFrame) -> str:
    """
    Generates a deterministic SQL JOIN query based on the graph relationships.
    Uses a Greedy approach to connect new tables to the existing cluster.
    """
    if len(selected_datasets) < 2:
        return "-- Please select at least 2 datasets to generate a JOIN."
    
    # 1. Build the Full Connection Graph
    G_full = nx.Graph()
    joins = get_joins(df)
    for _, r in joins.iterrows():
        # Add edge between datasets with the Key as the attribute
        G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    # 2. Initialize Query
    base_table = selected_datasets[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}
    
    sql_lines = [f"SELECT TOP 100", f"    {aliases[base_table]}.*"]
    sql_lines.append(f"FROM {base_table} {aliases[base_table]}")
    
    # Set of tables already added to the query
    joined_tables = {base_table}
    
    # 3. Iterate through remaining tables
    remaining_tables = selected_datasets[1:]
    
    for current_table in remaining_tables:
        # Try to find a direct connection to ANY table already in the join
        found_connection = False
        
        for existing_table in joined_tables:
            if G_full.has_edge(current_table, existing_table):
                # We found a path!
                key = G_full[current_table][existing_table]['key']
                
                # Write the LEFT JOIN
                sql_lines.append(
                    f"LEFT JOIN {current_table} {aliases[current_table]} "
                    f"ON {aliases[existing_table]}.{key} = {aliases[current_table]}.{key}"
                )
                
                joined_tables.add(current_table)
                found_connection = True
                break
        
        if not found_connection:
            # Fallback if no direct FK/PK relationship exists
            sql_lines.append(
                f"CROSS JOIN {current_table} {aliases[current_table]} "
                f"-- ⚠️ No direct relationship found to existing tables"
            )
            joined_tables.add(current_table)
            
    return "\n".join(sql_lines)

# =============================================================================
# 7. MAIN APP LOGIC
# =============================================================================

def main():
    df = load_data()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🌌 Data Universe")
        
        # Navigation
        view = st.radio("Navigation", ["Universe Map", "Schema Explorer", "AI Architect"], label_visibility="collapsed")
        st.divider()
        
        if not df.empty:
            st.caption(f"Loaded {df['dataset_name'].nunique()} Datasets")
        
        # Scraper Admin
        with st.expander("⚙️ Admin / Update Data"):
            txt = st.text_area("URLs", value=DEFAULT_URLS, height=100)
            if st.button("Run Scraper"):
                urls = [u.strip() for u in txt.split('\n') if u.startswith('http')]
                scrape_and_save(urls)
                load_data.clear()
                st.rerun()
        
        # Authentication
        if st.session_state['authenticated']:
            st.success("Authenticated")
            if st.button("Logout"): 
                logout()
                st.rerun()
        else:
            st.text_input("Password", type="password", key="password_input", on_change=perform_login)

    # Check Data Load
    if df.empty:
        st.title("Welcome")
        st.warning("No data loaded. Please use the Sidebar to scrape the KB articles.")
        st.stop()

    # --- VIEW 1: UNIVERSE MAP ---
    if view == "Universe Map":
        st.subheader("Interactive Data Map")
        
        # Map Controls
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

        # Split View
        c1, c2 = st.columns([3, 1])
        
        with c1:
            fig = create_hybrid_map(df, target_val)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            if target_val:
                st.markdown(f"### {target_val}")
                ds_data = df[df['dataset_name'] == target_val].iloc[0]
                st.caption(f"Category: {ds_data['category']}")
                
                if ds_data['url']: 
                    st.link_button("View Docs", ds_data['url'])
                
                st.markdown("#### Key Fields")
                subset = df[df['dataset_name'] == target_val]
                pks = subset[subset['is_primary_key']]['column_name'].tolist()
                fks = subset[subset['is_foreign_key']]['column_name'].tolist()
                
                if pks: st.markdown(f"🔑 **PK:** {', '.join(pks)}")
                if fks: st.markdown(f"🔗 **FK:** {', '.join(fks)}")
                
                with st.expander("Full Schema"):
                    st.dataframe(subset[['column_name', 'data_type']], hide_index=True)
            else:
                st.info("Select a dataset on the map or dropdown to view details.")

    # --- VIEW 2: SCHEMA EXPLORER ---
    elif view == "Schema Explorer":
        st.subheader("🔎 Schema Search & SQL Builder")
        
        c_search, c_sel = st.columns(2)
        with c_search:
            search = st.text_input("Find Column", placeholder="e.g. OrgUnitId")
            if search:
                hits = df[df['column_name'].str.contains(search, case=False)]
                if not hits.empty:
                    st.dataframe(hits[['dataset_name', 'column_name', 'data_type', 'category']], use_container_width=True, height=200)
                else:
                    st.warning("No matches found.")
        
        with c_sel:
            sel_ds = st.multiselect("Select Datasets for SQL Builder", sorted(df['dataset_name'].unique()))

        if sel_ds:
            st.divider()
            c_schema, c_sql = st.columns(2)
            
            with c_schema:
                st.markdown("#### 📋 Field Comparison")
                sub = df[df['dataset_name'].isin(sel_ds)]
                st.dataframe(sub[['dataset_name', 'column_name', 'data_type', 'is_primary_key', 'is_foreign_key']], use_container_width=True)
            
            with c_sql:
                st.markdown("#### ⚡ Generated SQL")
                sql_code = generate_manual_sql(sel_ds, df)
                st.code(sql_code, language="sql")

    # --- VIEW 3: AI ARCHITECT ---
    elif view == "AI Architect":
        st.subheader("🤖 Data Architect Assistant")
        
        if not st.session_state['authenticated']:
            st.warning("🔒 Login required to use AI features.")
            st.stop()
        
        c_set, c_chat = st.columns([1, 3])
        
        with c_set:
            st.markdown("#### AI Settings")
            prov = st.selectbox("Provider", ["OpenAI", "xAI"])
            key = st.text_input("API Key", type="password")
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
                
                if key:
                    try:
                        # Build Context
                        if use_full:
                            # Optimized summary for tokens
                            ctx = df.groupby('dataset_name').apply(
                                lambda x: f"{x.name}: {','.join(x['column_name'])}"
                            ).str.cat(sep="\n")
                            ctx_head = "FULL DATABASE SCHEMA"
                        else:
                            ctx = df.head(50).to_csv()
                            ctx_head = "PARTIAL SAMPLE"

                        # API Configuration
                        if prov == "xAI":
                            client = openai.OpenAI(api_key=key, base_url="https://api.x.ai/v1")
                            model = "grok-2-1212"
                        else:
                            client = openai.OpenAI(api_key=key)
                            model = "gpt-4o"
                        
                        # Call API
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
                    st.error("Please provide an API Key.")

if __name__ == "__main__":
    main()
