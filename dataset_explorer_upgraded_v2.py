# streamlit run dataset_explorer_upgraded_v2.py
import streamlit as st
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import plotly.graph_objects as go
import openai
import re
import logging
import math
import numpy as np

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
st.set_page_config(page_title="Brightspace Datasets Explorer & AI", layout="wide", page_icon="🕸️")

logging.basicConfig(filename='scraper.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0
if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state: st.session_state['auth_error'] = False
if 'map_view_mode' not in st.session_state: st.session_state['map_view_mode'] = 'Constellation'

# ==========================================
# AUTHENTICATION LOGIC
# ==========================================
def perform_login():
    """Callback to verify password."""
    pwd = st.secrets.get("app_password")
    # If no password set in secrets, allow bypass (for demo purposes)
    if not pwd:
        st.session_state['authenticated'] = True
        return

    if st.session_state.get("password_input") == pwd:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
    else:
        st.session_state['auth_error'] = True
        st.session_state['authenticated'] = False

def logout():
    st.session_state['authenticated'] = False
    st.session_state['password_input'] = ""

def clear_all_selections():
    for key in list(st.session_state.keys()):
        if key.startswith("sel_") or key == "global_search":
            st.session_state[key] = []

# ==========================================
# DATA LOADING & SCRAPING
# ==========================================
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

def parse_urls_from_text_area(text_block):
    urls = [line.strip() for line in text_block.split('\n') if line.strip()]
    valid_urls = [url for url in urls if url.startswith('http')]
    return sorted(list(set(valid_urls)))

def scrape_table(url, category_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200: return []
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        elements = soup.find_all(['h2', 'h3', 'table'])
        current_dataset = category_name
        for element in elements:
            if element.name in ['h2', 'h3']: 
                text = element.text.strip()
                if len(text) > 3: current_dataset = text.lower()
            elif element.name == 'table':
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                if not table_headers or not any(x in table_headers for x in ['type', 'description', 'data_type']): continue
                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) < len(table_headers): continue 
                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_): entry[header] = columns_[i].text.strip()
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    entry = {header_map.get(k, k): v for k, v in entry.items()}
                    if 'column_name' in entry and entry['column_name']:
                        entry['dataset_name'] = current_dataset
                        entry['category'] = category_name
                        entry['url'] = url 
                        data.append(entry)
        return data
    except Exception: return []

def scrape_and_save_from_list(url_list):
    all_data = []
    progress_bar = st.progress(0, "Initializing Scraper...")
    
    def get_category_from_url(url):
        return re.sub(r'^\d+\s*', '', os.path.basename(url).split('?')[0].replace('-data-sets', '').replace('-', ' ')).lower()

    with ThreadPoolExecutor(max_workers=10) as executor:
        args = [(url, get_category_from_url(url)) for url in url_list]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception: pass
            progress_bar.progress((i + 1) / len(url_list), f"Scraping {i+1}/{len(url_list)}...")
    progress_bar.empty()
    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data)
    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key', 'url']
    for col in expected_cols:
        if col not in df.columns: df[col] = ''
    df = df.fillna('')
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def load_data():
    if os.path.exists('dataset_metadata.csv'):
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()

@st.cache_data
def find_pk_fk_joins(df, selected_datasets=None):
    """
    Finds joins. 
    If selected_datasets is provided, filters for those.
    If None, returns ALL relationships (for global map).
    """
    if df.empty: return pd.DataFrame()
    
    pks = df[df['is_primary_key'] == True]
    fks = df[df['is_foreign_key'] == True]
    
    if selected_datasets:
        fks = fks[fks['dataset_name'].isin(selected_datasets)]
    
    if pks.empty or fks.empty: return pd.DataFrame()
    
    # Merge on column name
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    
    # Filter out self-joins if desired (though sometimes valid in hierarchy)
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    if joins.empty: return pd.DataFrame()
    
    result = joins[['dataset_name_fk', 'column_name', 'dataset_name_pk', 'category_pk', 'category_fk']]
    result.columns = ['Source Dataset', 'Join Column', 'Target Dataset', 'Target Category', 'Source Category']
    return result.drop_duplicates().reset_index(drop=True)

# ==========================================
# GRAPHING & VISUALIZATION LOGIC
# ==========================================

@st.cache_data
def build_constellation_map(df, show_connections=False):
    """
    Builds the 'Galaxy' map where Categories are central stars and Datasets orbit them.
    """
    categories = sorted(df['category'].unique())
    datasets = df[['dataset_name', 'category', 'description']].drop_duplicates('dataset_name')
    
    G = nx.Graph()
    
    # 1. Create Layout Coordinates Manually for "Solar System" effect
    pos = {}
    node_colors = []
    node_sizes = []
    node_texts = []
    node_types = []
    
    center_x, center_y = 0, 0
    cat_radius = 10 # Radius of the category ring
    ds_radius = 2   # Radius of datasets around their category
    
    # Add Categories
    angle_step = 2 * math.pi / len(categories) if categories else 1
    
    for i, cat in enumerate(categories):
        angle = i * angle_step
        cx = center_x + cat_radius * math.cos(angle)
        cy = center_y + cat_radius * math.sin(angle)
        pos[cat] = (cx, cy)
        
        G.add_node(cat, type='category')
        node_colors.append('#FFD700') # Gold for categories
        node_sizes.append(30)
        node_texts.append(f"<b>Category:</b> {cat}")
        node_types.append('category')
        
        # Add Datasets for this category
        cat_datasets = datasets[datasets['category'] == cat]
        ds_count = len(cat_datasets)
        if ds_count > 0:
            ds_angle_step = 2 * math.pi / ds_count
            for j, (_, row) in enumerate(cat_datasets.iterrows()):
                ds_name = row['dataset_name']
                ds_angle = j * ds_angle_step
                dx = cx + ds_radius * math.cos(ds_angle)
                dy = cy + ds_radius * math.sin(ds_angle)
                
                # Jitter slightly to avoid perfect overlap if logic fails
                pos[ds_name] = (dx, dy)
                G.add_node(ds_name, type='dataset')
                G.add_edge(cat, ds_name, color='#444444', width=1) # Link to category
                
                node_colors.append('#00CCFF') # Blue for datasets
                node_sizes.append(10)
                desc = str(row['description'])[:100] + "..." if len(str(row['description'])) > 100 else str(row['description'])
                node_texts.append(f"<b>Dataset:</b> {ds_name}<br><i>{desc}</i>")
                node_types.append('dataset')

    # Add PK/FK connections if requested
    edge_x, edge_y, edge_colors = [], [], []
    
    # Draw structural edges (Category -> Dataset)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_colors.append('#333333') # Faint lines for hierarchy

    # Draw Logic Edges (PK/FK)
    if show_connections:
        joins = find_pk_fk_joins(df) # Get all joins
        for _, row in joins.iterrows():
            s, t = row['Source Dataset'], row['Target Dataset']
            if s in pos and t in pos:
                x0, y0 = pos[s]
                x1, y1 = pos[t]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_colors.append('rgba(255, 255, 255, 0.1)') # Very faint white for connections

    # Create Plotly Traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#555'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_texts,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=1,
            line_color='white'
        )
    )
    
    # Add text labels for categories only (to reduce clutter)
    cat_x = [pos[n][0] for n in G.nodes() if G.nodes[n]['type'] == 'category']
    cat_y = [pos[n][1] for n in G.nodes() if G.nodes[n]['type'] == 'category']
    cat_text = [n for n in G.nodes() if G.nodes[n]['type'] == 'category']
    
    text_trace = go.Scatter(
        x=cat_x, y=cat_y,
        mode='text',
        text=cat_text,
        textposition="top center",
        textfont=dict(color='#FFD700', size=12, family="sans serif"),
        hoverinfo='none'
    )

    fig = go.Figure(data=[edge_trace, node_trace, text_trace],
             layout=go.Layout(
                title='<br>Full Dataset Constellation Map',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                plot_bgcolor='rgb(10,10,10)', # Dark background space theme
                paper_bgcolor='rgb(10,10,10)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=750
             ))
    return fig

def build_focus_graph(selected_datasets, join_data, mode):
    G = nx.DiGraph()
    if mode == 'Between selected (Focused)':
        for ds in selected_datasets: G.add_node(ds, type='focus')
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s, t = row['Source Dataset'], row['Target Dataset']
                if s in selected_datasets and t in selected_datasets:
                    G.add_edge(s, t, label=row['Join Column'])
    else: # Discovery
        for ds in selected_datasets: G.add_node(ds, type='focus')
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s, t = row['Source Dataset'], row['Target Dataset']
                if s in selected_datasets:
                    if not G.has_node(t): G.add_node(t, type='neighbor')
                    G.add_edge(s, t, label=row['Join Column'])

    if G.number_of_nodes() == 0: return None
    
    pos = nx.spring_layout(G, k=0.7, iterations=60)
    edge_x, edge_y, label_x, label_y, label_text = [], [], [], [], []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        label_x.append((x0 + x1) / 2)
        label_y.append((y0 + y1) / 2)
        label_text.append(data.get('label', '?'))

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    label_trace = go.Scatter(
        x=label_x, y=label_y, mode='text', text=label_text,
        textfont=dict(color='#00CCFF', size=11, family="monospace"), hoverinfo='none'
    )
    
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        n_type = G.nodes[node].get('type', 'focus')
        node_color.append('#FF4B4B' if n_type == 'focus' else '#1F77B4')
        node_size.append(25 if n_type == 'focus' else 15)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="top center",
        marker=dict(color=node_color, size=node_size, line=dict(width=2, color='white'))
    )
    
    return go.Figure(data=[edge_trace, label_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    ))

# ==========================================
# MAIN APP LOGIC
# ==========================================

# Load Data
df = load_data()

# SIDEBAR
with st.sidebar:
    st.title("BDS Explorer")
    
    # 1. Scraper
    with st.expander("🛠️ Data Scraper", expanded=False):
        if not df.empty:
            count_ds = df['dataset_name'].nunique()
            st.success(f"Loaded {count_ds} datasets.")
        else:
            st.error("No data found.")
        
        st.caption("Update Metadata from D2L")
        pasted_text = st.text_area("URLs", height=60, value=DEFAULT_URLS)
        
        if st.button("Scrape URLs", type="primary"):
            url_list = parse_urls_from_text_area(pasted_text)
            with st.spinner(f"Scraping {len(url_list)} pages..."):
                df_new = scrape_and_save_from_list(url_list)
                st.session_state['scrape_msg'] = f"Success: {df_new['dataset_name'].nunique()} Datasets"
                st.rerun()

    # 2. AI Login
    st.divider()
    if not st.session_state['authenticated']:
        st.caption("🔒 AI Features Locked")
        st.text_input("Password", type="password", key="password_input", on_change=perform_login)
        if st.session_state['auth_error']: st.error("Incorrect password.")
    else:
        st.success("🔓 AI Features Unlocked")
        if st.button("Logout", type="secondary"): logout(); st.rerun()

    # 3. Credits
    st.divider()
    st.markdown("### Related Tools")
    st.link_button("🔎 CSV Query Tool", "https://csvexpl0rer.streamlit.app/")

# MAIN CONTENT
if 'scrape_msg' in st.session_state:
    st.success(st.session_state['scrape_msg'])
    del st.session_state['scrape_msg']

if df.empty:
    st.title("Brightspace Datasets Explorer")
    st.warning("👈 Please use the sidebar scraper to load data first.")
    st.stop()

# TABS ARCHITECTURE
tab_focus, tab_map, tab_ai = st.tabs(["🔍 Focus Explorer", "🌌 The Galaxy Map", "🤖 AI Analyst"])

# ----------------------------------------------------
# TAB 1: FOCUS EXPLORER (Original Functionality)
# ----------------------------------------------------
with tab_focus:
    st.subheader("Filter & Analyze Specific Datasets")
    
    # Selection Controls
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        select_mode = st.radio("Selection Mode:", ["List All", "By Category"], horizontal=True, label_visibility="collapsed")
        
    with col_sel2:
        if st.button("Clear All Selections"): clear_all_selections()

    selected_datasets = []
    
    if select_mode == "By Category":
        all_cats = sorted(df['category'].unique())
        selected_cats = st.multiselect("Select Categories:", all_cats)
        if selected_cats:
            for cat in selected_cats:
                cat_ds = sorted(df[df['category'] == cat]['dataset_name'].unique())
                s = st.multiselect(f"📦 {cat}", cat_ds, key=f"sel_{cat}")
                selected_datasets.extend(s)
    else:
        all_ds = sorted(df['dataset_name'].unique())
        selected_datasets = st.multiselect("Search All Datasets:", all_ds, key="global_search")

    # Column Search
    with st.expander("🔎 Find containing column...", expanded=False):
        col_search = st.text_input("Column Name", placeholder="e.g. OrgUnitId")
        if col_search:
            matches = df[df['column_name'].astype(str).str.contains(col_search, case=False)]
            if not matches.empty:
                st.dataframe(matches[['dataset_name', 'column_name', 'category']].drop_duplicates(), hide_index=True)
            else:
                st.warning("No matches found.")

    st.divider()

    if selected_datasets:
        st.markdown(f"### Analysis of **{len(selected_datasets)}** Datasets")
        
        # 1. Schema
        with st.expander("📋 View Schema", expanded=False):
            subset = df[df['dataset_name'].isin(selected_datasets)]
            st.dataframe(subset[['dataset_name', 'column_name', 'data_type', 'description', 'key']], use_container_width=True, hide_index=True)

        # 2. Graph
        col_viz, col_sql = st.columns([3, 2])
        
        with col_viz:
            st.caption("Relationship Graph")
            graph_mode = st.radio("Mode", ['Between selected (Focused)', 'From selected (Discovery)'], horizontal=True, label_visibility="collapsed")
            join_data = find_pk_fk_joins(df, selected_datasets)
            fig = build_focus_graph(selected_datasets, join_data, graph_mode)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("No relationships found among selections.")

        # 3. SQL
        with col_sql:
            st.caption("Generated SQL Join")
            if len(selected_datasets) >= 2 and fig:
                base_table = selected_datasets[0]
                aliases = {name: f"t{i+1}" for i, name in enumerate(selected_datasets)}
                sql = [f"SELECT TOP 100", f"    {aliases[base_table]}.*"]
                sql.append(f"FROM {base_table} {aliases[base_table]}")
                
                # Simple heuristic join generation based on graph edges
                # (Re-calculating logic briefly for SQL text)
                joined = {base_table}
                G_temp = nx.Graph() 
                if not join_data.empty:
                    for _, r in join_data.iterrows(): G_temp.add_edge(r['Source Dataset'], r['Target Dataset'], label=r['Join Column'])
                
                # Traverse selection
                for i in range(1, len(selected_datasets)):
                    curr = selected_datasets[i]
                    # Find connection to anything in 'joined'
                    connected = False
                    for existing in joined:
                        if G_temp.has_edge(curr, existing):
                            col = G_temp[curr][existing]['label']
                            sql.append(f"LEFT JOIN {curr} {aliases[curr]} ON {aliases[existing]}.{col} = {aliases[curr]}.{col}")
                            joined.add(curr)
                            connected = True
                            break
                    if not connected:
                        sql.append(f"-- ⚠️ No direct FK found for {curr}, manual join needed")
                        sql.append(f"CROSS JOIN {curr} {aliases[curr]}")
                
                st.code("\n".join(sql), language="sql")
            else:
                st.info("Select 2+ connected datasets to generate SQL.")

# ----------------------------------------------------
# TAB 2: THE GALAXY MAP (New Feature)
# ----------------------------------------------------
with tab_map:
    st.subheader("🌌 The Dataset Constellation")
    st.markdown("Interactive map of the entire data ecosystem. **Scroll to Zoom, Drag to Pan.**")
    
    col_toggles, col_metrics = st.columns([1, 3])
    with col_toggles:
        show_conns = st.checkbox("Show Inter-Dataset Links", value=False, help="Draws lines between PK/FKs. Can be messy!")
    
    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Datasets", df['dataset_name'].nunique())
        m2.metric("Total Columns", len(df))
        m3.metric("Categories", df['category'].nunique())

    # Build the big map
    # We pass the whole dataframe to the map builder
    fig_map = build_constellation_map(df, show_connections=show_conns)
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.info("💡 **Tip:** Hover over blue dots to see dataset descriptions. Gold dots are Categories.")

# ----------------------------------------------------
# TAB 3: AI ANALYST (Existing Logic)
# ----------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI Data Assistant")
    
    if not st.session_state['authenticated']:
        st.warning("Please log in via the Sidebar to use AI features.")
    else:
        # Settings
        with st.expander("⚙️ Model Settings", expanded=False):
            ai_provider = st.radio("Provider", ["OpenAI (GPT-4o)", "xAI (Grok)"], horizontal=True)
            if "OpenAI" in ai_provider:
                api_key_name = "openai_api_key"
                base_url = None 
                model_name = "gpt-4o"
            else:
                api_key_name = "xai_api_key"
                base_url = "https://api.x.ai/v1"
                model_name = "grok-2-1212"
            
            api_key = st.secrets.get(api_key_name)
            if not api_key: api_key = st.text_input(f"Enter {api_key_name}", type="password")
            
            use_full_context = st.checkbox("Use Full Database Context", value=False, help="Sends ALL table names (tokens expensive). If unchecked, only sends selected datasets.")

        # Chat Interface
        if "messages" not in st.session_state: st.session_state.messages = []
        
        # Display History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
            
        # Input
        if prompt := st.chat_input("Ask about joins, specific columns, or data definitions..."):
            if not api_key:
                st.error("API Key missing.")
                st.stop()
                
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner(f"Consulting {model_name}..."):
                    try:
                        # Context Building
                        if use_full_context:
                            # Optimized context for full DB
                            schema_summary = df.groupby('dataset_name').apply(lambda x: f"{x.name}: {', '.join(x['column_name'].tolist())}").str.cat(sep="\n")
                            context = f"Full Schema Summary:\n{schema_summary}"
                        else:
                            # Detailed context for selection
                            target_df = df[df['dataset_name'].isin(selected_datasets)] if selected_datasets else df.head(50)
                            context = target_df.to_csv(index=False)
                        
                        system_msg = f"""
                        You are a Brightspace Data Expert.
                        Context: {context[:50000]} (Truncated if too large)
                        
                        Rules:
                        1. If the user asks for SQL, write standard SQL.
                        2. Identify Primary Keys (PK) and Foreign Keys (FK).
                        3. Be concise.
                        """
                        
                        client = openai.OpenAI(api_key=api_key, base_url=base_url)
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        reply = response.choices[0].message.content
                        
                        # Cost tracking (approximate)
                        if hasattr(response, 'usage'):
                            st.session_state['total_tokens'] += response.usage.total_tokens
                            
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        
                    except Exception as e:
                        st.error(f"AI Error: {e}")
