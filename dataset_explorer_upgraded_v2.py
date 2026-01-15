# streamlit run dataset_explorer_v17.py
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
# 1. app configuration & styling
# =============================================================================

st.set_page_config(
    page_title="Brightspace Data Universe",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# configure structured logging
logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# suppress insecure request warnings for d2l scrapers
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# apply pro ui css
st.markdown("""
<style>
    /* metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #1E232B;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] { color: #8B949E; }
    div[data-testid="stMetricValue"] { color: #58A6FF; font-size: 24px; }
    
    /* tabs styling */
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
    
    /* code blocks */
    .stCode { font-family: 'Fira Code', monospace; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. pricing registry (ai models)
# =============================================================================

# define supported models and their costs (usd per 1m tokens)
PRICING_REGISTRY = {
    # xai models
    "grok-2-1212":             {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-2-vision-1212":      {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-3":                  {"in": 3.00, "out": 15.00, "provider": "xAI"},
    "grok-3-mini":             {"in": 0.30, "out": 0.50,  "provider": "xAI"},
    "grok-4-0709":             {"in": 3.00, "out": 15.00, "provider": "xAI"},
    "grok-4-1-fast-reasoning": {"in": 0.20, "out": 0.50,  "provider": "xAI"},
    
    # openai models
    "gpt-4o":                  {"in": 2.50, "out": 10.00, "provider": "OpenAI"},
    "gpt-4o-mini":             {"in": 0.15, "out": 0.60,  "provider": "OpenAI"},
    "gpt-5-mini":              {"in": 0.25, "out": 2.00,  "provider": "OpenAI"},
    "gpt-5.1":                 {"in": 1.25, "out": 10.00, "provider": "OpenAI"},
    "gpt-5.2":                 {"in": 1.75, "out": 14.00, "provider": "OpenAI"},
}

# =============================================================================
# 3. session state management
# =============================================================================

def init_session_state():
    """initializes streamlit session state variables safely."""
    defaults = {
        'authenticated': False,
        'auth_error': False,
        'messages': [],
        'view_mode': 'Universe Map',
        'scrape_status': None,
        'total_cost': 0.0,
        'total_tokens': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# 4. authentication logic (case insensitive)
# =============================================================================

def get_secret(key_name: str) -> Optional[str]:
    """retrieves a secret, checking both lowercase and uppercase variations."""
    return st.secrets.get(key_name) or st.secrets.get(key_name.upper())

def perform_login():
    """verifies password against streamlit secrets or allows dev mode."""
    pwd_secret = get_secret("app_password")
    
    # dev mode: if no secret is configured, allow access
    if not pwd_secret:
        logger.warning("No password configured. Allowing open access.")
        st.session_state['authenticated'] = True
        return

    # production mode: check input
    if st.session_state.get("password_input") == pwd_secret:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
    else:
        st.session_state['auth_error'] = True
        st.session_state['authenticated'] = False

def logout():
    """clears authentication state."""
    st.session_state['authenticated'] = False
    st.session_state['password_input'] = ""
    st.session_state['messages'] = []  # clear chat on logout

# =============================================================================
# 5. data layer (scraper & storage)
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
    parses a d2l knowledge base page to extract dataset definitions.
    returns a list of dictionaries representing columns.
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
        
        # logic: headers (h2/h3) denote the dataset name, following table is schema
        elements = soup.find_all(['h2', 'h3', 'table'])
        for element in elements:
            if element.name in ['h2', 'h3']:
                text = element.text.strip()
                if len(text) > 3: 
                    current_dataset = text.lower()
                    
            elif element.name == 'table':
                # normalize headers
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                
                # validation: ensure this is a metadata table
                if not table_headers or not any(x in table_headers for x in ['type', 'description', 'data_type']):
                    continue
                
                # extract rows
                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) < len(table_headers): 
                        continue
                    
                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_): 
                            entry[header] = columns_[i].text.strip()
                    
                    # normalize keys
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
    orchestrates the scraping process using threadpoolexecutor.
    saves the result to 'dataset_metadata.csv'.
    """
    all_data = []
    progress_bar = st.progress(0, "Initializing Scraper...")
    
    # helper to clean urls
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

    # create dataframe
    df = pd.DataFrame(all_data)
    df = df.fillna('')
    
    # clean up text
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    
    # ensure 'key' column exists before checking pk/fk
    if 'key' not in df.columns:
        df['key'] = ''
    
    # logic flags for joins
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    
    # persist to csv
    df.to_csv('dataset_metadata.csv', index=False)
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """loads the csv from disk if it exists and is valid."""
    if os.path.exists('dataset_metadata.csv') and os.path.getsize('dataset_metadata.csv') > 10:
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()


@st.cache_data
def get_possible_joins(df: pd.DataFrame) -> pd.DataFrame:
    """calculates all possible join conditions based on pk/fk naming."""
    if df.empty:
        return pd.DataFrame()
    
    # ensure required columns exist
    if 'is_primary_key' not in df.columns or 'is_foreign_key' not in df.columns:
        return pd.DataFrame()
    
    pks = df[df['is_primary_key'] == True]
    fks = df[df['is_foreign_key'] == True]
    
    if pks.empty or fks.empty:
        return pd.DataFrame()
    
    # merge where foreign key column name matches primary key column name
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    
    # exclude self-joins (joining a table to itself)
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    return joins


# =============================================================================
# 6. visualization engine (orbital map & focused graph)
# =============================================================================

def create_focused_graph(df: pd.DataFrame, selected_datasets: List[str]) -> go.Figure:
    """
    creates a graph showing only connections between the selected datasets.
    complements the discovery-focused orbital map.
    """
    if len(selected_datasets) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Select 2+ datasets for Focused view", 
            showarrow=False, 
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    G = nx.DiGraph()
    joins = get_possible_joins(df)
    
    # add only selected datasets as nodes
    for ds in selected_datasets:
        G.add_node(ds)
    
    # add only edges between selected datasets
    if not joins.empty:
        for _, r in joins.iterrows():
            src = r['dataset_name_fk']
            tgt = r['dataset_name_pk']
            if src in selected_datasets and tgt in selected_datasets:
                G.add_edge(src, tgt, key=r['column_name'])
    
    # use spring layout for positioning
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # build edge traces
    edge_x = []
    edge_y = []
    edge_labels_x = []
    edge_labels_y = []
    edge_labels_text = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # store edge label position and text
        edge_labels_x.append((x0 + x1) / 2)
        edge_labels_y.append((y0 + y1) / 2)
        edge_labels_text.append(data.get('key', ''))
    
    # build node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())
    
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y, 
        mode='lines', 
        line=dict(width=2, color='#00CCFF'),
        hoverinfo='none'
    )
    
    edge_label_trace = go.Scatter(
        x=edge_labels_x,
        y=edge_labels_y,
        mode='text',
        text=edge_labels_text,
        textfont=dict(size=10, color='#00FF00', family='monospace'),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        mode='markers+text',
        text=node_text, 
        textposition='top center',
        textfont=dict(size=12, color='white'),
        marker=dict(size=25, color='#FF4B4B', line=dict(width=2, color='white')),
        hoverinfo='text',
        hovertext=node_text
    )
    
    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(b=20, l=20, r=20, t=20)
        )
    )
    return fig


@st.cache_data
def create_orbital_map(df: pd.DataFrame, target_node: str = None) -> go.Figure:
    """
    generates the 'solar system' map with deterministic geometry.
    categories are suns. datasets are planets.
    uses strict deterministic math (orbit) instead of physics (spring).
    """
    if df.empty:
        return go.Figure()
    
    # 1. prepare data - ensure required columns exist
    categories = sorted(df['category'].unique())
    
    required_cols = ['dataset_name', 'category']
    optional_cols = ['description']
    cols_to_use = required_cols + [c for c in optional_cols if c in df.columns]
    datasets = df[cols_to_use].drop_duplicates('dataset_name')
    
    # 2. define layout physics (orbit)
    pos = {}
    center_x = 0
    center_y = 0
    orbit_radius_cat = 20  # distance of category from center
    
    # place categories in a large ring
    cat_step = 2 * math.pi / len(categories) if categories else 1
    
    # trace containers
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_line_width = []
    node_line_color = []
    cat_x = []
    cat_y = []
    cat_text = []  # category labels
    
    # 3. determine highlights (hud logic)
    active_edges = []
    active_neighbors = set()
    
    if target_node:
        joins = get_possible_joins(df)
        
        if not joins.empty:
            # find outgoing neighbors
            out_ = joins[joins['dataset_name_fk'] == target_node]
            for _, r in out_.iterrows():
                active_edges.append((target_node, r['dataset_name_pk'], r['column_name']))
                active_neighbors.add(r['dataset_name_pk'])
                
            # find incoming neighbors
            in_ = joins[joins['dataset_name_pk'] == target_node]
            for _, r in in_.iterrows():
                active_edges.append((r['dataset_name_fk'], target_node, r['column_name']))
                active_neighbors.add(r['dataset_name_fk'])

    # 4. build nodes
    for i, cat in enumerate(categories):
        # category position
        angle = i * cat_step
        cx = center_x + orbit_radius_cat * math.cos(angle)
        cy = center_y + orbit_radius_cat * math.sin(angle)
        pos[cat] = (cx, cy)
        
        # add category node
        node_x.append(cx)
        node_y.append(cy)
        node_text.append(f"Category: {cat}")
        
        # visuals: categories are gold suns
        is_dim = (target_node is not None)
        node_color.append('rgba(255, 215, 0, 0.2)' if is_dim else 'rgba(255, 215, 0, 1)')
        node_size.append(35)
        node_line_width.append(0)
        node_line_color.append('rgba(0,0,0,0)')
        
        # add category label
        cat_x.append(cx)
        cat_y.append(cy + 3)
        cat_text.append(cat)

        # dataset positions (orbiting the category)
        cat_ds = datasets[datasets['category'] == cat]
        ds_count = len(cat_ds)
        
        if ds_count > 0:
            # dynamic radius based on dataset count to prevent overlap
            min_radius = 3
            radius_per_node = 0.5
            ds_radius = min_radius + (ds_count * radius_per_node / (2 * math.pi))
            ds_step = 2 * math.pi / ds_count
            
            for j, (_, row) in enumerate(cat_ds.iterrows()):
                ds_name = row['dataset_name']
                ds_angle = j * ds_step
                dx = cx + ds_radius * math.cos(ds_angle)
                dy = cy + ds_radius * math.sin(ds_angle)
                pos[ds_name] = (dx, dy)
                
                # add dataset node position
                node_x.append(dx)
                node_y.append(dy)
                
                # visual logic for dataset based on selection state
                if target_node:
                    if ds_name == target_node:
                        # target node - increased visibility
                        node_color.append('#00FF00')  # bright green
                        node_size.append(50)  # large size
                        node_line_width.append(5)
                        node_line_color.append('white')  # thick border
                    elif ds_name in active_neighbors:
                        # neighbor node
                        node_color.append('#00CCFF')  # blue
                        node_size.append(15)
                        node_line_width.append(1)
                        node_line_color.append('white')
                    else:
                        # inactive node
                        node_color.append('rgba(50,50,50,0.3)')
                        node_size.append(8)
                        node_line_width.append(0)
                        node_line_color.append('rgba(0,0,0,0)')
                else:
                    # default state - no target selected
                    node_color.append('#00CCFF')
                    node_size.append(10)
                    node_line_width.append(1)
                    node_line_color.append('rgba(255,255,255,0.3)')

                # build hover text with safe access to description
                desc_short = str(row.get('description', ''))[:80]
                if desc_short:
                    desc_short += "..."
                    hover_text = f"<b>{ds_name}</b><br>{desc_short}"
                else:
                    hover_text = f"<b>{ds_name}</b>"
                node_text.append(hover_text)

    # 5. build edges (lines)
    edge_x = []
    edge_y = []
    label_x = []
    label_y = []
    label_text = []

    for s, t, k in active_edges:
        if s in pos and t in pos:
            x0, y0 = pos[s]
            x1, y1 = pos[t]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            label_text.append(k)

    # 6. create traces
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y, 
        mode='lines', 
        line=dict(width=2, color='#00FF00'), 
        hoverinfo='none'
    )
    
    label_trace = go.Scatter(
        x=label_x, 
        y=label_y, 
        mode='text', 
        text=label_text,
        textfont=dict(color='#00FF00', size=11, family="monospace"),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        mode='markers', 
        hoverinfo='text', 
        hovertext=node_text,
        marker=dict(
            color=node_color, 
            size=node_size, 
            line=dict(width=node_line_width, color=node_line_color)
        )
    )
    
    cat_label_trace = go.Scatter(
        x=cat_x, 
        y=cat_y, 
        mode='text', 
        text=cat_text,
        textfont=dict(color='gold', size=10), 
        hoverinfo='none'
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
# 7. sql builder engine
# =============================================================================

def generate_manual_sql(selected_datasets: List[str], df: pd.DataFrame) -> str:
    """
    generates a deterministic sql join query based on the graph relationships.
    uses a 'greedy' approach: connect each new table to the existing joined cluster.
    """
    if len(selected_datasets) < 2:
        return "-- please select at least 2 datasets to generate a join."
    
    # 1. build the full connection graph
    G_full = nx.Graph()
    joins = get_possible_joins(df)
    
    if not joins.empty:
        for _, r in joins.iterrows():
            # store the join key as an edge attribute
            G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    # 2. initialize query
    base_table = selected_datasets[0]
    # create simple aliases (t1, t2, etc.)
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}
    
    sql_lines = [f"SELECT TOP 100", f"    {aliases[base_table]}.*"]
    sql_lines.append(f"FROM {base_table} {aliases[base_table]}")
    
    # set of tables already added to the query
    joined_tables = {base_table}
    
    # 3. iterate through remaining tables
    remaining_tables = selected_datasets[1:]
    
    for current_table in remaining_tables:
        found_connection = False
        
        # check if current_table connects to any table already in the query
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
            # fallback for unconnected tables
            sql_lines.append(
                f"CROSS JOIN {current_table} {aliases[current_table]} "
                f"-- ⚠️ no direct relationship found in metadata"
            )
            joined_tables.add(current_table)
            
    return "\n".join(sql_lines)


# =============================================================================
# 8. helper functions for relationship analysis
# =============================================================================

def show_relationship_summary(df: pd.DataFrame, dataset_name: str):
    """shows quick stats about a dataset's connectivity."""
    joins = get_possible_joins(df)
    
    if joins.empty:
        outgoing = 0
        incoming = 0
    else:
        outgoing = len(joins[joins['dataset_name_fk'] == dataset_name])
        incoming = len(joins[joins['dataset_name_pk'] == dataset_name])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Outgoing FKs", outgoing, help="This dataset references other tables")
    col2.metric("Incoming FKs", incoming, help="Other tables reference this one")
    col3.metric("Total Connections", outgoing + incoming)


def find_join_path(df: pd.DataFrame, source_dataset: str, target_dataset: str) -> Optional[List[str]]:
    """finds the shortest path of joins between two datasets."""
    joins = get_possible_joins(df)
    
    if joins.empty:
        return None
    
    G = nx.Graph()
    for _, r in joins.iterrows():
        G.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])
    
    try:
        path = nx.shortest_path(G, source_dataset, target_dataset)
        return path
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None


def create_relationship_matrix(df: pd.DataFrame) -> go.Figure:
    """creates a heatmap showing which datasets connect to which."""
    joins = get_possible_joins(df)
    datasets = sorted(df['dataset_name'].unique())
    
    # create adjacency matrix
    matrix = pd.DataFrame(0, index=datasets, columns=datasets)
    
    if not joins.empty:
        for _, r in joins.iterrows():
            src = r['dataset_name_fk']
            tgt = r['dataset_name_pk']
            if src in matrix.index and tgt in matrix.columns:
                matrix.loc[src, tgt] += 1
    
    fig = px.imshow(
        matrix, 
        labels=dict(x="Target (PK)", y="Source (FK)", color="Connections"),
        color_continuous_scale="Blues"
    )
    return fig


# =============================================================================
# 9. view controllers (modular ui)
# =============================================================================

def render_sidebar(df: pd.DataFrame) -> str:
    """renders the sidebar navigation and admin tools."""
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
        
        # scraper accordion
        with st.expander("⚙️ Admin / Update Data"):
            txt = st.text_area("URLs", value=DEFAULT_URLS, height=100)
            if st.button("Run Scraper"):
                urls = [u.strip() for u in txt.split('\n') if u.startswith('http')]
                scrape_and_save(urls)
                load_data.clear()
                st.rerun()
        
        # authentication status
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
    """renders the interactive galaxy map with both discovery and focused modes."""
    st.subheader("Interactive Data Map")
    
    # control row
    col_search, col_stats = st.columns([3, 1])
    
    with col_search:
        all_ds = sorted(df['dataset_name'].unique())
        
        # mode selector for exploration type
        explore_mode = st.radio(
            "Exploration Mode",
            ["Single Target (Discovery)", "Multi-Select (Focused)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if explore_mode == "Single Target (Discovery)":
            target = st.selectbox("🎯 Target Dataset", ["None"] + all_ds)
            target_val = None if target == "None" else target
            multi_targets = None
        else:
            multi_targets = st.multiselect("Select 2+ Datasets", all_ds)
            target_val = None
    
    with col_stats:
        if target_val:
            row_cnt = len(df[df['dataset_name'] == target_val])
            st.metric("Column Count", row_cnt)
        elif multi_targets:
            st.metric("Selected Datasets", len(multi_targets))
        else:
            st.metric("Total Datasets", df['dataset_name'].nunique())

    # map and details row
    c1, c2 = st.columns([3, 1])
    
    with c1:
        # render the appropriate graph based on mode
        if explore_mode == "Single Target (Discovery)":
            fig = create_orbital_map(df, target_val)
        else:
            # multi-select mode uses the focused graph
            if multi_targets and len(multi_targets) >= 2:
                fig = create_focused_graph(df, multi_targets)
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please select 2+ datasets for Focused view",
                    showarrow=False,
                    font=dict(size=14, color='gray')
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        # right panel adapts based on exploration mode
        if explore_mode == "Single Target (Discovery)" and target_val:
            # single target view - show dataset details
            st.markdown(f"### {target_val}")
            
            # show relationship summary metrics
            show_relationship_summary(df, target_val)
            
            ds_data = df[df['dataset_name'] == target_val].iloc[0]
            st.caption(f"Category: {ds_data['category']}")
            
            # show documentation link if url exists
            if 'url' in ds_data and ds_data['url']: 
                st.link_button("View Documentation", ds_data['url'])
            
            # show pk/fk information
            subset = df[df['dataset_name'] == target_val]
            
            if 'is_primary_key' in subset.columns:
                pks = subset[subset['is_primary_key']]['column_name'].tolist()
                if pks:
                    st.markdown(f"🔑 **PK:** {', '.join(pks)}")
            
            if 'is_foreign_key' in subset.columns:
                fks = subset[subset['is_foreign_key']]['column_name'].tolist()
                if fks:
                    st.markdown(f"🔗 **FK:** {', '.join(fks)}")
            
            with st.expander("Full Schema", expanded=True):
                display_cols = ['column_name', 'data_type']
                available_cols = [c for c in display_cols if c in subset.columns]
                if available_cols:
                    st.dataframe(
                        subset[available_cols], 
                        hide_index=True,
                        use_container_width=True
                    )
                    
        elif explore_mode == "Multi-Select (Focused)" and multi_targets:
            # multi-select view - show comparison details
            st.markdown(f"### Comparing {len(multi_targets)} Datasets")
            
            # show common columns between selected datasets
            if len(multi_targets) >= 2:
                sets = [set(df[df['dataset_name'] == ds]['column_name']) for ds in multi_targets]
                common = sets[0].intersection(*sets[1:])
                if common:
                    st.success(f"**Shared Columns:** {', '.join(sorted(common))}")
                else:
                    st.warning("No shared column names found.")
            
            # show each selected dataset with expandable schema
            for ds in multi_targets:
                with st.expander(ds, expanded=False):
                    subset = df[df['dataset_name'] == ds]
                    display_cols = ['column_name', 'data_type']
                    available_cols = [c for c in display_cols if c in subset.columns]
                    if available_cols:
                        st.dataframe(
                            subset[available_cols], 
                            hide_index=True,
                            use_container_width=True
                        )
                        
            # show generated sql for the multi-select
            if len(multi_targets) >= 2:
                st.divider()
                st.markdown("#### ⚡ Generated SQL")
                sql_code = generate_manual_sql(multi_targets, df)
                st.code(sql_code, language="sql")
        else:
            # default state - no selection made
            st.info("Select a dataset to view details, or select 2+ datasets for comparison.")


def render_schema_view(df: pd.DataFrame):
    """renders the schema search and manual sql builder."""
    st.subheader("🔎 Schema Search & SQL Builder")
    
    c_search, c_sel = st.columns(2)
    
    with c_search:
        search = st.text_input("Find Column", placeholder="e.g. OrgUnitId")
        if search:
            hits = df[df['column_name'].str.contains(search, case=False, na=False)]
            if not hits.empty:
                display_cols = ['dataset_name', 'column_name', 'data_type', 'category']
                available_cols = [c for c in display_cols if c in hits.columns]
                st.dataframe(
                    hits[available_cols], 
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
            display_cols = ['dataset_name', 'column_name', 'data_type', 'is_primary_key', 'is_foreign_key']
            available_cols = [c for c in display_cols if c in sub.columns]
            st.dataframe(
                sub[available_cols], 
                use_container_width=True
            )
        
        with c_sql:
            st.markdown("#### ⚡ Generated SQL")
            sql_code = generate_manual_sql(sel_ds, df)
            st.code(sql_code, language="sql")


def render_ai_view(df: pd.DataFrame):
    """renders the ai chat interface with dynamic cost logic."""
    st.subheader("🤖 Data Architect Assistant")
    
    if not st.session_state['authenticated']:
        st.warning("🔒 Login required to use AI features. Please enter password in sidebar.")
        st.stop()
    
    c_set, c_chat = st.columns([1, 3])
    
    with c_set:
        st.markdown("#### Settings")
        
        # 1. select model directly (provider is inferred)
        model_options = list(PRICING_REGISTRY.keys())
        selected_model_id = st.selectbox("Select Model", model_options, index=3)  # default to grok-3-mini
        
        # get metadata for selected model
        model_info = PRICING_REGISTRY[selected_model_id]
        provider = model_info['provider']
        
        st.caption(f"Provider: **{provider}**")
        st.caption(f"Input: ${model_info['in']:.2f}/1M | Output: ${model_info['out']:.2f}/1M")
        
        # 2. key check (dynamic based on provider)
        key_name = "openai_api_key" if provider == "OpenAI" else "xai_api_key"
        secret_key = get_secret(key_name)
        
        if secret_key:
            st.success(f"✅ {provider} Key Loaded")
            api_key = secret_key
        else:
            api_key = st.text_input(f"{provider} API Key", type="password")
            
        use_full = st.checkbox("Full Context", value=True)
        
        # 3. cost tracker
        with st.expander("💰 Cost Estimator", expanded=True):
            st.metric("Total Cost", f"${st.session_state['total_cost']:.4f}")
            st.metric("Total Tokens", f"{st.session_state['total_tokens']:,}")
            if st.button("Reset Cost"):
                st.session_state['total_cost'] = 0.0
                st.session_state['total_tokens'] = 0
                st.rerun()
        
        if st.button("Clear Chat"): 
            st.session_state.messages = []
            st.rerun()

    with c_chat:
        # display chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): 
                st.markdown(m["content"])
                
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            if api_key:
                try:
                    # build context based on full or partial mode
                    if use_full:
                        ctx = df.groupby('dataset_name').apply(
                            lambda x: f"{x.name}: {','.join(x['column_name'])}"
                        ).str.cat(sep="\n")
                        ctx_head = "FULL DATABASE SCHEMA"
                    else:
                        ctx = df.head(50).to_csv()
                        ctx_head = "PARTIAL SAMPLE"

                    # provider logic for api endpoint
                    base_url = "https://api.x.ai/v1" if provider == "xAI" else None
                    client = openai.OpenAI(api_key=api_key, base_url=base_url)
                    
                    # call api
                    with st.spinner(f"Consulting {selected_model_id}..."):
                        resp = client.chat.completions.create(
                            model=selected_model_id,
                            messages=[
                                {"role": "system", "content": f"You are a Brightspace SQL Expert. Context ({ctx_head}):\n{ctx[:60000]}"},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        reply = resp.choices[0].message.content
                        
                        # calculate cost
                        if hasattr(resp, 'usage') and resp.usage:
                            in_tok = resp.usage.prompt_tokens
                            out_tok = resp.usage.completion_tokens
                            # (tokens * price) / 1,000,000
                            cost = (in_tok * model_info['in'] / 1_000_000) + (out_tok * model_info['out'] / 1_000_000)
                            
                            st.session_state['total_tokens'] += (in_tok + out_tok)
                            st.session_state['total_cost'] += cost
                    
                    # save response and refresh
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"AI Error: {str(e)}")
            else:
                st.error(f"Please provide an API Key for {provider}.")


# =============================================================================
# 10. main orchestrator
# =============================================================================

def main():
    """main entry point that orchestrates the application."""
    df = load_data()
    
    view = render_sidebar(df)
    
    if df.empty:
        st.title("Welcome")
        st.info("No data found. Please use the Sidebar to scrape the KB articles.")
        return

    # view router - render appropriate view based on sidebar selection
    if view == "Universe Map":
        render_map_view(df)
    elif view == "Schema Explorer":
        render_schema_view(df)
    elif view == "AI Architect":
        render_ai_view(df)


if __name__ == "__main__":
    # =========================================================
    # 🍞 UPGRADE TOAST (Placed BEFORE main so it always runs)
    # =========================================================
    try:
        import streamlit as st
        # Pop-up notification
        st.toast("📢 **New Version Available!** Check the Unified Explorer.", icon="🚀")
        
        # Sidebar button (sticky)
        st.sidebar.markdown("---")
        st.sidebar.link_button("✨ Go to Unified Explorer v2", "https://datasetsunifiedexplorer.streamlit.app/", type="primary")
    except Exception:
        pass

    # =========================================================
    # 🏃 RUN APP
    # =========================================================
    main()
