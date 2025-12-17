# streamlit run dataset_explorer_v6.py
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
st.set_page_config(page_title="Brightspace Datasets Explorer", layout="wide", page_icon="🌌")

logging.basicConfig(filename='scraper.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# SESSION STATE & AUTH
# ==========================================
if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0
if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state: st.session_state['auth_error'] = False

def perform_login():
    pwd_secret = st.secrets.get("app_password")
    if not pwd_secret:
        # If no secret set, allow dev access
        st.session_state['authenticated'] = True
        return
    if st.session_state.get("password_input") == pwd_secret:
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
# DATA LOGIC (SCRAPER & LOADER)
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
    return sorted(list(set([u for u in urls if u.startswith('http')])))

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
                all_data.extend(future.result())
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
    if os.path.exists('dataset_metadata.csv') and os.path.getsize('dataset_metadata.csv') > 10:
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()

@st.cache_data
def find_pk_fk_joins(df, selected_datasets=None):
    if df.empty: return pd.DataFrame()
    pks = df[df['is_primary_key'] == True]
    fks = df[df['is_foreign_key'] == True]
    if selected_datasets: fks = fks[fks['dataset_name'].isin(selected_datasets)]
    if pks.empty or fks.empty: return pd.DataFrame()
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    if joins.empty: return pd.DataFrame()
    result = joins[['dataset_name_fk', 'column_name', 'dataset_name_pk']]
    result.columns = ['Source', 'Key', 'Target']
    return result.drop_duplicates()

# ==========================================
# VISUALIZATION ENGINES
# ==========================================

@st.cache_data
def build_game_map(df, target_dataset=None):
    """
    The Clean, Game-Like HUD Map (V5 Style).
    Hidden lines by default. Locks on and shows fields when target selected.
    """
    categories = sorted(df['category'].unique())
    datasets = df[['dataset_name', 'category', 'description']].drop_duplicates('dataset_name')
    joins = find_pk_fk_joins(df) # Global joins

    # Layout Physics
    pos = {}
    center_x, center_y = 0, 0
    cat_radius = 20
    
    cat_step = 2 * math.pi / len(categories) if categories else 1
    for i, cat in enumerate(categories):
        angle = i * cat_step
        cx = center_x + cat_radius * math.cos(angle)
        cy = center_y + cat_radius * math.sin(angle)
        pos[cat] = (cx, cy)
        
        # Place datasets
        cat_ds = datasets[datasets['category'] == cat]
        ds_count = len(cat_ds)
        if ds_count > 0:
            ds_radius = 4
            ds_step = 2 * math.pi / ds_count
            for j, (_, row) in enumerate(cat_ds.iterrows()):
                ds_name = row['dataset_name']
                ds_angle = j * ds_step
                dx = cx + ds_radius * math.cos(ds_angle)
                dy = cy + ds_radius * math.sin(ds_angle)
                pos[ds_name] = (dx, dy)

    # Active Focus Logic
    active_nodes = set()
    active_edges = []
    
    if target_dataset and target_dataset != "None":
        active_nodes.add(target_dataset)
        # Find Neighbors
        out_ = joins[joins['Source'] == target_dataset]
        for _, r in out_.iterrows():
            active_nodes.add(r['Target'])
            active_edges.append((r['Source'], r['Target'], r['Key']))
        
        in_ = joins[joins['Target'] == target_dataset]
        for _, r in in_.iterrows():
            active_nodes.add(r['Source'])
            active_edges.append((r['Source'], r['Target'], r['Key']))

    # Build Traces
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    cat_x, cat_y, cat_text = [], [], []

    # Categories
    for cat in categories:
        x, y = pos[cat]
        is_dim = (target_dataset and target_dataset != "None")
        cat_x.append(x); cat_y.append(y + 2.5); cat_text.append(cat)
        node_x.append(x); node_y.append(y)
        node_text.append(f"Category: {cat}")
        node_color.append('rgba(255, 215, 0, 0.2)' if is_dim else 'rgba(255, 215, 0, 1)')
        node_size.append(30)

    # Datasets
    for _, row in datasets.iterrows():
        ds = row['dataset_name']
        if ds in pos:
            x, y = pos[ds]
            if target_dataset and target_dataset != "None":
                if ds == target_dataset:
                    color, size, opacity = '#00FF00', 20, 1
                elif ds in active_nodes:
                    color, size, opacity = '#00CCFF', 15, 1
                else:
                    color, size, opacity = '#333333', 8, 0.3
            else:
                color, size, opacity = '#00CCFF', 10, 0.8
            
            node_x.append(x); node_y.append(y)
            node_text.append(f"{ds}<br><i>{str(row['description'])[:80]}...</i>")
            node_color.append(color)
            node_size.append(size)

    # Edges & Labels
    edge_x, edge_y = [], []
    label_x, label_y, label_txt = [], [], []

    if active_edges:
        for s, t, k in active_edges:
            if s in pos and t in pos:
                x0, y0 = pos[s]
                x1, y1 = pos[t]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                label_x.append((x0 + x1) / 2)
                label_y.append((y0 + y1) / 2)
                label_txt.append(k)

    traces = []
    # Edges
    traces.append(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, color='#00FF00'), hoverinfo='none'))
    # Edge Labels
    if label_txt:
        traces.append(go.Scatter(
            x=label_x, y=label_y, mode='text', text=label_txt,
            textfont=dict(color='#00FF00', size=12, family="monospace", weight="bold"), hoverinfo='none'
        ))
    # Nodes
    traces.append(go.Scatter(
        x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text',
        marker=dict(color=node_color, size=node_size, line=dict(width=1, color='white'))
    ))
    # Cat Labels
    traces.append(go.Scatter(
        x=cat_x, y=cat_y, mode='text', text=cat_text,
        textfont=dict(color='gold', size=10), hoverinfo='none'
    ))

    layout = go.Layout(
        title=dict(text="Data Ecosystem Command", font=dict(color="white")),
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=40),
        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=750
    )
    return go.Figure(data=traces, layout=layout)

@st.cache_data
def build_focus_graph(selected_datasets, join_data):
    """
    The V4 Style Graph used for the 'Focus Explorer' tab.
    Calculates springs only for selected nodes.
    """
    G = nx.DiGraph()
    # Add nodes
    for ds in selected_datasets: G.add_node(ds)
    # Add edges
    if not join_data.empty:
        for _, row in join_data.iterrows():
            s, t = row['Source'], row['Target']
            if s in selected_datasets and t in selected_datasets:
                G.add_edge(s, t, label=row['Key'])

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
    
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="top center",
        marker=dict(color='#FF4B4B', size=25, line=dict(width=2, color='white'))
    )
    
    return go.Figure(data=[edge_trace, label_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    ))

# ==========================================
# MAIN UI
# ==========================================
df = load_data()

with st.sidebar:
    st.title("BDS Explorer")
    if df.empty:
        st.error("No data loaded.")
    else:
        st.success(f"{df['dataset_name'].nunique()} Datasets Online")
    
    with st.expander("🛠️ Scraper & Admin"):
        pasted_text = st.text_area("URLs", height=60, value=DEFAULT_URLS)
        if st.button("Update Data"):
            url_list = parse_urls_from_text_area(pasted_text)
            scrape_and_save_from_list(url_list)
            load_data.clear()
            st.rerun()

    if not st.session_state['authenticated']:
        st.divider()
        st.caption("Locked Features")
        st.text_input("Password", type="password", key="password_input", on_change=perform_login)
        if st.session_state['auth_error']: st.error("Access Denied")
    else:
        st.divider()
        st.success("Unlocked")
        if st.button("Logout"): logout(); st.rerun()

if df.empty:
    st.title("Brightspace Datasets Explorer")
    st.warning("Please use the Sidebar Scraper to load data.")
    st.stop()

# TABBED INTERFACE
tab_cmd, tab_sql, tab_ai = st.tabs(["🛰️ Command Center (Map)", "🔎 Focus & SQL", "🤖 Full AI Chat"])

# ----------------------------------------------------
# TAB 1: COMMAND CENTER (Game Map from V5)
# ----------------------------------------------------
with tab_cmd:
    col_map, col_info = st.columns([3, 1])
    
    with col_map:
        all_ds = ["None"] + sorted(df['dataset_name'].unique())
        target = st.selectbox("🎯 Select Target Dataset to Map Connections:", all_ds, index=0)
        fig_game = build_game_map(df, target_dataset=target if target != "None" else None)
        st.plotly_chart(fig_game, use_container_width=True)

    with col_info:
        st.markdown("#### 📋 Target Schema")
        if target and target != "None":
            st.info(f"Analyzing: **{target}**")
            subset = df[df['dataset_name'] == target]
            if not subset.empty:
                for _, row in subset.iterrows():
                    icon = "🔑" if row['is_primary_key'] else "🔗" if row['is_foreign_key'] else "📄"
                    st.markdown(f"**{icon} {row['column_name']}**")
                    st.caption(f"{row['data_type']}")
                    st.divider()
        else:
            st.caption("Select a dataset to view its immediate connections and schema.")

# ----------------------------------------------------
# TAB 2: FOCUS EXPLORER & SQL (From V4)
# ----------------------------------------------------
with tab_sql:
    st.subheader("Ad-Hoc SQL Builder")
    
    # Selectors
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        sel_ds = st.multiselect("Select Datasets to Join:", sorted(df['dataset_name'].unique()))
    with col_sel2:
        if st.button("Clear Selections"): st.rerun()
        
    if sel_ds:
        col_graph, col_code = st.columns([3, 2])
        
        # 1. Focus Graph
        with col_graph:
            joins_sub = find_pk_fk_joins(df, sel_ds)
            fig_foc = build_focus_graph(sel_ds, joins_sub)
            if fig_foc: st.plotly_chart(fig_foc, use_container_width=True)
            else: st.info("No direct relationships found among selections.")

        # 2. SQL Generation
        with col_code:
            if len(sel_ds) >= 2:
                base = sel_ds[0]
                aliases = {name: f"t{i+1}" for i, name in enumerate(sel_ds)}
                sql = [f"SELECT TOP 100", f"    {aliases[base]}.*"]
                sql.append(f"FROM {base} {aliases[base]}")
                
                # Logic to find path
                joined = {base}
                G_temp = nx.Graph()
                all_joins = find_pk_fk_joins(df)
                for _, r in all_joins.iterrows(): G_temp.add_edge(r['Source'], r['Target'], label=r['Key'])
                
                for i in range(1, len(sel_ds)):
                    curr = sel_ds[i]
                    connected = False
                    for existing in joined:
                        if G_temp.has_edge(curr, existing):
                            col = G_temp[curr][existing]['label']
                            sql.append(f"LEFT JOIN {curr} {aliases[curr]} ON {aliases[existing]}.{col} = {aliases[curr]}.{col}")
                            joined.add(curr)
                            connected = True
                            break
                    if not connected:
                        sql.append(f"CROSS JOIN {curr} {aliases[curr]} -- ⚠️ No Direct Link Found")
                        joined.add(curr)
                        
                st.code("\n".join(sql), language="sql")
            else:
                st.info("Select 2+ datasets to build SQL.")

# ----------------------------------------------------
# TAB 3: FULL AI CHAT (From V4)
# ----------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI Data Analyst")
    
    if not st.session_state['authenticated']:
        st.warning("Please log in via Sidebar.")
    else:
        # Chat Settings
        with st.expander("Settings", expanded=False):
            provider = st.radio("Model", ["OpenAI (GPT-4o)", "xAI (Grok)"], horizontal=True)
            
            # API Key Logic
            if "OpenAI" in provider:
                key_name = "openai_api_key"
                base_url = None; model = "gpt-4o"
            else:
                key_name = "xai_api_key"
                base_url = "https://api.x.ai/v1"; model = "grok-2-1212"
                
            api_key = st.secrets.get(key_name)
            if not api_key: api_key = st.text_input(f"Enter {key_name}", type="password")

        # Chat UI
        if "messages" not in st.session_state: st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
            
        if prompt := st.chat_input("Ask about the data..."):
            if not api_key: st.error("No API Key"); st.stop()
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Context: Use target if selected in Tab 1, else generic
                        target_ctx = df[df['dataset_name'] == target].to_csv() if 'target' in locals() and target != "None" else df.head(30).to_csv()
                        
                        client = openai.OpenAI(api_key=api_key, base_url=base_url)
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": f"Data Expert. Context:\n{target_ctx}"},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        reply = resp.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(str(e))
