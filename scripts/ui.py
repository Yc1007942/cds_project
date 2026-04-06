import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import warnings
import time
import traceback

import plotly.express as px
import plotly.graph_objects as go

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_READY = True
except ModuleNotFoundError as e:
    RandomForestClassifier = None
    KMeans = None
    PCA = None
    StandardScaler = None
    SKLEARN_READY = False
    SKLEARN_IMPORT_ERROR = str(e)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_READY = True
except ModuleNotFoundError as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_READY = False
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = str(e)

warnings.filterwarnings('ignore')

# --- CONFIG AND CACHING ---
st.set_page_config(page_title="MOLTNET // Neural Operations Deck", layout="wide", initial_sidebar_state="expanded")


def _ensure_debug_state():
    try:
        if "debug_events" not in st.session_state:
            st.session_state["debug_events"] = []
        return True
    except Exception:
        # Fails gracefully inside @st.cache_data / @st.cache_resource contexts
        return False

def log_checkpoint(stage, message, **details):
    can_log = _ensure_debug_state()
    ts = time.strftime("%H:%M:%S")
    detail_str = ""
    if details:
        detail_str = f" | {details}"
    event = f"{ts} | {stage} | {message}{detail_str}"
    
    if can_log:
        try:
            st.session_state["debug_events"].append(event)
            st.session_state["debug_events"] = st.session_state["debug_events"][-300:]
        except Exception:
            pass
            
    print(f"[MOLTNET-DEBUG] {event}")


def log_exception(stage, err):
    log_checkpoint(stage, "exception", error=str(err), err_type=type(err).__name__)
    print(traceback.format_exc())


def render_debug_panel():
    st.sidebar.markdown("---")
    show_debug = st.sidebar.toggle("DEBUG_CHECKPOINTS", value=True)
    if not show_debug:
        return
    _ensure_debug_state()
    if st.sidebar.button("CLEAR_DEBUG_LOGS", use_container_width=True):
        st.session_state["debug_events"] = []
    lines = "\n".join(st.session_state["debug_events"][-80:])
    st.sidebar.text_area("RUNTIME_CHECKPOINT_LOG", value=lines, height=240)
    st.sidebar.caption(f"entries: {len(st.session_state['debug_events'])}")


def dependency_guard():
    log_checkpoint("dependency_guard", "start")
    missing = []
    if not SKLEARN_READY:
        missing.append(f"scikit-learn import failed: {SKLEARN_IMPORT_ERROR}")
    if not SENTENCE_TRANSFORMERS_READY:
        missing.append(f"sentence-transformers import failed: {SENTENCE_TRANSFORMERS_IMPORT_ERROR}")

    if missing:
        log_checkpoint("dependency_guard", "missing_modules", missing_count=len(missing))
        st.title(">> DEPENDENCY_BOOT_FAILURE")
        st.error("Missing Python dependencies prevent app startup.")
        for item in missing:
            st.write(f"- {item}")
        st.info("Activate your project environment, then install dependencies.")
        st.code("python3 -m pip install -r requirements.txt", language="bash")
        st.stop()
    log_checkpoint("dependency_guard", "ok")


dependency_guard()

# Sci-fi AI cockpit styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

    :root {
        --bg-deep: #04070d;
        --bg-panel: rgba(8, 14, 28, 0.74);
        --cyan: #3be3ff;
        --neon: #77f7ff;
        --violet: #67ff9f;
        --warning: #ff6fa8;
        --text-main: #d9f7ff;
        --line: rgba(59, 227, 255, 0.28);
    }
    
    html, body, .stApp, section, div {
        font-family: 'Share Tech Mono', monospace;
        color: var(--text-main);
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 8%, rgba(59, 227, 255, 0.15), transparent 35%),
            radial-gradient(circle at 82% 12%, rgba(103, 255, 159, 0.12), transparent 34%),
            linear-gradient(125deg, #02040a 0%, #050d1d 52%, #02040a 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 38px 38px;
        pointer-events: none;
        animation: drift 24s linear infinite;
        z-index: -1;
    }

    @keyframes drift {
        from { transform: translateY(0px); }
        to { transform: translateY(38px); }
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--neon) !important;
        text-shadow: 0 0 16px rgba(59, 227, 255, 0.35);
        border-bottom: 1px solid var(--line);
        padding-bottom: 10px;
    }
    
    .hud-card {
        border: 1px solid var(--line);
        background: linear-gradient(150deg, rgba(10, 23, 42, 0.82), rgba(6, 12, 24, 0.78));
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 0 0 1px rgba(59,227,255,0.1) inset, 0 0 22px rgba(59, 227, 255, 0.14);
        margin-bottom: 12px;
    }

    .hud-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 4px 0 12px 0;
    }

    .hud-pill {
        border: 1px solid var(--line);
        background: rgba(4, 10, 20, 0.8);
        border-radius: 999px;
        padding: 6px 12px;
        color: var(--text-main);
        font-size: 0.76rem;
        letter-spacing: 1px;
    }

    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin-right: 8px;
        border-radius: 50%;
        background: var(--violet);
        box-shadow: 0 0 10px var(--violet);
        animation: pulse 1.5s infinite ease-in-out;
    }

    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.5; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.5; }
    }

    [data-testid="stMetricValue"] {
        color: var(--cyan) !important;
        text-shadow: 0 0 18px rgba(59, 227, 255, 0.65);
        font-size: 2.2rem !important;
    }

    hr {
        border-color: var(--line) !important;
        box-shadow: 0 0 10px rgba(59, 227, 255, 0.2);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050b18 0%, #04070f 100%);
        border-right: 1px solid var(--line);
        box-shadow: 6px 0 24px rgba(59, 227, 255, 0.12);
    }

    section[data-testid="stSidebar"] * {
        color: #bceeff !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        box-shadow: 0 0 14px rgba(59,227,255,0.18);
        border-radius: 8px;
    }

    .stButton>button {
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        border: 1px solid var(--line);
        color: var(--text-main);
        background: linear-gradient(90deg, rgba(21, 56, 86, 0.65), rgba(11, 28, 48, 0.76));
        box-shadow: 0 0 8px rgba(59,227,255,0.18);
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(59,227,255,0.7);
        border-color: var(--cyan);
        transform: translateY(-1px);
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label, .stTextInput label, .stTextArea label {
        color: #a8ebff !important;
        letter-spacing: 0.5px;
    }

    .stAlert {
        border: 1px solid var(--line);
        background: rgba(8, 20, 34, 0.85);
    }

    .signal-wrap {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(7, 16, 30, 0.78);
        padding: 10px 12px;
        margin-bottom: 12px;
    }

    .signal-grid {
        display: flex;
        align-items: flex-end;
        gap: 6px;
        height: 34px;
    }

    .signal-bar {
        width: 8px;
        border-radius: 4px;
        background: linear-gradient(180deg, rgba(59,227,255,0.95), rgba(103,255,159,0.8));
        animation: uplink 1.2s ease-in-out infinite;
        box-shadow: 0 0 12px rgba(59, 227, 255, 0.45);
    }

    @keyframes uplink {
        0%, 100% { height: 24%; opacity: 0.55; }
        50% { height: 100%; opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    log_checkpoint("load_data", "start", path="data/features_subset_train.parquet")
    try:
        df = pd.read_parquet("data/features_subset_train.parquet")
        log_checkpoint("load_data", "success", rows=len(df), cols=len(df.columns))
        return df
    except Exception as e:
        log_exception("load_data", e)
        st.error(f"> ERR_LOAD_DATA: {e}")
        return pd.DataFrame()

@st.cache_data
def load_feature_matrix():
    log_checkpoint("load_feature_matrix", "start", path="data/feature_matrix_full_subset_train.parquet")
    try:
        df = pd.read_parquet("data/feature_matrix_full_subset_train.parquet")
        log_checkpoint("load_feature_matrix", "success", rows=len(df), cols=len(df.columns))
        return df
    except Exception as e:
        log_exception("load_feature_matrix", e)
        st.error(f"> ERR_LOAD_MATRIX: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_model(df_matrix):
    log_checkpoint("get_model", "start", empty=df_matrix.empty)
    if df_matrix.empty:
        log_checkpoint("get_model", "skipped_empty_matrix")
        return None, []
    
    features = [c for c in df_matrix.columns if c not in [
        'label', 'id', 'split', 'author', 'subreddit', 'text', 
        'created_utc', 'interaction_type', 'source', 'is_comment', 
        'text_clean', 'timestamp', 'hour_of_day', 'day_of_week'
    ]]
    
    X = df_matrix[features].select_dtypes(include=[np.number]).fillna(0)
    y = df_matrix['label']
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    rf.fit(X, y)
    log_checkpoint("get_model", "trained", feature_count=len(X.columns), samples=len(X))
    return rf, X.columns.tolist()

@st.cache_resource
def get_sentence_transformer():
    log_checkpoint("get_sentence_transformer", "start", model_name="all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    log_checkpoint("get_sentence_transformer", "loaded")
    return model

def extract_features(text, st_model):
    log_checkpoint("extract_features", "start", char_len=len(text), token_est=len(text.split()))
    text_clean = re.sub(r'http\S+|www\.\S+', '', text)
    text_clean = re.sub(r'[ \t]+', ' ', text_clean).strip()
    
    char_count = len(text_clean)
    words = text_clean.split()
    word_count = len(words)
    
    embeddings = st_model.encode([text_clean])[0]
    
    feature_dict = {
        'char_count': char_count,
        'word_count': word_count,
        'flesch_kincaid': 8.0,
        'gunning_fog': 8.0,
        'coleman_liau': 8.0,
        'automated_readability': 8.0,
        'ttr': len(set(words))/max(word_count, 1),
        'sentiment_compound': 0.0,
        'perplexity': 50.0,
        'burstiness': 0.5,
    }
    for i, emb in enumerate(embeddings):
        feature_dict[f'emb_{i}'] = emb
    log_checkpoint("extract_features", "success", embedding_dim=len(embeddings), words=word_count)
    return pd.DataFrame([feature_dict])


def render_hud(title, subtitle, mode_label):
    st.markdown(
        f"""
        <div class='hud-card'>
            <div style='display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;'>
                <div>
                    <div style='font-family:Orbitron,sans-serif; color:#77f7ff; font-size:1rem; letter-spacing:1.2px;'>{title}</div>
                    <div style='opacity:0.85; font-size:0.85rem;'>{subtitle}</div>
                </div>
                <div class='hud-pill'><span class='pulse-dot'></span>{mode_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_bars(label):
    bars = "".join(
        [f"<div class='signal-bar' style='animation-delay:{i * 0.08}s'></div>" for i in range(14)]
    )
    st.markdown(
        f"""
        <div class='signal-wrap'>
            <div style='font-size:0.8rem; letter-spacing:1px; opacity:0.9; margin-bottom:6px;'>{label}</div>
            <div class='signal-grid'>{bars}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def compute_ai_clusters(df, numeric_cols, sample_size, n_clusters):
    log_checkpoint("compute_ai_clusters", "start", sample_size=sample_size, n_clusters=n_clusters, numeric_cols=len(numeric_cols))
    if 'label' not in df.columns:
        log_checkpoint("compute_ai_clusters", "no_label_column")
        return pd.DataFrame(), "NO_LABEL_COLUMN"

    ai_df = df[df['label'] == 1].copy()
    if ai_df.empty:
        log_checkpoint("compute_ai_clusters", "no_ai_rows")
        return pd.DataFrame(), "NO_AI_ROWS"

    use_cols = [c for c in numeric_cols if c in ai_df.columns]
    if len(use_cols) < 2:
        log_checkpoint("compute_ai_clusters", "insufficient_features", usable=len(use_cols))
        return pd.DataFrame(), "INSUFFICIENT_NUMERIC_FEATURES"

    ai_df = ai_df.sample(min(sample_size, len(ai_df)), random_state=42)
    x = ai_df[use_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if x.shape[0] < 3:
        log_checkpoint("compute_ai_clusters", "insufficient_samples", samples=int(x.shape[0]))
        return pd.DataFrame(), "NOT_ENOUGH_AI_SAMPLES"

    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(x_scaled)

    k = max(2, min(n_clusters, len(ai_df) - 1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(points)

    plot_df = ai_df.copy()
    plot_df['pc1'] = points[:, 0]
    plot_df['pc2'] = points[:, 1]
    plot_df['cluster_id'] = cluster_ids.astype(str)
    plot_df['cluster_size'] = plot_df.groupby('cluster_id')['cluster_id'].transform('count')

    explained_var = float(np.sum(pca.explained_variance_ratio_))
    log_checkpoint("compute_ai_clusters", "success", returned_rows=len(plot_df), explained_var=round(explained_var, 4))
    return plot_df, explained_var


@st.cache_data(show_spinner=False)
def compute_ai_clusters_3d(df, numeric_cols, sample_size, n_clusters):
    log_checkpoint("compute_ai_clusters_3d", "start", sample_size=sample_size, n_clusters=n_clusters, numeric_cols=len(numeric_cols))
    if 'label' not in df.columns:
        log_checkpoint("compute_ai_clusters_3d", "no_label_column")
        return pd.DataFrame(), "NO_LABEL_COLUMN"

    ai_df = df[df['label'] == 1].copy()
    if ai_df.empty:
        log_checkpoint("compute_ai_clusters_3d", "no_ai_rows")
        return pd.DataFrame(), "NO_AI_ROWS"

    use_cols = [c for c in numeric_cols if c in ai_df.columns]
    if len(use_cols) < 3:
        log_checkpoint("compute_ai_clusters_3d", "insufficient_features", usable=len(use_cols))
        return pd.DataFrame(), "INSUFFICIENT_NUMERIC_FEATURES"

    ai_df = ai_df.sample(min(sample_size, len(ai_df)), random_state=7)
    x = ai_df[use_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if x.shape[0] < 4:
        log_checkpoint("compute_ai_clusters_3d", "insufficient_samples", samples=int(x.shape[0]))
        return pd.DataFrame(), "NOT_ENOUGH_AI_SAMPLES"

    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3, random_state=42)
    points = pca.fit_transform(x_scaled)

    k = max(2, min(n_clusters, len(ai_df) - 1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(points)

    plot_df = ai_df.copy()
    plot_df['pc1'] = points[:, 0]
    plot_df['pc2'] = points[:, 1]
    plot_df['pc3'] = points[:, 2]
    plot_df['cluster_id'] = cluster_ids.astype(str)
    plot_df['cluster_size'] = plot_df.groupby('cluster_id')['cluster_id'].transform('count')
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    log_checkpoint("compute_ai_clusters_3d", "success", returned_rows=len(plot_df), explained_var=round(explained_var, 4))
    return plot_df, explained_var

# --- TERMINAL UI LOGIC ---
st.sidebar.markdown("<h2 style='color:#77f7ff;'>[NEURAL.NAV]</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Modern Taskbar styling
page = st.sidebar.radio(
    "NAV_MODE",
    ["[01] DATA_EXPLORER", "[02] FEATURE_MATRIX", "[03] INFERENCE_CORE"],
    label_visibility="collapsed"
)
log_checkpoint("navigation", "page_selected", page=page)

st.sidebar.markdown("---")
st.sidebar.markdown("<small style='color:#71d9ff;'>V.2.0.0.CYBERDECK // LINK STABLE</small>", unsafe_allow_html=True)
render_debug_panel()

df_features = load_data()
log_checkpoint("dataframe_state", "df_features_loaded", empty=df_features.empty)

# Global plotly layout config
plotly_layout = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Share Tech Mono', color='#c8f5ff'),
    legend=dict(bgcolor='rgba(6,14,28,0.55)', bordercolor='rgba(59,227,255,0.35)', borderwidth=1),
    margin=dict(l=20, r=20, t=40, b=20)
)

if page == "[01] DATA_EXPLORER":
    st.title(">> DATA_EXPLORER.EXE")
    render_hud("OBSERVATION DECK", "SYSTEM STATUS: ACCESSING RAW DATA_STREAMS", "SCAN MODE: EXPLORATION")
    render_signal_bars("NEURAL TRAFFIC / LIVE")
    
    if not df_features.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("TOTAL_DOCS", f"{len(df_features):,}")
        col2.metric("AI_AGENTS", f"{len(df_features[df_features['label'] == 1]):,}")
        col3.metric("HUMAN_USERS", f"{len(df_features[df_features['label'] == 0]):,}")
        
        st.markdown("---")
        st.subheader(">> QUERY_FILTERS")
        
        # Interactive Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            min_words = int(df_features['word_count'].min()) if 'word_count' in df_features.columns else 0
            max_words = int(df_features['word_count'].max()) if 'word_count' in df_features.columns else 1000
            word_range = st.slider("SET_WORD_COUNT_FILTER", min_value=min_words, max_value=max_words, value=(min_words, max_words))
            
        with f_col2:
            selected_subs = []
            if 'subreddit' in df_features.columns:
                subs = df_features['subreddit'].dropna().unique().tolist()
                selected_subs = st.multiselect("SET_SECTOR_FILTER [SUBREDDIT]", options=subs, default=subs[:3] if len(subs)>3 else subs)

        with f_col3:
            keyword = st.text_input("SEARCH_PAYLOAD", placeholder="author/text contains...")
            max_rows = st.slider("TABLE_ROW_CAP", min_value=50, max_value=1000, value=300, step=50)
        
        # Apply filters
        filtered_df = df_features[
            (df_features['word_count'] >= word_range[0]) & 
            (df_features['word_count'] <= word_range[1])
        ]
        if 'subreddit' in df_features.columns and selected_subs:
            filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subs)]
        if keyword:
            pattern = re.escape(keyword.lower())
            if 'text' in filtered_df.columns:
                text_hits = filtered_df['text'].fillna('').str.lower().str.contains(pattern, regex=True)
            else:
                text_hits = pd.Series(False, index=filtered_df.index)
            if 'author' in filtered_df.columns:
                author_hits = filtered_df['author'].fillna('').str.lower().str.contains(pattern, regex=True)
            else:
                author_hits = pd.Series(False, index=filtered_df.index)
            filtered_df = filtered_df[text_hits | author_hits]
            
        st.markdown(f"`FETCHED {len(filtered_df):,} RECORDS MATCHING PARAMETERS...`")

        if {'word_count', 'char_count', 'label'}.issubset(filtered_df.columns):
            sample_n = min(1200, len(filtered_df))
            viz_df = filtered_df.sample(sample_n, random_state=42) if len(filtered_df) > sample_n else filtered_df
            viz_df = viz_df.assign(class_name=viz_df['label'].replace({0: 'HUMAN', 1: 'AI'}))
            fig_scan = px.scatter(
                viz_df,
                x='word_count',
                y='char_count',
                color='class_name',
                size='word_count',
                hover_data=['author'] if 'author' in viz_df.columns else None,
                color_discrete_map={'HUMAN': '#67ff9f', 'AI': '#3be3ff'},
                opacity=0.72,
                title="TRAFFIC SCAN // WORD_COUNT x CHAR_COUNT",
            )
            fig_scan.update_layout(**plotly_layout)
            st.plotly_chart(fig_scan, use_container_width=True)
        
        # DataFrame rendered
        display_cols = [c for c in ['author', 'text', 'subreddit', 'label', 'word_count', 'sentiment_compound'] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head(max_rows), use_container_width=True)

        st.download_button(
            label="EXPORT_FILTERED_VIEW_CSV",
            data=filtered_df[display_cols].to_csv(index=False).encode('utf-8'),
            file_name="filtered_datastream.csv",
            mime="text/csv",
            use_container_width=True,
        )
            
    else:
        st.warning("> ERR: DATA NOT FOUND.")

elif page == "[02] FEATURE_MATRIX":
    st.title(">> FEATURE_MATRIX.EXE")
    render_hud("VECTOR INTELLIGENCE", "SYSTEM STATUS: VISUALIZING LINGUISTIC VECTORS", "SCAN MODE: FEATURE SYNTH")
    render_signal_bars("EMBEDDING BUS / HIGH THROUGHPUT")
    
    if not df_features.empty:
        features_to_plot = ['word_count', 'char_count', 'sentiment_compound', 'perplexity', 'burstiness', 'ttr']
        existing_features = [f for f in features_to_plot if f in df_features.columns]
        
        if existing_features:
            col_sel, col_empty = st.columns([1, 2])
            with col_sel:
                selected_feature = st.selectbox("> TARGET_VARIABLE", existing_features)
                chart_type = st.radio("> RENDER_MODE", ["DISTRIBUTION", "BOX_PLOT"])
                show_density = st.toggle("ENABLE_DENSITY_OVERLAY", value=True)
                
            st.markdown("---")
            
            labels = df_features['label'].replace({0: 'HUMAN', 1: 'AI'})
            
            # Matrix Colors: AI=Cyan, Human=Green
            color_map = {'HUMAN': '#67ff9f', 'AI': '#3be3ff'}
            
            if chart_type == "DISTRIBUTION":
                fig = px.histogram(
                    df_features, x=selected_feature, color=labels,
                    barmode="overlay", nbins=50,
                    color_discrete_map=color_map,
                    title=f"VECTOR_DIST // {selected_feature.upper()}"
                )
                fig.update_layout(**plotly_layout)
                st.plotly_chart(fig, use_container_width=True)

                if show_density:
                    fig_violin = px.violin(
                        df_features.assign(class_name=labels),
                        x='class_name',
                        y=selected_feature,
                        color='class_name',
                        box=True,
                        points='outliers',
                        color_discrete_map=color_map,
                        title=f"DENSITY_OVERLAY // {selected_feature.upper()}"
                    )
                    fig_violin.update_layout(**plotly_layout)
                    st.plotly_chart(fig_violin, use_container_width=True)
            else:
                fig = px.box(
                    df_features, x=labels, y=selected_feature, color=labels,
                    color_discrete_map=color_map,
                    title=f"VARIANCE_SWEEP // {selected_feature.upper()}"
                )
                fig.update_layout(**plotly_layout)
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown("---")
            st.subheader(">> 2D_CORRELATION_MAP")
            scat_col1, scat_col2 = st.columns(2)
            with scat_col1:
                x_axis = st.selectbox("> X_AXIS", existing_features, index=0)
            with scat_col2:
                y_axis = st.selectbox("> Y_AXIS", existing_features, index=max(1, len(existing_features)-1))
                
            plot_df = df_features.sample(min(2000, len(df_features)))
            plot_labels = plot_df['label'].replace({0: 'HUMAN', 1: 'AI'})
            fig_scatter = px.scatter(
                plot_df, x=x_axis, y=y_axis, color=plot_labels,
                color_discrete_map=color_map,
                opacity=0.7, title=f"CROSS_CORRELATION // {x_axis.upper()} VS {y_axis.upper()}"
            )
            fig_scatter.update_layout(**plotly_layout)
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")
            st.subheader(">> MULTI_DIMENSIONAL_SCAN")
            mode_3d = st.toggle("ENABLE_3D_POINT_CLOUD", value=True)
            if mode_3d and len(existing_features) >= 3:
                c1, c2, c3 = st.columns(3)
                with c1:
                    x3 = st.selectbox("> X3_AXIS", existing_features, index=0)
                with c2:
                    y3 = st.selectbox("> Y3_AXIS", existing_features, index=1)
                with c3:
                    z3 = st.selectbox("> Z3_AXIS", existing_features, index=2)

                cloud_df = df_features.sample(min(1800, len(df_features)), random_state=7)
                cloud_labels = cloud_df['label'].replace({0: 'HUMAN', 1: 'AI'})
                fig_3d = px.scatter_3d(
                    cloud_df,
                    x=x3,
                    y=y3,
                    z=z3,
                    color=cloud_labels,
                    color_discrete_map=color_map,
                    opacity=0.55,
                    title=f"POINT_CLOUD // {x3.upper()} x {y3.upper()} x {z3.upper()}"
                )
                fig_3d.update_layout(**plotly_layout)
                st.plotly_chart(fig_3d, use_container_width=True)

            heat_features = st.multiselect(
                "CORRELATION_FEATURE_SET",
                options=existing_features,
                default=existing_features[: min(5, len(existing_features))],
            )
            if len(heat_features) >= 2:
                corr = df_features[heat_features].corr(numeric_only=True)
                fig_heat = px.imshow(
                    corr,
                    text_auto='.2f',
                    color_continuous_scale='Tealgrn',
                    title="CORRELATION_HEATMAP // SELECTED FEATURES",
                )
                fig_heat.update_layout(**plotly_layout)
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("---")
            st.subheader(">> AI_AGENT_CLUSTER_RENDER")
            ai_count = int((df_features['label'] == 1).sum()) if 'label' in df_features.columns else 0
            emb_cols = [c for c in df_features.columns if c.startswith('emb_')]
            numeric_fallback = [
                c for c in df_features.select_dtypes(include=[np.number]).columns
                if c not in ['label']
            ]
            cluster_features = emb_cols if len(emb_cols) >= 2 else numeric_fallback

            c_cfg1, c_cfg2 = st.columns(2)
            with c_cfg1:
                sample_max = max(200, min(ai_count if ai_count > 0 else 1000, 5000))
                cluster_sample = st.slider(
                    "CLUSTER_SAMPLE_SIZE",
                    min_value=200,
                    max_value=sample_max,
                    value=min(1200, sample_max),
                    step=100,
                    disabled=ai_count < 200,
                )
            with c_cfg2:
                cluster_k = st.slider(
                    "CLUSTER_COUNT_K",
                    min_value=2,
                    max_value=12,
                    value=5,
                    step=1,
                    disabled=ai_count < 3,
                )

            if ai_count < 3:
                st.info("Need at least 3 AI rows to render cluster map.")
            elif len(cluster_features) < 2:
                st.info("Need at least 2 numeric/embedding features for cluster rendering.")
            else:
                cluster_df, explained = compute_ai_clusters(
                    df_features,
                    cluster_features,
                    sample_size=cluster_sample,
                    n_clusters=cluster_k,
                )
                if cluster_df.empty:
                    st.info("Cluster rendering unavailable for current data shape.")
                else:
                    hover_items = [c for c in ['author', 'subreddit', 'word_count', 'char_count', 'cluster_size'] if c in cluster_df.columns]
                    fig_cluster = px.scatter(
                        cluster_df,
                        x='pc1',
                        y='pc2',
                        color='cluster_id',
                        size='cluster_size',
                        hover_data=hover_items,
                        opacity=0.8,
                        title="AI_AGENT_CLUSTER_MAP // PCA(2D) + KMEANS",
                    )
                    fig_cluster.update_traces(marker=dict(line=dict(width=0.6, color='rgba(220,255,255,0.35)')))
                    fig_cluster.update_layout(**plotly_layout)
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    st.caption(f"Projected variance preserved: {explained:.1%} | Source features: {len(cluster_features)}")

                    st.markdown("---")
                    st.subheader(">> TEMPORAL_CLUSTER_SWEEP")
                    sweep_bins = st.slider("SWEEP_PHASE_BINS", min_value=6, max_value=20, value=12, step=1)
                    sweep_df = cluster_df.copy()
                    theta = np.arctan2(sweep_df['pc2'], sweep_df['pc1'])
                    norm_theta = (theta + np.pi) / (2 * np.pi)
                    sweep_df['phase_bin'] = np.floor(norm_theta * sweep_bins).clip(0, sweep_bins - 1).astype(int)
                    sweep_df['phase'] = sweep_df['phase_bin'].apply(lambda p: f"PHASE_{p:02d}")

                    fig_sweep = px.scatter(
                        sweep_df,
                        x='pc1',
                        y='pc2',
                        color='cluster_id',
                        size='cluster_size',
                        animation_frame='phase',
                        hover_data=hover_items,
                        opacity=0.82,
                        title="CLUSTER_SWEEP_ANIMATION // PCA(2D) PHASE SCAN",
                    )
                    fig_sweep.update_layout(**plotly_layout)
                    fig_sweep.update_xaxes(range=[sweep_df['pc1'].min() * 1.1, sweep_df['pc1'].max() * 1.1])
                    fig_sweep.update_yaxes(range=[sweep_df['pc2'].min() * 1.1, sweep_df['pc2'].max() * 1.1])
                    st.plotly_chart(fig_sweep, use_container_width=True)

                    st.markdown("---")
                    st.subheader(">> ROTATING_3D_CLUSTER_VIEW")
                    cluster_df_3d, explained_3d = compute_ai_clusters_3d(
                        df_features,
                        cluster_features,
                        sample_size=cluster_sample,
                        n_clusters=cluster_k,
                    )

                    if cluster_df_3d.empty:
                        st.info("3D cluster view needs at least 3 numeric/embedding features and enough AI rows.")
                    else:
                        if 'cluster_angle' not in st.session_state:
                            st.session_state['cluster_angle'] = 35

                        c3d_a, c3d_b, c3d_c = st.columns(3)
                        with c3d_a:
                            cam_radius = st.slider("CAMERA_RADIUS", min_value=1.0, max_value=3.0, value=1.8, step=0.1)
                        with c3d_b:
                            angle_step = st.slider("ROTATION_STEP", min_value=5, max_value=45, value=15, step=5)
                        with c3d_c:
                            if st.button("ROTATE_VIEW", use_container_width=True):
                                st.session_state['cluster_angle'] = (st.session_state['cluster_angle'] + angle_step) % 360

                        manual_angle = st.slider(
                            "MANUAL_CAMERA_ANGLE",
                            min_value=0,
                            max_value=359,
                            value=int(st.session_state['cluster_angle']),
                            step=1,
                        )
                        st.session_state['cluster_angle'] = manual_angle

                        angle_rad = np.deg2rad(manual_angle)
                        cam_eye = dict(
                            x=cam_radius * np.cos(angle_rad),
                            y=cam_radius * np.sin(angle_rad),
                            z=0.85,
                        )

                        hover_3d = [c for c in ['author', 'subreddit', 'cluster_size'] if c in cluster_df_3d.columns]
                        fig_3cluster = px.scatter_3d(
                            cluster_df_3d,
                            x='pc1',
                            y='pc2',
                            z='pc3',
                            color='cluster_id',
                            size='cluster_size',
                            hover_data=hover_3d,
                            opacity=0.72,
                            title="AI_SWARM_3D // INTERACTIVE CAMERA",
                        )
                        fig_3cluster.update_layout(
                            **plotly_layout,
                            scene_camera=dict(eye=cam_eye),
                            scene=dict(
                                bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(backgroundcolor='rgba(6,12,24,0.4)', gridcolor='rgba(133,216,255,0.15)'),
                                yaxis=dict(backgroundcolor='rgba(6,12,24,0.4)', gridcolor='rgba(133,216,255,0.15)'),
                                zaxis=dict(backgroundcolor='rgba(6,12,24,0.4)', gridcolor='rgba(133,216,255,0.15)')
                            ),
                        )
                        st.plotly_chart(fig_3cluster, use_container_width=True)
                        st.caption(f"3D projected variance preserved: {explained_3d:.1%} | Camera angle: {manual_angle}°")
        else:
            st.warning("> ERR: MATRIX COLUMNS NOT DETECTED.")

elif page == "[03] INFERENCE_CORE":
    st.title(">> INFERENCE_CORE.EXE")
    render_hud("PREDICTION CHAMBER", "INITIALIZING NEURAL CLASSIFIER BACKEND", "SCAN MODE: LIVE INFERENCE")
    render_signal_bars("CLASSIFIER CORE / SYNCHRONIZED")
    
    model = None
    model_features = []
    st_model = None
    with st.status("> BOOT_SEQUENCE / ENGAGED", expanded=True) as status:
        try:
            log_checkpoint("boot_sequence", "start")
            st.write("> MOUNTING FEATURE_MATRIX...")
            df_matrix = load_feature_matrix()
            st.write("> TRAINING FOREST_CLASSIFIER...")
            model, model_features = get_model(df_matrix)
            st.write("> LOADING TRANSFORMER_WEIGHTS...")
            st_model = get_sentence_transformer()
            log_checkpoint("boot_sequence", "complete", model_ready=model is not None, feature_count=len(model_features))
            status.update(label="> BOOT_SEQUENCE / COMPLETE", state="complete", expanded=False)
        except Exception as e:
            log_exception("boot_sequence", e)
            status.update(label="> BOOT_SEQUENCE / FAILED", state="error", expanded=True)
            st.exception(e)
        
    if model and st_model is not None:
        st.markdown("---")
        st.subheader(">> INPUT_TERMINAL")
        user_input = st.text_area("> AWAITING_TEXT_PAYLOAD:", height=150, placeholder="// ENTER SUSPECT STRING DATA HERE...")
        
        if st.button("> EXECUTE_ANALYSIS", use_container_width=True):
            log_checkpoint("inference", "execute_clicked", input_chars=len(user_input))
            if not user_input.strip():
                log_checkpoint("inference", "empty_payload")
                st.warning("> ERR: PAYLOAD_EMPTY")
            else:
                try:
                    with st.spinner("> EXTRACTING_VECTORS..."):
                        input_df = extract_features(user_input, st_model)
                        
                        X_infer = pd.DataFrame(columns=model_features)
                        for col in model_features:
                            if col in input_df.columns:
                                X_infer.loc[0, col] = input_df[col].iloc[0]
                            else:
                                X_infer.loc[0, col] = 0.0
                                
                        X_infer = X_infer.fillna(0)
                        log_checkpoint("inference", "inference_vector_ready", feature_count=len(model_features))
                        
                        prediction = model.predict(X_infer)[0]
                        proba = model.predict_proba(X_infer)[0]
                        log_checkpoint("inference", "prediction_complete", prediction=int(prediction), ai_prob=float(proba[1]))
                    
                    ai_prob = float(proba[1])
                    human_prob = float(proba[0])
                    
                    result_string = "++ ARTIFICIAL INTELLIGENCE DETECTED ++" if prediction == 1 else "++ HUMAN ORIGIN VERIFIED ++"
                    result_color = "#00ffff" if prediction == 1 else "#00ff41" # Cyan for AI, Green for Human
                    
                    st.markdown("---")
                    st.subheader(">> OUTPUT_STREAM")
                    
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.markdown(f"""
                        <div style='background-color:rgba(0,0,0,0.5); border:1px solid {result_color}; padding: 20px; text-align:center;'>
                            <h2 style='color: {result_color} !important; border:none; text-shadow: 0 0 15px {result_color};'>{result_string}</h2>
                            <p>CONFIDENCE_AI: {ai_prob:.2%}</p>
                            <p>CONFIDENCE_HUMAN: {human_prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>> CONFIDENCE_INTERVAL_BAR:", unsafe_allow_html=True)
                        st.progress(ai_prob if prediction == 1 else human_prob)
                    
                    with res_col2:
                        gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=ai_prob * 100,
                            title={'text': "AI PROBABILITY"},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': '#c8f5ff'},
                                'bar': {'color': '#3be3ff'},
                                'steps': [
                                    {'range': [0, 35], 'color': 'rgba(103,255,159,0.22)'},
                                    {'range': [35, 65], 'color': 'rgba(255,196,87,0.2)'},
                                    {'range': [65, 100], 'color': 'rgba(59,227,255,0.34)'}
                                ],
                            }
                        ))
                        gauge.update_layout(**plotly_layout, title="PROBABILITY_GAUGE")
                        st.plotly_chart(gauge, use_container_width=True)

                        radar_features = ['word_count', 'char_count', 'perplexity', 'burstiness', 'sentiment_compound']
                        valid_radar = [f for f in radar_features if f in df_features.columns]
                        
                        if len(valid_radar) >= 3:
                            def normalize(val, col):
                                v_min = df_features[col].min()
                                v_max = df_features[col].max()
                                if v_max == v_min: return 0.5
                                return (val - v_min) / (v_max - v_min)
                            
                            means_df = df_features.groupby('label')[valid_radar].mean()
                            human_means = [normalize(means_df.loc[0, f], f) if 0 in means_df.index else 0 for f in valid_radar]
                            ai_means = [normalize(means_df.loc[1, f], f) if 1 in means_df.index else 0 for f in valid_radar]
                            input_vals = [normalize(input_df.loc[0, f], f) if f in input_df.columns else 0 for f in valid_radar]
                            
                            fig_radar = go.Figure()
                            # Matrix radar styling
                            fig_radar.add_trace(go.Scatterpolar(r=human_means, theta=valid_radar, fill='toself', name='TARGET_HOMO_SAPIEN', line_color='#00ff41', fillcolor='rgba(0,255,65,0.1)'))
                            fig_radar.add_trace(go.Scatterpolar(r=ai_means, theta=valid_radar, fill='toself', name='TARGET_SYNTHETIC', line_color='#00ffff', fillcolor='rgba(0,255,255,0.1)'))
                            fig_radar.add_trace(go.Scatterpolar(r=input_vals, theta=valid_radar, fill='toself', name='INPUT_DATA', line_color='#ffffff', line_width=3))
                            
                            fig_radar.update_layout(
                                **plotly_layout,
                                polar=dict(
                                    radialaxis=dict(visible=False, range=[0, 1]),
                                    bgcolor='rgba(0,0,0,0)'
                                ),
                                showlegend=True,
                                title=">> SPATIAL_DIVERGENCE_MAP",
                                margin=dict(l=20, r=20, t=30, b=20)
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)

                    with st.expander("> DIAGNOSTIC_TRACE", expanded=False):
                        st.write(f"TOKENS_EST: {len(user_input.split())}")
                        st.write(f"CHAR_COUNT: {len(user_input)}")
                        st.write(f"MODEL_FEATURES_USED: {len(model_features)}")
                        st.write(f"PRED_LABEL: {'AI' if prediction == 1 else 'HUMAN'}")
                except Exception as e:
                    log_exception("inference", e)
                    st.error("> INFERENCE_FAILURE")
                    st.exception(e)
    else:
        log_checkpoint("inference_core", "backend_offline")
        st.error("> CRITICAL_FAILURE: BACKEND ENGINE OFFLINE.")
