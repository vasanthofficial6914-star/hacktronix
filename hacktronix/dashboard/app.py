"""
Streamlit Dashboard for HackModel-AI.

Displays:
- Live camera/synthetic frame with detection overlays
- World Model entity browser
- Interactive Knowledge Graph (PyVis HTML embed)
- Agent objective control and step-by-step reasoning log
- Inventory and state timeline
- System statistics
"""

import os
import sys
import json
import time
import base64

import streamlit as st
import numpy as np
import cv2

# ─── Configure page (must be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="HackModel-AI | World Modeling Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Import project modules (add project root to sys.path) ───────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hacktronix.application.world_model_service import build_world_model_stack

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1117 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.action-box {
    background: #1a1f2e;
    border-left: 4px solid #4299e1;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
}
.entity-chip {
    display: inline-block;
    background: #2d3748;
    color: #63b3ed;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.78rem;
}
.goal-banner {
    background: linear-gradient(90deg, #22543d, #276749);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    color: #68d391;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 10px 0;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Session State / Service Stack ───────────────────────────────────────────
@st.cache_resource
def get_stack():
    return build_world_model_stack(db_path="data/world_model_dashboard.db")

stack = get_stack()
repository   = stack["repository"]
graph_store  = stack["graph_store"]
extractor    = stack["extractor"]
updater      = stack["updater"]
query_layer  = stack["query_layer"]
text_env     = stack["text_env"]
video_env    = stack["video_env"]
agent        = stack["agent"]

if "reasoning_log" not in st.session_state:
    st.session_state.reasoning_log = []
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False
if "current_objective" not in st.session_state:
    st.session_state.current_objective = "Find and collect the EXIT GEM from the Throne Room"

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 HackModel-AI")
    st.markdown("**World Modeling for Autonomous Agents**")
    st.markdown("---")

    st.markdown("### 🎯 Agent Objective")
    objective = st.text_area(
        "Set Objective",
        value=st.session_state.current_objective,
        height=90,
        key="objective_input",
        label_visibility="collapsed",
    )
    st.session_state.current_objective = objective

    st.markdown("### ⚙️ Controls")
    col1, col2 = st.columns(2)
    with col1:
        step_btn = st.button("▶ Step", type="primary", use_container_width=True)
    with col2:
        auto_btn = st.button("⚡ Auto Run", use_container_width=True)

    reset_btn = st.button("🔄 Reset World", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 LLM Settings")
    use_ollama = st.toggle("Use Ollama LLM", value=False)
    if use_ollama:
        st.info("💡 Make sure `ollama serve` is running with gemma3:4b")
    else:
        st.success("✅ Using Deterministic Mock LLM (offline demo mode)")

    st.markdown("---")
    st.caption("HackTronix 2.0 | Track B | AI World Modeling")


# ─── Handle sidebar actions ──────────────────────────────────────────────────
if reset_btn:
    from hacktronix.environment.text_env import TextAdventureEnv
    stack["text_env"] = TextAdventureEnv()
    stack["agent"].env = stack["text_env"]
    text_env = stack["text_env"]
    st.session_state.reasoning_log = []
    st.success("🔄 World reset to initial state.")

if step_btn:
    # Seed if first step
    if not st.session_state.reasoning_log:
        raw_obs = text_env.observe()
        obs = extractor.extract_from_text_obs(raw_obs)
        updater.process_observation(obs)

    if not text_env.is_goal_achieved():
        step_result = agent.step_once(objective)
        st.session_state.reasoning_log.append(step_result)
    else:
        st.balloons()

if auto_btn:
    if not st.session_state.reasoning_log:
        raw_obs = text_env.observe()
        obs = extractor.extract_from_text_obs(raw_obs)
        updater.process_observation(obs)

    steps_run = 0
    progress = st.progress(0, text="Agent running...")
    max_steps = 15
    while steps_run < max_steps and not text_env.is_goal_achieved():
        step_result = agent.step_once(objective)
        st.session_state.reasoning_log.append(step_result)
        steps_run += 1
        progress.progress(steps_run / max_steps, text=f"Step {steps_run}/{max_steps}: {step_result.get('action', '')}")
        time.sleep(0.15)
    progress.empty()
    if text_env.is_goal_achieved():
        st.balloons()


# ─── Main Title ──────────────────────────────────────────────────────────────
st.markdown("# 🌍 HackModel-AI — World Model Dashboard")
st.markdown("*Persistent Self-Correcting World Model for Autonomous Agents | HackTronix 2.0 Track B*")

if text_env.is_goal_achieved():
    st.markdown('<div class="goal-banner">🏆 GOAL ACHIEVED — EXIT GEM COLLECTED!</div>', unsafe_allow_html=True)

# ─── Statistics Row ──────────────────────────────────────────────────────────
entities    = repository.get_all_entities()
rels        = repository.get_all_relationships()
inv         = repository.get_inventory()
history     = repository.get_state_history(limit=100)

cols = st.columns(6)
stats = [
    ("Entities", len(entities), "#63b3ed"),
    ("Relations", len(rels), "#68d391"),
    ("Graph Nodes", graph_store.graph.number_of_nodes(), "#f6ad55"),
    ("Graph Edges", graph_store.graph.number_of_edges(), "#fc8181"),
    ("Inventory", len(inv), "#b794f4"),
    ("Steps Taken", len(st.session_state.reasoning_log), "#76e4f7"),
]
for col, (label, val, color) in zip(cols, stats):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─── Main Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Agent Reasoning",
    "🗺️ World State",
    "🕸️ Knowledge Graph",
    "📹 Vision Stream",
    "📜 Timeline & History",
])

# ─────────────────── TAB 1: Agent Reasoning ─────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🎯 Current Objective")
        st.info(objective)

        st.markdown("### 📦 Agent Inventory")
        if inv:
            for item in inv:
                st.markdown(f'<span class="entity-chip">🎒 {item.name}</span>', unsafe_allow_html=True)
        else:
            st.caption("*Inventory is empty*")

        st.markdown("### 🗺️ Current Location")
        current_room_id = text_env.agent_room_id
        room_data = text_env.rooms.get(current_room_id)
        if room_data:
            st.markdown(f"**{room_data.name}**")
            st.caption(room_data.description)
            exits_str = " | ".join([f"**{d}** → {r}" for d, r in room_data.exits.items()])
            st.markdown(f"Exits: {exits_str}")
        
        st.markdown("### 👁️ World Slice Preview")
        ws = query_layer.retrieve_slice(objective, current_room_id=current_room_id)
        st.code(ws.format_as_text_slice(), language="text")

    with col_right:
        st.markdown("### 🧠 Reasoning Log")
        if not st.session_state.reasoning_log:
            st.info("Press **▶ Step** or **⚡ Auto Run** to start the agent.")
        else:
            for entry in reversed(st.session_state.reasoning_log[-10:]):
                step_n = entry.get("step", "?")
                room_n = entry.get("room", "?")
                action = entry.get("action", "?")
                result = entry.get("result", "?")
                reasoning = entry.get("reasoning", "?")
                goal = "✅ GOAL!" if entry.get("goal_achieved") else ""
                st.markdown(f"""
                <div class="action-box">
                    <b>Step {step_n}</b> @ {room_n} {goal}<br>
                    <span style="color:#63b3ed">ACTION:</span> {action}<br>
                    <span style="color:#68d391">RESULT:</span> {result}<br>
                    <span style="color:#fbd38d">REASONING:</span> {reasoning}
                </div>
                """, unsafe_allow_html=True)


# ─────────────────── TAB 2: World State ─────────────────────────────────────
with tab2:
    st.markdown("### 🏛️ All Entities in World Model")

    cat_filter = st.selectbox("Filter by Category", ["all", "room", "object", "person", "inventory", "unknown"])

    filtered = entities if cat_filter == "all" else [e for e in entities if e.category.value == cat_filter]

    if not filtered:
        st.info("No entities yet. Run the agent to build the World Model.")
    else:
        for ent in filtered:
            with st.expander(f"{'🏠' if ent.category.value == 'room' else '📦'} {ent.name} [{ent.category.value.upper()}] — conf: {ent.confidence.value:.2f}"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown(f"**ID:** `{ent.id}`")
                    st.markdown(f"**Room ID:** `{ent.room_id or 'N/A'}`")
                    st.markdown(f"**Confidence:** `{ent.confidence.value:.4f}`")
                with ec2:
                    if ent.states:
                        st.markdown("**States:**")
                        for k, v in ent.states.items():
                            val = v.value if hasattr(v, 'value') else str(v)
                            st.markdown(f"  - `{k}` = `{val}`")

    st.markdown("---")
    st.markdown("### 🔗 Relationships")
    if rels:
        rel_data = [{"Source": r.source_id, "Type": r.relation_type.value if hasattr(r.relation_type, 'value') else str(r.relation_type), "Target": r.target_id, "Confidence": f"{r.confidence.value:.3f}"} for r in rels[:30]]
        st.dataframe(rel_data, use_container_width=True)
    else:
        st.info("No relationships yet.")


# ─────────────────── TAB 3: Knowledge Graph ──────────────────────────────────
with tab3:
    st.markdown("### 🕸️ Live Knowledge Graph (Interactive)")
    if graph_store.graph.number_of_nodes() == 0:
        st.info("🔴 Knowledge Graph is empty. Run the agent to populate it.")
    else:
        graph_path = "data/live_graph.html"
        os.makedirs("data", exist_ok=True)
        graph_store.export_pyvis_html(graph_path)
        with open(graph_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.metric("Graph Nodes", graph_store.graph.number_of_nodes())
    with gcol2:
        st.metric("Graph Edges", graph_store.graph.number_of_edges())


# ─────────────────── TAB 4: Vision Stream ────────────────────────────────────
with tab4:
    st.markdown("### 📹 Vision World Modeler")
    st.caption("YOLOv11 + MediaPipe + OpenCV — Live Detection & World Model Update")

    vcol1, vcol2 = st.columns([2, 1])

    with vcol1:
        capture_btn = st.button("📸 Capture Frame & Detect")
        frame_placeholder = st.empty()

        obs_result = video_env.get_observation()
        frame = obs_result.get("annotated_frame")
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption="Current Scene (Synthetic/Live)", use_column_width=True)

    with vcol2:
        st.markdown("#### 🔍 Detections")
        detections = obs_result.get("detections", {})
        all_det = detections.get("objects", []) + detections.get("people", []) + detections.get("faces", [])

        if all_det:
            for det in all_det:
                icon = "👤" if det["category"] == "person" else "😊" if det["category"] == "face" else "📦"
                st.markdown(f"{icon} **{det['name']}** ({det['confidence']:.2f})")
        else:
            st.caption("*No detections in current frame*")

    if capture_btn:
        vis_obs = extractor.extract_from_vision_obs(detections)
        summary = updater.process_observation(vis_obs)
        st.success(f"✅ World Model Updated: +{len(summary['added'])} added, {len(summary['updated'])} updated")


# ─────────────────── TAB 5: Timeline & History ───────────────────────────────
with tab5:
    st.markdown("### 📜 State Version History")
    if history:
        hist_data = [
            {
                "Version": e.version_id,
                "Event": e.event_type,
                "Entity": e.entity_id or "—",
                "Description": e.description[:60],
                "Timestamp": time.strftime("%H:%M:%S", time.localtime(e.timestamp)),
            }
            for e in history
        ]
        st.dataframe(hist_data, use_container_width=True)
    else:
        st.info("No history yet. Run the agent to generate state versions.")

    st.markdown("### 🕐 Observation Timeline")
    timeline = repository.get_timeline(limit=20)
    if timeline:
        for t in timeline[:10]:
            with st.expander(f"[{t.source_type.upper()}] {time.strftime('%H:%M:%S', time.localtime(t.timestamp))}"):
                st.code(t.raw_observation[:300], language="text")
    else:
        st.info("No timeline events yet.")
