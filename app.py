

import json
import logging
import os
import sys
import time
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.llm_backend import (
    get_llm_response, HF_MODELS, ollama_available, ollama_list_models,
    OLLAMA_DEFAULT_MODEL,
)
from pipeline.layer5_knowledge_graph import configure_neo4j


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


st.set_page_config(
    page_title="ThinkSym | LLM Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ThinkSym — Neurosymbolic Defense Framework for LLM Security\n\nMCA Final Project",
    },
)


st.markdown("""
<style>
/* ─── Global ───────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117 !important;
    color: #e2e8f0;
}

/* ─── Header ────────────────────────────────────── */
.thinksym-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.thinksym-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #89b4fa, #cba6f7, #f38ba8, #fab387);
}
.thinksym-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #89b4fa 0%, #cba6f7 50%, #f38ba8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.thinksym-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 6px;
    font-weight: 400;
}
.version-badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 12px;
    vertical-align: middle;
}

/* ─── Layer Cards ────────────────────────────────── */
.layer-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: border-color 0.3s ease;
}
.layer-card:hover {
    border-color: #4a5568;
}
.layer-card-active {
    border-color: #89b4fa;
    box-shadow: 0 0 0 1px #89b4fa22;
}
.layer-card-done {
    border-color: #a6e3a1;
}
.layer-card-warn {
    border-color: #f9e2af;
}
.layer-card-block {
    border-color: #f38ba8;
}

/* ─── Verdict Card ───────────────────────────────── */
.verdict-card {
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.verdict-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
}
.verdict-allow  { border-color: #a6e3a1; background: #0f1a14; }
.verdict-allow::before  { background: linear-gradient(90deg, #a6e3a1, #94e2d5); }
.verdict-warn   { border-color: #f9e2af; background: #1a1a0f; }
.verdict-warn::before   { background: linear-gradient(90deg, #f9e2af, #fab387); }
.verdict-block  { border-color: #f38ba8; background: #1a0f14; }
.verdict-block::before  { background: linear-gradient(90deg, #f38ba8, #cba6f7); }
.verdict-title  { font-size: 1.4rem; font-weight: 700; margin: 0; }
.verdict-score  { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 600; }
.verdict-reason { font-size: 0.9rem; color: #94a3b8; margin-top: 8px; }

/* ─── Metric Tiles ───────────────────────────────── */
.metric-tile {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #89b4fa;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ─── Code/Trace blocks ───────────────────────────── */
.trace-block {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #e6edf3;
    overflow-x: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* ─── Pipeline flow ───────────────────────────────── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
}
.step-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4a5568;
}
.step-active  { background: #89b4fa; box-shadow: 0 0 8px #89b4fa88; }
.step-done    { background: #a6e3a1; }
.step-blocked { background: #f38ba8; }

/* ─── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d111b !important;
    border-right: 1px solid #1e2435 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #89b4fa !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

/* ─── Progress bar ────────────────────────────────── */
.stProgress > div > div > div > div {
    background-color: #89b4fa !important;
}

/* ─── Buttons ─────────────────────────────────────── */
button[kind="primaryFormSubmit"], button[kind="primary"] {
    background: linear-gradient(135deg, #89b4fa, #cba6f7) !important;
    border: none !important;
    color: #0f1117 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ─── Expanders ─────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
}

/* ─── Tags ─────────────────────────────────────── */
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}
.tag-critical { background: #3d1a1a; color: #f38ba8; border: 1px solid #f38ba844; }
.tag-high     { background: #3d2a1a; color: #fab387; border: 1px solid #fab38744; }
.tag-medium   { background: #3d3a1a; color: #f9e2af; border: 1px solid #f9e2af44; }
.tag-low      { background: #1a3d2a; color: #a6e3a1; border: 1px solid #a6e3a144; }
.tag-safe     { background: #1a2a3d; color: #89dceb; border: 1px solid #89dceb44; }
.tag-info     { background: #1a2a3d; color: #89b4fa; border: 1px solid #89b4fa44; }
</style>
""", unsafe_allow_html=True)


from config import (
    APP_VERSION,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RISK_LEVELS,
)
from pipeline.orchestrator import ThinkSymOrchestrator, AnalysisResult


DATA_PATH = Path("data/sample_prompts.json")

@st.cache_data
def load_sample_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Initializing ThinkSym (loading ModernBERT…)")
def get_orchestrator(device: str = "cuda", build_id: str = APP_VERSION) -> ThinkSymOrchestrator:
    _ = build_id
    return ThinkSymOrchestrator(device=device, session_id="demo_session")


def risk_color(level: str) -> str:
    return RISK_LEVELS.get(level, RISK_LEVELS["SAFE"])["color"]

def risk_emoji(level: str) -> str:
    return RISK_LEVELS.get(level, RISK_LEVELS["SAFE"])["emoji"]

def decision_css_class(decision: str) -> str:
    return {"ALLOW": "verdict-allow", "WARN": "verdict-warn", "BLOCK": "verdict-block"}.get(decision, "verdict-allow")


def _threat_radar_chart(neural_result) -> go.Figure:
    categories = [ts.category.replace("_", " ").title() for ts in neural_result.threat_scores.values()]
    scores     = [ts.combined_score for ts in neural_result.threat_scores.values()]
    fig = go.Figure(go.Scatterpolar(
        r      = scores + [scores[0]],
        theta  = categories + [categories[0]],
        fill   = "toself",
        line_color = "#89b4fa",
        fillcolor  = "rgba(137, 180, 250, 0.15)",
        name   = "Threat Score",
    ))
    fig.update_layout(
        polar = dict(
            bgcolor    = "#1a1f2e",
            radialaxis = dict(visible=True, range=[0, 1], tickfont=dict(color="#64748b", size=9), gridcolor="#2d3748"),
            angularaxis = dict(tickfont=dict(color="#94a3b8", size=10), gridcolor="#2d3748"),
        ),
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#0f1117",
        font          = dict(color="#e2e8f0"),
        showlegend    = False,
        margin        = dict(t=30, b=30, l=30, r=30),
        height        = 320,
    )
    return fig


def _threat_bar_chart(neural_result) -> go.Figure:
    cats   = [ts.category.replace("_", " ").title() for ts in neural_result.threat_scores.values()]
    scores = [ts.combined_score for ts in neural_result.threat_scores.values()]
    colors = [
        "#f38ba8" if s >= 0.70 else
        "#fab387" if s >= 0.50 else
        "#f9e2af" if s >= 0.30 else
        "#a6e3a1" if s >= 0.10 else
        "#4a5568"
        for s in scores
    ]
    fig = go.Figure(go.Bar(
        x           = cats,
        y           = scores,
        marker_color = colors,
        text        = [f"{s:.3f}" for s in scores],
        textposition = "outside",
        textfont    = dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#1a1f2e",
        font          = dict(color="#e2e8f0"),
        xaxis = dict(
            tickfont    = dict(size=9),
            gridcolor   = "#2d3748",
            tickangle   = -35,
        ),
        yaxis = dict(
            range       = [0, 1.1],
            gridcolor   = "#2d3748",
            tickfont    = dict(size=9),
        ),
        margin = dict(t=20, b=80, l=30, r=20),
        height = 280,
        bargap = 0.3,
        shapes=[dict(
            type="line", y0=0.45, y1=0.45, x0=-0.5, x1=len(cats) - 0.5,
            line=dict(color="rgba(243,139,168,0.4)", dash="dash", width=1)
        )],
    )
    return fig


def _session_timeline_chart(history: list) -> go.Figure:
    turns  = [r.turn_number + 1 for r in history]
    scores = [r.verdict.composite_score if r.verdict else 0 for r in history]
    colors = [
        risk_color(r.verdict.risk_level if r.verdict else "SAFE")
        for r in history
    ]
    symbols = [
        "x" if (r.verdict and r.verdict.decision == "BLOCK") else
        "diamond" if (r.verdict and r.verdict.decision == "WARN") else
        "circle"
        for r in history
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turns, y=scores,
        mode="lines+markers",
        line=dict(color="#89b4fa", width=2),
        marker=dict(size=12, color=colors, symbol=symbols, line=dict(color="#0f1117", width=1)),
        name="Composite Score",
        hovertemplate="Turn %{x}<br>Score: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.70, line_dash="dash", line_color="rgba(243,139,168,0.4)", annotation_text="Block threshold", annotation_font_color="#f38ba8")
    fig.add_hline(y=0.40, line_dash="dash", line_color="rgba(249,226,175,0.4)", annotation_text="Warn threshold", annotation_font_color="#f9e2af")
    fig.update_layout(
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#1a1f2e",
        font          = dict(color="#e2e8f0"),
        xaxis = dict(title="Turn", gridcolor="#2d3748", tickfont=dict(size=10)),
        yaxis = dict(title="Threat Score", range=[0, 1.05], gridcolor="#2d3748"),
        margin = dict(t=10, b=40, l=50, r=20),
        height = 220,
        showlegend = False,
    )
    return fig


def _display_layer1(result: AnalysisResult):
    sr = result.smooth_result
    if not sr:
        st.info("SmoothLLM layer disabled")
        return
    flag_colors = {"PROCEED": "#a6e3a1", "WARN": "#f9e2af", "BLOCK": "#f38ba8"}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Perturbations",      sr.n_perturbations)
    col2.metric("Structural Fragility", f"{sr.structural_fragility:.3f}")
    col3.metric("Semantic Coherence",  f"{sr.semantic_coherence_score:.3f}")
    col4.metric("Recommendation",      sr.recommendation)

    st.markdown(f"""
    <div style="margin-top:12px;">
        <span class="tag tag-info">Char Budget: {sr.perturbation_budget*100:.0f}%</span>
        <span class="tag tag-info">Avg Edit Distance: {sr.avg_char_edit_distance:.1f}</span>
        <span class="tag tag-info">Fragility Level: {sr.flag_level}</span>
        <span class="tag {'tag-critical' if sr.flagged_as_fragile else 'tag-safe'}">
            {'⚠ Fragile Structure' if sr.flagged_as_fragile else '✓ Robust Structure'}
        </span>
        <span class="tag tag-info">⏱ {sr.processing_time_ms:.0f}ms</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Perturbation Mode Distribution", expanded=False):
        mode_counts: dict[str, int] = {}
        for copy in sr.perturbed_copies:
            mode_counts[copy.perturbation_mode] = mode_counts.get(copy.perturbation_mode, 0) + 1
        mode_fig = go.Figure(go.Bar(
            x=list(mode_counts.keys()), y=list(mode_counts.values()),
            marker_color="#89b4fa",
        ))
        mode_fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=160, margin=dict(t=10, b=10, l=10, r=10),
            font=dict(color="#e2e8f0", size=10),
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(gridcolor="#2d3748"),
        )
        st.plotly_chart(mode_fig, use_container_width=True)


def _display_layer2(result: AnalysisResult):
    nr = result.neural_result
    if not nr:
        st.info("Neural classifier layer disabled")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top Threat",          nr.top_threat_category.replace("_"," ").title())
    col2.metric("Top Threat Score",    f"{nr.top_threat_score:.3f}")
    col3.metric("Composite Threat",    f"{nr.composite_threat_score:.3f}")
    col4.metric("Verdict",             "⚠ ADVERSARIAL" if nr.is_adversarial else "✓ BENIGN")

    st.markdown(f"""
    <div style="margin-top:12px;">
        <span class="tag tag-info">Model: ModernBERT-base</span>
        <span class="tag tag-info">Device: {nr.device_used.upper()}</span>
        <span class="tag tag-info">Inference: {nr.inference_time_ms:.0f}ms</span>
        <span class="tag {'tag-critical' if nr.is_adversarial else 'tag-safe'}">
            {'⚠ Adversarial' if nr.is_adversarial else '✓ Benign'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Threat Scores (Bar)", "Threat Radar"])
    with tab1:
        st.plotly_chart(_threat_bar_chart(nr), use_container_width=True)
    with tab2:
        st.plotly_chart(_threat_radar_chart(nr), use_container_width=True)

    with st.expander("Keyword Matches", expanded=False):
        matched = [(cat, ts) for cat, ts in nr.threat_scores.items() if ts.matched_keywords]
        if matched:
            for cat, ts in sorted(matched, key=lambda x: x[1].combined_score, reverse=True):
                kws = ", ".join(f"`{kw}`" for kw in ts.matched_keywords[:5])
                st.markdown(f"**{cat.replace('_',' ').title()}** ({ts.combined_score:.3f}): {kws}")
        else:
            st.info("No keyword matches found — signal driven by semantic similarity.")


def _display_layer3(result: AnalysisResult):
    sr = result.scallop_result
    if not sr:
        st.info("Scallop reasoner layer disabled")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbolic Risk",        f"{sr.symbolic_risk_score:.3f}")
    col2.metric("Policy Violations",    len(sr.violated_policies))
    col3.metric("Multi-turn Attack",    "DETECTED" if sr.multi_turn_attack_detected else "None")
    col4.metric("Engine",               sr.engine_used)

    if sr.violated_policies:
        st.markdown("**Violated Policies:**")
        for p in sr.violated_policies:
            st.markdown(f"""<span class="tag tag-critical">🚨 {p}</span>""", unsafe_allow_html=True)
        st.markdown("")

    if sr.multi_turn_attack_detected and sr.escalation_evidence:
        ev = sr.escalation_evidence
        st.markdown(f"""
        <div class="layer-card layer-card-block" style="margin-top:12px;">
            <b style="color:#f38ba8;">🔥 Multi-turn Escalation Detected</b><br>
            <span style="font-size:0.85rem;color:#94a3b8;">
                Category: <b>{ev.threat_categories[0]}</b> |
                Peak Score: <b>{ev.peak_score:.3f}</b> |
                Escalation Rate: <b>{ev.escalation_rate:+.3f}/turn</b><br>
                Score trajectory: {" → ".join(f"{s:.3f}" for s in ev.score_trajectory)}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:12px;">
        <span class="tag tag-info">Turn #{sr.session_turn_count}</span>
        <span class="tag {'tag-critical' if sr.symbolic_risk_score >= 0.6 else 'tag-medium' if sr.symbolic_risk_score >= 0.4 else 'tag-safe'}">
            Risk: {sr.risk_label}
        </span>
        <span class="tag tag-info">⏱ {sr.processing_time_ms:.0f}ms</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Datalog Rules Applied", expanded=False):
        for rule in sr.rules_applied:
            st.markdown(f"```prolog\n{rule}\n```")

    with st.expander("Scallop Program", expanded=False):
        st.markdown(f"""<div class="trace-block">{sr.scallop_program.strip()}</div>""",
                    unsafe_allow_html=True)


def _display_layer4(result: AnalysisResult):
    zr = result.z3_result
    if not zr:
        st.info("Z3 verifier layer disabled")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Formal Verdict",       zr.formal_verdict)
    col2.metric("Policies Triggered",   f"{zr.triggered_policies}/{zr.total_policies}")
    col3.metric("Verification Score",   f"{zr.verification_score:.3f}")
    col4.metric("Z3 Version",           zr.z3_version)


    for pr in zr.policy_results:
        icon  = "🔴" if pr.triggered else "✅"
        color = "#f38ba8" if pr.triggered else "#a6e3a1"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0;border-bottom:1px solid #1e2435;">
            <span style="width:24px;">{icon}</span>
            <span style="font-weight:600;color:{color};width:80px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;">{pr.policy_id}</span>
            <span style="flex:1;font-size:0.9rem;">{pr.policy_name}</span>
            <span style="color:#64748b;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">{pr.condition_str}</span>
            <span style="width:60px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:{color};">{pr.smt_result}</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Z3 Formal Proof Trace", expanded=False):
        st.markdown(f"""<div class="trace-block">{zr.formal_proof}</div>""",
                    unsafe_allow_html=True)


def _display_layer5(result: AnalysisResult):
    kr = result.kg_result
    if not kr:
        st.info("Knowledge graph layer disabled")
        return
    gs = kr.graph_stats

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched Patterns",   kr.total_matches)
    col2.metric("OWASP Categories",   len(kr.owasp_hits))
    col3.metric("KG Risk Score",      f"{kr.kg_risk_score:.3f}")
    col4.metric("Graph Size",         f"{gs.total_nodes}N / {gs.total_edges}E" if gs else "—")


    if kr.owasp_hits or kr.mitre_hits:
        st.markdown("")
        for h in kr.owasp_hits:
            st.markdown(f"""<span class="tag tag-critical">{h}</span>""", unsafe_allow_html=True)
        for h in kr.mitre_hits:
            st.markdown(f"""<span class="tag tag-high">{h}</span>""", unsafe_allow_html=True)
        st.markdown("")


    if kr.matched_attacks:
        st.markdown("**Top Matched Attack Patterns:**")
        for match in kr.matched_attacks[:6]:
            sev_class = {
                "CRITICAL": "tag-critical", "HIGH": "tag-high",
                "MEDIUM": "tag-medium", "LOW": "tag-low"
            }.get(match.severity, "tag-info")
            st.markdown(f"""
            <div style="padding:8px 0;border-bottom:1px solid #1e2435;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    <span style="font-weight:600;font-size:0.9rem;">{match.name}</span>
                    <span class="tag {sev_class}">{match.severity}</span>
                    <span class="tag tag-info">{match.node_type}</span>
                    {f'<span class="tag tag-high">{match.owasp_id}</span>' if match.owasp_id else ''}
                    {f'<span class="tag tag-medium">{match.mitre_id}</span>' if match.mitre_id else ''}
                    <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#64748b;">conf: {match.confidence:.3f}</span>
                </div>
                <div style="font-size:0.8rem;color:#64748b;">
                    Matched via: {", ".join(f'<i>{kw}</i>' for kw in match.matched_via[:3])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:12px;font-size:0.85rem;color:#94a3b8;">
        {kr.attack_vector_summary}
    </div>
    """, unsafe_allow_html=True)


def _display_verdict(result: AnalysisResult):
    v = result.verdict
    if not v:
        return

    css_class = decision_css_class(v.decision)
    decision_icon = {"ALLOW": "✅", "WARN": "⚠️", "BLOCK": "🚫"}.get(v.decision, "—")

    st.markdown(f"""
    <div class="verdict-card {css_class}">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div>
                <p class="verdict-title">{decision_icon} {v.decision} — {v.risk_level}</p>
                <p class="verdict-reason">{v.primary_reason}</p>
            </div>
            <div style="text-align:right;">
                <div class="verdict-score">{v.composite_score:.3f}</div>
                <div style="font-size:0.75rem;color:#64748b;">Composite Threat Score</div>
            </div>
        </div>
        <hr style="border-color:#2d3748;margin:16px 0;">
        <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
            <span style="font-size:0.8rem;color:#64748b;margin-right:4px;">Layer signals:</span>
            <span class="tag {'tag-critical' if v.smoothllm_flag == 'BLOCK' else 'tag-medium' if v.smoothllm_flag == 'WARN' else 'tag-safe'}">
                L1: {v.smoothllm_flag}
            </span>
            <span class="tag {'tag-critical' if v.neural_flag else 'tag-safe'}">
                L2: {'THREAT' if v.neural_flag else 'CLEAR'}
            </span>
            <span class="tag {'tag-critical' if v.scallop_flag in ('CRITICAL','HIGH') else 'tag-medium' if v.scallop_flag=='MEDIUM' else 'tag-safe'}">
                L3: {v.scallop_flag}
            </span>
            <span class="tag {'tag-critical' if v.z3_flag == 'BLOCK' else 'tag-safe'}">
                L4: {v.z3_flag}
            </span>
            <span class="tag {'tag-critical' if v.kg_flag in ('CRITICAL','HIGH') else 'tag-medium' if v.kg_flag=='MEDIUM' else 'tag-safe'}">
                L5: {v.kg_flag}
            </span>
            <span class="tag tag-info" style="margin-left:auto;">
                Confidence: {v.confidence:.0%}
            </span>
            <span class="tag tag-info">⏱ {result.total_time_ms:.0f}ms</span>
        </div>
        {'<div style="margin-top:12px;"><span style="font-size:0.8rem;color:#64748b;">Contributing factors: </span>' +
         " | ".join(f'<span style="font-size:0.8rem;color:#94a3b8;">{f}</span>' for f in v.contributing_factors) +
         '</div>' if v.contributing_factors else ''}
    </div>
    """, unsafe_allow_html=True)


def main():
    if st.session_state.get("pipeline_build_id") != APP_VERSION:
        get_orchestrator.clear()
        st.session_state.pipeline_build_id = APP_VERSION
        for key in ["analysis_result", "session_results", "session_stats", "chat_messages"]:
            st.session_state.pop(key, None)


    st.markdown(f"""
    <div class="thinksym-header">
        <h1 class="thinksym-title">🛡 ThinkSym <span class="version-badge">v{APP_VERSION}</span></h1>
        <p class="thinksym-subtitle">Neurosymbolic Defense Framework for Large Language Model Security</p>
        <p style="font-size:0.78rem;color:#475569;margin-top:8px;">
            ModernBERT · Scallop Datalog · Z3 SMT · OWASP LLM Top 10 · MITRE ATT&CK for AI · HarmBench
        </p>
    </div>
    """, unsafe_allow_html=True)


    with st.sidebar:
        st.markdown("### ⚙ Configuration")

        device = st.selectbox(
            "Compute Device",
            ["cuda", "cpu"],
            index=0,
            help="GPU strongly recommended for ModernBERT inference",
        )

        st.markdown("---")
        st.markdown("### 🤖 LLM Backend")


        _ollama_up = ollama_available()
        if _ollama_up:
            _ollama_models = ollama_list_models()
            st.success("🟢 Ollama running")
            if _ollama_models:
                ollama_model_sel = st.selectbox(
                    "Local model",
                    _ollama_models,
                    index=next(
                        (i for i, m in enumerate(_ollama_models) if "llama3.2" in m or "llama3" in m),
                        0,
                    ),
                )
            else:
                st.warning("No models pulled yet. Run: `ollama pull llama3.2:3b`")
                ollama_model_sel = OLLAMA_DEFAULT_MODEL
        else:
            st.warning("⚫ Ollama not running")
            st.caption("Start it: open a terminal → `ollama serve` then `ollama pull llama3.2:3b`")
            ollama_model_sel = OLLAMA_DEFAULT_MODEL


        with st.expander("HuggingFace fallback (optional)", expanded=not _ollama_up):
            hf_token = st.text_input(
                "HF API Token",
                type="password",
                placeholder="hf_...",
                help="Only used when Ollama is unavailable.",
            )
            llm_model = st.selectbox("HF Model", list(HF_MODELS.keys()), index=0)

        st.markdown("---")
        st.markdown("### 🔧 Pipeline Layers")
        enable_smooth  = st.toggle("L1 · SmoothLLM",        value=True)
        enable_neural  = st.toggle("L2 · ModernBERT Neural", value=True)
        enable_scallop = st.toggle("L3 · Scallop Symbolic",  value=True)
        enable_z3      = st.toggle("L4 · Z3 Formal Verify",  value=True)
        enable_kg      = st.toggle("L5 · Knowledge Graph",   value=True)

        st.markdown("---")
        st.markdown("### 🗄️ Neo4j Knowledge Graph")
        if "neo4j_connected" not in st.session_state:
            st.session_state.neo4j_connected = False
            if NEO4J_PASSWORD.strip():
                st.session_state.neo4j_connected = configure_neo4j(
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
                )

        neo4j_uri  = st.text_input("URI",       value=NEO4J_URI)
        neo4j_user = st.text_input("Username",  value=NEO4J_USER)
        neo4j_pass = st.text_input("Password",  type="password", value=NEO4J_PASSWORD, placeholder="your Neo4j password")
        neo4j_db   = st.text_input("Database",  value=NEO4J_DATABASE)

        if st.button("🔌 Connect to Neo4j", use_container_width=True):
            if neo4j_pass.strip():
                with st.spinner("Connecting…"):
                    ok = configure_neo4j(neo4j_uri, neo4j_user, neo4j_pass, neo4j_db)
                st.session_state.neo4j_connected = ok
                if ok:
                    st.success("✅ Connected — graph will rebuild in Neo4j on next analysis.")
                else:
                    st.error("❌ Connection failed. Check URI / credentials and that Neo4j is running.")
            else:
                st.warning("Enter a password first.")

        if st.session_state.neo4j_connected:
            st.caption("🟢 Neo4j backend connected")

        if not neo4j_pass.strip():
            st.caption("🔵 Using NetworkX (in-memory). Connect Neo4j above to use the database backend.")

        st.markdown("---")
        st.markdown("### 📋 Sample Prompts")
        samples = load_sample_prompts()
        adv_samples = samples.get("adversarial", [])
        ben_samples = samples.get("benign", [])

        sample_mode = st.radio("Category", ["Adversarial", "Benign"], horizontal=True)
        sample_list = adv_samples if sample_mode == "Adversarial" else ben_samples

        sample_labels = [f"[{s.get('attack_type', s.get('category',''))}] {s['prompt'][:50]}…"
                         for s in sample_list]
        selected_idx = st.selectbox("Choose sample", range(len(sample_labels)),
                                    format_func=lambda i: sample_labels[i])
        if st.button("Load Sample →", use_container_width=True):
            st.session_state.prefill_prompt = sample_list[selected_idx]["prompt"]

        st.markdown("---")
        st.markdown("### 🎭 Multi-turn Demo")
        if st.button("Load Escalation Sequence", use_container_width=True):
            st.session_state.load_mturn = True

        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):

            get_orchestrator.clear()
            for key in ["analysis_result", "session_results", "session_stats", "prefill_prompt"]:
                st.session_state.pop(key, None)
            st.success("Session reset.")
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem;color:#4a5568;text-align:center;padding-top:8px;">
            MCA Final Project<br>Neurosymbolic LLM Security
        </div>
        """, unsafe_allow_html=True)


    orchestrator = get_orchestrator(device, APP_VERSION)


    if "session_results" not in st.session_state:
        st.session_state.session_results = []


    tab_chat, tab_analyze, tab_session, tab_mturn, tab_benchmark = st.tabs([
        "💬 Secure Chat",
        "🔍 Single Analysis",
        "📊 Session Overview",
        "🎭 Multi-turn Demo",
        "📈 Benchmark Mode",
    ])


    with tab_chat:
        _ollama_ready = ollama_available() and bool(ollama_list_models())
        if not _ollama_ready and not (hf_token or os.getenv("HF_TOKEN", "")):
            st.warning(
                "⚫ **No LLM backend available.** "
                "Start Ollama in a terminal: `ollama serve` then `ollama pull llama3.2:3b`. "
                "The pipeline analysis still works; you just won't get LLM responses until a backend is connected."
            )
        else:
            backend_label = f"Ollama · {ollama_model_sel}" if _ollama_ready else f"HF API · {llm_model}"
            st.caption(f"LLM backend: **{backend_label}**  ·  ThinkSym intercepts threats before they reach it.")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []


        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                if msg.get("unsafe_html"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])


        chat_prompt = st.chat_input("Send a message to the guarded LLM…")
        if chat_prompt:

            st.session_state.chat_messages.append({"role": "user", "content": chat_prompt})


            with st.spinner("🛡 ThinkSym analyzing prompt…"):
                chat_result = orchestrator.analyze(chat_prompt, update_session=True)
                st.session_state.session_results.append(chat_result)
                st.session_state.session_stats = orchestrator.get_session_stats()

            v = chat_result.verdict
            decision = v.decision if v else "ALLOW"

            if decision == "BLOCK":
                _scallop = chat_result.scallop_result
                _vp = _scallop.violated_policies if _scallop else []
                violated = ", ".join(_vp[:3]) if _vp else "policy violation"
                block_html = f"""
<div style="border:2px solid #f38ba8;border-radius:10px;padding:16px 20px;background:#1a0a0f;margin:4px 0;">
  <div style="color:#f38ba8;font-weight:700;font-size:1rem;margin-bottom:8px;">🚫 Request Blocked by ThinkSym</div>
  <div style="color:#cbd5e1;font-size:0.88rem;margin-bottom:12px;">{v.primary_reason if v else 'Security policy violation detected.'}</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
    <span style="background:#2d1f22;color:#f38ba8;border-radius:4px;padding:2px 8px;font-size:0.78rem;font-family:'JetBrains Mono',monospace;">Score: {f"{v.composite_score:.3f}" if v else '—'}</span>
    <span style="background:#2d1f22;color:#f38ba8;border-radius:4px;padding:2px 8px;font-size:0.78rem;font-family:'JetBrains Mono',monospace;">Risk: {v.risk_level if v else '—'}</span>
    <span style="background:#2d1f22;color:#f38ba8;border-radius:4px;padding:2px 8px;font-size:0.78rem;font-family:'JetBrains Mono',monospace;">Policies: {violated}</span>
  </div>
  <div style="color:#64748b;font-size:0.78rem;">The prompt was intercepted — no LLM was contacted.</div>
</div>"""
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": block_html, "unsafe_html": True,
                })

            elif decision == "WARN":
                with st.spinner("⚠ Elevated risk — forwarding to LLM with caution…"):
                    llm_resp = get_llm_response(
                        chat_prompt, hf_token or None, llm_model,
                        ollama_model=ollama_model_sel,
                    )
                if not llm_resp.success:
                    content = f"⚠ LLM backend error: {llm_resp.error}"
                    st.session_state.chat_messages.append({"role": "assistant", "content": content})
                else:
                    warn_prefix = f"""<div style="border:1px solid #f9e2af;border-radius:8px;padding:8px 14px;background:#1a1708;margin-bottom:10px;">
    <span style="color:#f9e2af;font-size:0.8rem;">⚠ Elevated risk (score: {f"{v.composite_score:.3f}" if v else '—'}) — response shown with caution.</span>
</div>
"""
                    content = warn_prefix + llm_resp.text
                    st.session_state.chat_messages.append({
                        "role": "assistant", "content": content, "unsafe_html": True,
                    })

            else:
                with st.spinner("💬 Generating response…"):
                    llm_resp = get_llm_response(
                        chat_prompt, hf_token or None, llm_model,
                        ollama_model=ollama_model_sel,
                    )
                if not llm_resp.success:
                    content = f"❌ LLM backend error: {llm_resp.error}"
                    st.session_state.chat_messages.append({"role": "assistant", "content": content})
                else:
                    allow_suffix = f"\n\n---\n*ThinkSym passed · score {f'{v.composite_score:.3f}' if v else '—'} · {v.risk_level if v else 'SAFE'} · via {llm_resp.backend}*"
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": llm_resp.text + allow_suffix,
                    })

            st.rerun()


    with tab_analyze:
        prefill = st.session_state.pop("prefill_prompt", "")
        prompt = st.text_area(
            "Enter a prompt to analyze",
            value=prefill,
            height=120,
            placeholder="Type a prompt or load a sample from the sidebar…",
            label_visibility="collapsed",
        )

        analyze_btn = st.button("🛡 Analyze Prompt", type="primary", use_container_width=True)

        if analyze_btn and prompt.strip():
            result = _run_analysis_with_progress(
                prompt, orchestrator,
                enable_smooth, enable_neural, enable_scallop, enable_z3, enable_kg,
            )
            st.session_state.analysis_result = result
            st.session_state.session_results.append(result)
            st.session_state.session_stats = orchestrator.get_session_stats()

        elif analyze_btn:
            st.warning("Please enter a prompt first.")


        if "analysis_result" in st.session_state:
            result: AnalysisResult = st.session_state.analysis_result
            _render_full_result(result)


    with tab_session:
        stats = st.session_state.get("session_stats", {})
        session_results = st.session_state.get("session_results", [])

        if not stats:
            st.info("No prompts analyzed yet. Use the 'Single Analysis' tab to start.")
        else:

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{stats.get('turns',0)}</div>
                <div class="metric-label">Total Turns</div></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:#f38ba8;">{stats.get('blocked',0)}</div>
                <div class="metric-label">BLOCKED</div></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:#f9e2af;">{stats.get('warned',0)}</div>
                <div class="metric-label">WARNED</div></div>""", unsafe_allow_html=True)
            c4.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:#a6e3a1;">{stats.get('allowed',0)}</div>
                <div class="metric-label">ALLOWED</div></div>""", unsafe_allow_html=True)
            c5.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{stats.get('block_rate',0):.0%}</div>
                <div class="metric-label">Block Rate</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            if len(session_results) > 1:
                st.markdown("#### Threat Score Timeline")
                st.plotly_chart(_session_timeline_chart(session_results), use_container_width=True)


            if session_results:
                st.markdown("#### Analysis History")
                for r in reversed(session_results[-10:]):
                    v = r.verdict
                    if not v:
                        continue
                    icon = {"ALLOW": "✅", "WARN": "⚠️", "BLOCK": "🚫"}[v.decision]
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #1e2435;font-size:0.85rem;">
                        <span style="color:#64748b;font-family:'JetBrains Mono',monospace;width:50px;">T{r.turn_number+1:03d}</span>
                        <span>{icon}</span>
                        <span style="color:{risk_color(v.risk_level)};width:60px;font-weight:600;">{v.risk_level}</span>
                        <span style="font-family:'JetBrains Mono',monospace;color:#64748b;width:55px;">{v.composite_score:.3f}</span>
                        <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#94a3b8;">{r.prompt[:80]}</span>
                        <span style="color:#4a5568;font-size:0.75rem;">{r.total_time_ms:.0f}ms</span>
                    </div>
                    """, unsafe_allow_html=True)


    with tab_mturn:
        st.markdown("""
        ### 🎭 Multi-turn Escalation Attack Demo
        This demonstrates ThinkSym's unique capability: detecting **gradual escalation attacks**
        that span multiple conversation turns — a pattern that single-turn defenses miss.
        """)

        seq = samples.get("multi_turn_attack_sequence", {})
        turns_data = seq.get("turns", [])

        if turns_data:
            st.info(f"**Scenario**: {seq.get('description','')}\n\n"
                    f"5-turn conversation that progressively escalates to a critical attack.")
            if st.button("▶ Run Full Escalation Demo", type="primary"):

                get_orchestrator.clear()
                st.session_state.pop("multi_turn_results", None)
                orchestrator_fresh = get_orchestrator(device)

                mt_results = []
                progress = st.progress(0, text="Simulating escalation sequence…")
                turn_placeholder = st.empty()
                for i, turn in enumerate(turns_data):
                    progress.progress((i + 1) / len(turns_data), text=f"Analyzing turn {i+1}/{len(turns_data)}…")
                    res = orchestrator_fresh.analyze(turn["prompt"], update_session=True)
                    mt_results.append((turn, res))
                    time.sleep(0.3)
                progress.empty()
                st.session_state.multi_turn_results = mt_results
                st.rerun()

        if "multi_turn_results" in st.session_state:
            mt_results = st.session_state.multi_turn_results
            st.markdown("---")
            st.markdown("#### Escalation Sequence Results")

            for turn_data, res in mt_results:
                v = res.verdict
                icon = {"ALLOW": "✅", "WARN": "⚠️", "BLOCK": "🚫"}[v.decision] if v else "—"
                color = risk_color(v.risk_level) if v else "#4a5568"
                card_border = "#f38ba8" if v and v.decision == "BLOCK" else "#f9e2af" if v and v.decision == "WARN" else "#2d3748"

                st.markdown(f"""
                <div style="border:1px solid {card_border};border-radius:10px;padding:14px 18px;margin:8px 0;background:#1a1f2e;">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <span style="font-family:'JetBrains Mono',monospace;color:#64748b;font-size:0.75rem;">TURN {turn_data['turn']}</span>
                        <span class="tag {'tag-critical' if turn_data['risk_label']=='CRITICAL' else 'tag-high' if turn_data['risk_label']=='HIGH' else 'tag-medium' if turn_data['risk_label']=='MEDIUM' else 'tag-safe'}">
                            Stage {turn_data['escalation_stage']}: {turn_data['risk_label']}
                        </span>
                        <span style="margin-left:auto;">{icon}
                            <span style="color:{color};font-weight:600;margin-left:6px;">{v.decision if v else '—'}</span>
                            <span style="font-family:'JetBrains Mono',monospace;color:#64748b;margin-left:8px;font-size:0.85rem;">{f"{v.composite_score:.3f}" if v else '—'}</span>
                        </span>
                    </div>
                    <div style="font-size:0.85rem;color:#94a3b8;font-style:italic;">"{turn_data['prompt']}"</div>
                    {f'<div style="margin-top:8px;font-size:0.8rem;color:#64748b;">{v.primary_reason}</div>' if v else ''}
                    {f'<div style="margin-top:6px;"><span class="tag tag-critical">⚠ Temporal Escalation: {res.scallop_result.temporal_escalation_score:.3f}</span></div>' if res.scallop_result and res.scallop_result.multi_turn_attack_detected else ''}
                </div>
                """, unsafe_allow_html=True)


            if len(mt_results) >= 2:
                scores = [r.verdict.composite_score if r.verdict else 0 for _, r in mt_results]
                stages = [f"T{t['turn']}" for t, _ in mt_results]
                esc_fig = go.Figure()
                esc_fig.add_trace(go.Scatter(
                    x=stages, y=scores, mode="lines+markers+text",
                    text=[f"{s:.3f}" for s in scores], textposition="top center",
                    line=dict(color="#cba6f7", width=3),
                    marker=dict(size=12, color=[
                        "#f38ba8" if s >= 0.70 else "#f9e2af" if s >= 0.40 else "#a6e3a1"
                        for s in scores
                    ]),
                ))
                esc_fig.add_hline(y=0.70, line_dash="dash", line_color="rgba(243,139,168,0.4)",
                                  annotation_text="Block threshold")
                esc_fig.update_layout(
                    title=dict(text="Escalation Trajectory", font=dict(color="#cba6f7")),
                    paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#2d3748"),
                    yaxis=dict(range=[0, 1.1], gridcolor="#2d3748"),
                    height=250, margin=dict(t=40, b=20, l=40, r=20),
                    showlegend=False,
                )
                st.plotly_chart(esc_fig, use_container_width=True)


    with tab_benchmark:
        st.markdown("""
        ### 📈 Benchmark Analysis
        Run ThinkSym against the full curated adversarial test set to measure
        **detection rate**, **false positive rate**, and **per-category accuracy**.
        """)

        col1, col2 = st.columns(2)
        with col1:
            run_adv = st.checkbox("Include Adversarial Samples", value=True)
            n_adv   = st.slider("Adversarial samples", 1, len(adv_samples), min(5, len(adv_samples)))
        with col2:
            run_ben = st.checkbox("Include Benign Samples", value=True)
            n_ben   = st.slider("Benign samples", 1, len(ben_samples), min(5, len(ben_samples)))

        if st.button("▶ Run Benchmark", type="primary", use_container_width=True):
            bench_results = _run_benchmark(
                orchestrator, adv_samples[:n_adv] if run_adv else [],
                ben_samples[:n_ben] if run_ben else []
            )
            st.session_state.bench_results = bench_results

        if "bench_results" in st.session_state:
            _display_benchmark(st.session_state.bench_results)


def _run_analysis_with_progress(prompt, orchestrator, *toggles) -> AnalysisResult:

    enable_smooth, enable_neural, enable_scallop, enable_z3, enable_kg = toggles


    orchestrator.enable_smoothllm = enable_smooth
    orchestrator.enable_neural    = enable_neural
    orchestrator.enable_scallop   = enable_scallop
    orchestrator.enable_z3        = enable_z3
    orchestrator.enable_kg        = enable_kg

    layers = [
        ("L1", "SmoothLLM Perturbation",     enable_smooth),
        ("L2", "ModernBERT Neural Classify", enable_neural),
        ("L3", "Scallop Symbolic Reason",    enable_scallop),
        ("L4", "Z3 Formal Verification",     enable_z3),
        ("L5", "Knowledge Graph Match",      enable_kg),
    ]
    total_enabled = sum(1 for _, _, en in layers if en)

    progress_bar = st.progress(0, text="Starting ThinkSym pipeline…")
    status_placeholder = st.empty()

    completed = 0
    for layer_id, layer_name, enabled in layers:
        if enabled:
            status_placeholder.markdown(
                f"<div style='font-size:0.85rem;color:#89b4fa;'>⚙ Running {layer_id} · {layer_name}…</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.1)
        completed += 1 if enabled else 0
        progress_bar.progress(
            min(completed / max(total_enabled, 1), 0.90),
            text=f"Processing… {layer_id} ({layer_name})"
        )

    result = orchestrator.analyze(prompt)
    progress_bar.progress(1.0, text="✅ Analysis complete")
    status_placeholder.empty()
    time.sleep(0.4)
    progress_bar.empty()
    return result


def _render_full_result(result: AnalysisResult):
    st.markdown("---")


    _display_verdict(result)


    st.markdown("#### 🔬 Layer-by-Layer Analysis")
    layer_configs = [
        ("🔀 Layer 1 — SmoothLLM Perturbation",             _display_layer1),
        ("🧠 Layer 2 — ModernBERT Neural Classifier",        _display_layer2),
        ("⚙ Layer 3 — Scallop Neurosymbolic Reasoner",      _display_layer3),
        ("🔐 Layer 4 — Z3 Formal Policy Verifier",           _display_layer4),
        ("🕸 Layer 5 — Security Knowledge Graph",            _display_layer5),
    ]
    for title, fn in layer_configs:
        with st.expander(title, expanded=False):
            fn(result)


    with st.expander("📄 Export: Full Analysis Trace (JSON)", expanded=False):
        trace = {
            "prompt": result.prompt,
            "turn": result.turn_number,
            "verdict": {
                "decision":        result.verdict.decision if result.verdict else None,
                "risk_level":      result.verdict.risk_level if result.verdict else None,
                "composite_score": result.verdict.composite_score if result.verdict else None,
                "primary_reason":  result.verdict.primary_reason if result.verdict else None,
            },
            "layer_timings_ms": result.layer_timings_ms,
            "total_time_ms":    result.total_time_ms,
            "neural_scores":    {
                cat: ts.combined_score
                for cat, ts in (result.neural_result.threat_scores.items() if result.neural_result else {}.items())
            },
            "scallop_violations": result.scallop_result.violated_policies if result.scallop_result else [],
            "z3_verdict":         result.z3_result.formal_verdict if result.z3_result else None,
            "owasp_matches":      result.kg_result.owasp_hits if result.kg_result else [],
            "mitre_matches":      result.kg_result.mitre_hits if result.kg_result else [],
        }
        st.json(trace)


def _run_benchmark(orchestrator, adv_samples, ben_samples):
    all_samples = [
        {"prompt": s["prompt"], "label": "adversarial", "meta": s}
        for s in adv_samples
    ] + [
        {"prompt": s["prompt"], "label": "benign", "meta": s}
        for s in ben_samples
    ]

    results = []
    prog = st.progress(0, text="Running benchmark…")
    for i, sample in enumerate(all_samples):
        prog.progress((i + 1) / len(all_samples), text=f"Analyzing {i+1}/{len(all_samples)}…")
        res = orchestrator.analyze(sample["prompt"])
        predicted_adv = res.verdict and res.verdict.decision in ("BLOCK", "WARN")
        actual_adv    = sample["label"] == "adversarial"
        results.append({
            "prompt":       sample["prompt"][:60] + "…",
            "true_label":   sample["label"],
            "predicted":    res.verdict.decision if res.verdict else "—",
            "score":        res.verdict.composite_score if res.verdict else 0,
            "risk":         res.verdict.risk_level if res.verdict else "SAFE",
            "correct":      (predicted_adv == actual_adv),
            "top_threat":   res.neural_result.top_threat_category if res.neural_result else "—",
            "time_ms":      res.total_time_ms,
        })
    prog.empty()
    return results


def _display_benchmark(results):
    st.markdown("---")
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    adv_results = [r for r in results if r["true_label"] == "adversarial"]
    ben_results = [r for r in results if r["true_label"] == "benign"]

    tp = sum(1 for r in adv_results if r["predicted"] in ("BLOCK", "WARN"))
    tn = sum(1 for r in ben_results if r["predicted"] == "ALLOW")
    fp = sum(1 for r in ben_results if r["predicted"] in ("BLOCK", "WARN"))
    fn = sum(1 for r in adv_results if r["predicted"] == "ALLOW")

    precision =  tp / max(tp + fp, 1)
    recall    =  tp / max(tp + fn, 1)
    f1        =  2 * precision * recall / max(precision + recall, 1e-9)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="metric-tile"><div class="metric-value">{correct/total:.1%}</div><div class="metric-label">Accuracy</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-tile"><div class="metric-value" style="color:#a6e3a1;">{precision:.1%}</div><div class="metric-label">Precision</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-tile"><div class="metric-value" style="color:#89b4fa;">{recall:.1%}</div><div class="metric-label">Recall (TPR)</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-tile"><div class="metric-value" style="color:#cba6f7;">{f1:.1%}</div><div class="metric-label">F1 Score</div></div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="metric-tile"><div class="metric-value" style="color:{'#a6e3a1' if fp==0 else '#f9e2af'}">{fp/max(len(ben_results),1):.1%}</div><div class="metric-label">False Positive</div></div>""", unsafe_allow_html=True)


    cm_fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Benign", "Predicted Threat"],
        y=["Actual Benign", "Actual Adversarial"],
        colorscale=[[0, "#1a1f2e"], [0.5, "#89b4fa44"], [1, "#89b4fa"]],
        text=[[str(tn), str(fp)], [str(fn), str(tp)]],
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        showscale=False,
    ))
    cm_fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(color="#89b4fa")),
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        font=dict(color="#e2e8f0"),
        height=280, margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    avg_time = sum(r["time_ms"] for r in results) / len(results)
    st.caption(f"Avg inference time: {avg_time:.0f}ms/prompt | Total: {len(results)} samples")


if __name__ == "__main__":
    main()
