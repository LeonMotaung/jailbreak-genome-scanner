"""Professional Jailbreak Arena Dashboard - Real-time Evaluation Interface"""

import streamlit as st
import asyncio
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import pandas as pd
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.arena.jailbreak_arena import JailbreakArena
from src.defenders.llm_defender import LLMDefender
from src.attackers.prompt_generator import PromptGenerator
from src.models.jailbreak import AttackStrategy, EvaluationResult, SeverityLevel
from src.integrations.lambda_scraper import LambdaWebScraper
from src.intelligence.threat_intelligence import ThreatIntelligenceEngine
from src.scoring.jvi_calculator import JVICalculator
from src.visualization.vector3d_generator import Vector3DGenerator
from src.utils.logger import setup_logger, log

# Page config
st.set_page_config(
    page_title="JGS - Active Defense Infrastructure",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Jailbreak Genome Scanner - Automated Red-Teaming & Threat Radar System"
    }
)

# Professional Defense Dashboard - Defensive Acceleration Hackathon
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" integrity="sha512-iecdLmaskl7CVkqkXNQ/ZH/XLlvWZOJyj7Yy7tcenmpD1ypASozpmT/E0iPtmFIB46ZmdtAc9eNBvH0H/ZpiBw==" crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
    /* Professional fonts for enterprise UI */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Icon styling */
    .icon {
        display: inline-block;
        font-family: 'Font Awesome 6 Free';
        font-weight: 900;
        margin-right: 0.5rem;
        color: inherit;
        vertical-align: middle;
    }
    
    .icon-success {
        color: #22c55e;
    }
    
    .icon-error {
        color: #ef4444;
    }
    
    .icon-warning {
        color: #f59e0b;
    }
    
    .icon-info {
        color: #06b6d4;
    }
    
    /* Root variables for consistent theming - Professional Blue/Cyan */
    :root {
        --primary-glow: rgba(6, 182, 212, 0.4);
        --success-glow: rgba(34, 197, 94, 0.4);
        --danger-glow: rgba(239, 68, 68, 0.4);
        --glass-bg: rgba(15, 23, 42, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glass-shadow: rgba(0, 0, 0, 0.3);
    }
    
    /* Clean professional background */
    .stApp {
        background: #0f172a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content area - Glassmorphic */
    .main {
        position: relative;
        z-index: 1;
    }
    
    /* Premium Header with Glass Effect - Professional Blue */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #06b6d4 50%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        margin-bottom: 2rem;
        padding: 2rem 0;
        position: relative;
        animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.8), transparent);
        border-radius: 3px;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.6);
        animation: expandLine 1s ease-out 0.5s both;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes expandLine {
        from {
            width: 0;
            opacity: 0;
        }
        to {
            width: 200px;
            opacity: 1;
        }
    }
    
    /* Glassmorphic Stat Boxes */
    .stat-box {
        background: var(--glass-bg);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        color: #e0e7ff;
        box-shadow: 
            0 8px 32px var(--glass-shadow),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .stat-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stat-box:hover::before {
        left: 100%;
    }
    
    .stat-box:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(6, 182, 212, 0.5);
        box-shadow: 
            0 20px 60px rgba(6, 182, 212, 0.3),
            0 0 0 1px rgba(6, 182, 212, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        background: rgba(15, 23, 42, 0.85);
    }
    
    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Premium Live Battle Container - Glassmorphic */
    .live-battle {
        background: var(--glass-bg);
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
        animation: slideUpFade 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .live-battle::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent,
            rgba(6, 182, 212, 0.8),
            rgba(14, 165, 233, 0.8),
            rgba(6, 182, 212, 0.8),
            transparent);
        
    }
    
    
        50% {
            opacity: 1;
            box-shadow: 0 0 40px rgba(6, 182, 212, 0.6);
        }
    }
    
    /* Premium Log Styling - Glass Cards */
    .attack-log,
    .success-log,
    .fail-log {
        background: var(--glass-bg);
        backdrop-filter: blur(15px) saturate(180%);
        -webkit-backdrop-filter: blur(15px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.875rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .attack-log {
        border-left: 3px solid rgba(6, 182, 212, 0.6);
        color: #a5f3fc;
    }
    
    .success-log {
        background: rgba(34, 197, 94, 0.1);
        border-left: 3px solid rgba(34, 197, 94, 0.8);
        border-color: rgba(34, 197, 94, 0.2);
        color: #86efac;
        box-shadow: 
            0 4px 16px rgba(34, 197, 94, 0.2),
            0 0 0 1px rgba(34, 197, 94, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .fail-log {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid rgba(239, 68, 68, 0.8);
        border-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        box-shadow: 
            0 4px 16px rgba(239, 68, 68, 0.2),
            0 0 0 1px rgba(239, 68, 68, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .attack-log:hover,
    .success-log:hover,
    .fail-log:hover {
        transform: translateX(8px) scale(1.01);
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 30, 0.8) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        color: #e0e7ff;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.03);
        border-color: rgba(6, 182, 212, 0.5);
        box-shadow: 
            0 16px 40px rgba(6, 182, 212, 0.3),
            0 0 0 1px rgba(6, 182, 212, 0.2);
        background: rgba(15, 23, 42, 0.9);
    }
    
    /* Professional buttons */
    .stButton > button {
        background: #3b82f6 !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.625rem 1.25rem !important;
        transition: background-color 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #60a5fa !important;
    }
    
    /* Clean progress bar */
    .stProgress > div > div > div > div {
        background: #3b82f6 !important;
        border-radius: 4px !important;
    }
    
    /* Premium Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    p, label, .stText {
        color: #e0e7ff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Premium Input Fields - Glass */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: rgba(6, 182, 212, 0.6) !important;
        box-shadow: 
            0 0 0 3px rgba(6, 182, 212, 0.1),
            inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        outline: none !important;
    }
    
    /* Premium Expander */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: #e0e7ff !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(15, 23, 42, 0.9) !important;
        border-color: rgba(6, 182, 212, 0.4) !important;
    }
    
    /* Loading Animation */
    .loading-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 15, 30, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(6, 182, 212, 0.6), rgba(14, 165, 233, 0.6));
        border-radius: 10px;
        border: 2px solid rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, rgba(6, 182, 212, 0.8), rgba(14, 165, 233, 0.8));
    }
    
    /* Professional metric values */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
    
    /* Smooth transitions for all elements */
    * {
        transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Button icons via CSS */
    button[data-testid*="test_api"]::before {
        content: "\\f002";
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        margin-right: 0.5rem;
    }
    
    button[data-testid*="ssh_tunnel"]::before {
        content: "\\f0c1";
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        margin-right: 0.5rem;
    }
    
    button[data-testid*="test_current"]::before {
        content: "\\f002";
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        margin-right: 0.5rem;
    }
    
    /* Status indicators */
    .status-running::before {
        content: "\\f2f9";
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        margin-right: 0.5rem;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Tab icons via CSS - Add icons to tab labels */
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="false"] {
        font-family: 'Inter', sans-serif !important;
        color: #e0e7ff !important;
        position: relative;
    }
    
    /* Trends tab icon */
    button[data-baseweb="tab"]:has-text("Trends")::before,
    div[data-testid*="stTabs"] button:contains("Trends")::before {
        content: "\\f201";
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        margin-right: 0.5rem;
        color: #06b6d4;
    }
    
    /* Strategies tab icon */
    button[data-baseweb="tab"]:has-text("Strategies")::before {
        content: "\\f140";
        font-family: "Font Awesome 6 Free";
        font-weight: 900;
        margin-right: 0.5rem;
        color: #06b6d4;
    }
    
    /* Alternative: Use CSS attribute selectors for tabs */
    div[data-testid="stTabs"] button {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Style tabs with glassmorphic effect */
    div[data-testid="stTabs"] > div {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }
    
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: rgba(6, 182, 212, 0.2) !important;
        border-bottom: 2px solid #06b6d4 !important;
        color: #06b6d4 !important;
    }
    
    div[data-testid="stTabs"] button[aria-selected="false"] {
        color: #94a3b8 !important;
    }
    
    div[data-testid="stTabs"] button:hover {
        background: rgba(6, 182, 212, 0.1) !important;
        color: #06b6d4 !important;
    }
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    """Run async function in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
    
    return loop.run_until_complete(coro)


def create_jvi_gauge(value):
    """Create professional JVI score gauge visualization with enhanced styling."""
    # Color scale: green (safe) to red (vulnerable) - only for visualization
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    thresholds = [20, 50, 80, 100]
    
    # Determine color based on value
    color = colors[0]
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            color = colors[i] if i < len(colors) else colors[-1]
            break
    
    # Get category label
    if value < 20:
        category = "Very Low Risk"
    elif value < 40:
        category = "Low Risk"
    elif value < 60:
        category = "Moderate Risk"
    elif value < 80:
        category = "High Risk"
    else:
        category = "Critical Risk"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"JVI Score<br><span style='font-size:0.8em;color:{color}'>{category}</span>", 
               'font': {'size': 22, 'color': '#ffffff'}},
        number={'font': {'size': 40, 'color': color}, 'suffix': '/100'},
        delta={'reference': 50, 'position': "top", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#e0e0e0', 'tickwidth': 2, 'tickmode': 'linear', 'tick0': 0, 'dtick': 20},
            'bar': {'color': color, 'thickness': 0.15},
            'bgcolor': '#1a1a1a',
            'borderwidth': 3,
            'bordercolor': '#555555',
            'steps': [
                {'range': [0, 20], 'color': '#1a3a1a'},
                {'range': [20, 50], 'color': '#3a3a1a'},
                {'range': [50, 80], 'color': '#3a1a1a'},
                {'range': [80, 100], 'color': '#3a0a0a'}
            ],
            'threshold': {
                'line': {'color': '#dc3545', 'width': 4},
                'thickness': 0.8,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(26, 26, 38, 0.5)',
        plot_bgcolor='rgba(26, 26, 38, 0.3)',
        font={'color': '#e0e7ff', 'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', 'size': 12},
        showlegend=False
    )
    return fig


def create_trend_chart(evaluation_history: List[Dict]) -> Optional[go.Figure]:
    """Create trend chart showing exploit rate over time."""
    if not evaluation_history:
        return None
    
    # Process history to get exploit rates over rounds
    rounds_data = {}
    for i, eval_result in enumerate(evaluation_history):
        round_num = i // 10 + 1  # Approximate round number
        if round_num not in rounds_data:
            rounds_data[round_num] = {'total': 0, 'exploits': 0}
        
        rounds_data[round_num]['total'] += 1
        is_jailbroken = eval_result.get('is_jailbroken', False) if isinstance(eval_result, dict) else eval_result.is_jailbroken
        if is_jailbroken:
            rounds_data[round_num]['exploits'] += 1
    
    rounds = sorted(rounds_data.keys())
    exploit_rates = [(rounds_data[r]['exploits'] / rounds_data[r]['total'] * 100) for r in rounds]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds,
        y=exploit_rates,
        mode='lines+markers',
        name='Exploit Rate',
        line=dict(color='#dc3545', width=3),
        marker=dict(size=8, color='#dc3545')
    ))
    
    fig.update_layout(
        title={'text': 'Exploit Rate Trend', 'font': {'size': 18, 'color': '#ffffff'}},
        xaxis={'title': 'Round', 'tickfont': {'color': '#e0e0e0'}, 'gridcolor': '#333333'},
        yaxis={'title': 'Exploit Rate (%)', 'tickfont': {'color': '#e0e0e0'}, 'gridcolor': '#333333'},
        height=300,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': '#e0e0e0'},
        hovermode='x unified'
    )
    
    return fig


def create_strategy_distribution_chart(evaluation_history: List[Dict]) -> Optional[go.Figure]:
    """Create pie chart showing attack strategy distribution."""
    if not evaluation_history:
        return None
    
    strategy_counts = {}
    for eval_result in evaluation_history:
        if isinstance(eval_result, dict):
            strategy = eval_result.get('attack_strategy', {})
            if isinstance(strategy, dict):
                strategy_name = strategy.get('value', 'unknown')
            else:
                strategy_name = str(strategy)
            is_jailbroken = eval_result.get('is_jailbroken', False)
        else:
            strategy_name = str(eval_result.attack_strategy.value) if hasattr(eval_result.attack_strategy, 'value') else str(eval_result.attack_strategy)
            is_jailbroken = eval_result.is_jailbroken
        
        if is_jailbroken:
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    
    if not strategy_counts:
        return None
    
    strategies = list(strategy_counts.keys())
    counts = list(strategy_counts.values())
    
    colors = px.colors.sequential.Reds_r[:len(strategies)]
    
    fig = go.Figure(data=[go.Pie(
        labels=strategies,
        values=counts,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont={'color': '#e0e0e0', 'size': 12}
    )])
    
    fig.update_layout(
        title={'text': 'Successful Attacks by Strategy', 'font': {'size': 18, 'color': '#ffffff'}},
        height=350,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': '#e0e0e0'},
        showlegend=True,
        legend={'font': {'color': '#e0e0e0'}}
    )
    
    return fig


def create_severity_chart(evaluation_history: List[Dict]) -> Optional[go.Figure]:
    """Create bar chart showing severity distribution."""
    if not evaluation_history:
        return None
    
    severity_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for eval_result in evaluation_history:
        is_jailbroken = eval_result.get('is_jailbroken', False) if isinstance(eval_result, dict) else eval_result.is_jailbroken
        if is_jailbroken:
            if isinstance(eval_result, dict):
                severity = eval_result.get('severity', {})
                severity_val = severity.get('value', 0) if isinstance(severity, dict) else severity
            else:
                severity_val = eval_result.severity.value if hasattr(eval_result.severity, 'value') else eval_result.severity
            
            if 1 <= severity_val <= 5:
                severity_counts[severity_val] = severity_counts.get(severity_val, 0) + 1
    
    severities = list(severity_counts.keys())
    counts = list(severity_counts.values())
    
    if sum(counts) == 0:
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=severities,
        y=counts,
        marker_color='#dc3545',
        text=counts,
        textposition='outside',
        textfont={'color': '#e0e0e0', 'size': 12}
    )])
    
    fig.update_layout(
        title={'text': 'Severity Distribution', 'font': {'size': 18, 'color': '#ffffff'}},
        xaxis={'title': 'Severity Level', 'tickfont': {'color': '#e0e0e0'}, 'gridcolor': '#333333'},
        yaxis={'title': 'Count', 'tickfont': {'color': '#e0e0e0'}, 'gridcolor': '#333333'},
        height=300,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': '#e0e0e0'}
    )
    
    return fig


def create_leaderboard_chart(attackers):
    """Create enhanced professional leaderboard chart with multiple metrics."""
    if not attackers:
        return None
    
    # Handle both dict and object access
    if isinstance(attackers[0], dict):
        attacker_names = [a.get('name', 'Unknown') for a in attackers[:10]]
        points = [a.get('total_points', 0) for a in attackers[:10]]
        success_rates = [a.get('success_rate', 0) * 100 if isinstance(a.get('success_rate'), (int, float)) else 0 for a in attackers[:10]]
    else:
        attacker_names = [a.name for a in attackers[:10]]
        points = [a.total_points for a in attackers[:10]]
        success_rates = [a.success_rate * 100 for a in attackers[:10]]
    
    fig = go.Figure()
    
    # Add success rate as secondary axis
    fig.add_trace(go.Bar(
        x=attacker_names,
        y=points,
        name="Total Points",
        marker_color='#28a745',
        text=[f"{p:.1f}" for p in points],
        textposition='outside',
        textfont={'color': '#e0e0e0', 'size': 11},
        yaxis='y',
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=attacker_names,
        y=success_rates,
        name="Success Rate (%)",
        mode='lines+markers',
        marker=dict(color='#ffc107', size=8),
        line=dict(color='#ffc107', width=2),
        yaxis='y2',
        text=[f"{sr:.1f}%" for sr in success_rates],
        textposition='top center',
        textfont={'color': '#ffc107', 'size': 10}
    ))
    
    fig.update_layout(
        title={'text': "Attacker Leaderboard", 'font': {'size': 20, 'color': '#ffffff'}},
        xaxis={'title': {'text': 'Attackers', 'font': {'color': '#e0e0e0', 'size': 14}}, 
               'tickfont': {'color': '#e0e0e0'}, 'gridcolor': '#333333'},
        yaxis={'title': {'text': 'Total Points', 'font': {'color': '#28a745', 'size': 14}}, 
               'tickfont': {'color': '#e0e0e0'}, 'gridcolor': '#333333'},
        yaxis2={'title': {'text': 'Success Rate (%)', 'font': {'color': '#ffc107', 'size': 14}},
                'overlaying': 'y', 'side': 'right', 'tickfont': {'color': '#ffc107'}},
        height=450,
        showlegend=True,
        legend={'x': 0.7, 'y': 0.95, 'font': {'color': '#e0e0e0'}, 'bgcolor': '#1a1a1a'},
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': '#e0e0e0'},
        hovermode='x unified',
        margin=dict(l=60, r=80, t=60, b=60)
    )
    
    return fig


def make_json_serializable(obj):
    """Convert objects to JSON-serializable types (sets to lists, etc.)."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (like Enum values)
        try:
            return obj.value if hasattr(obj, 'value') else str(obj)
        except:
            return str(obj)
    else:
        return obj


async def gather_recent_attacks(scraper, instance_id=None):
    """Use Lambda scraper to gather recent attack patterns for intelligence gathering."""
    if not scraper:
        return None
    
    try:
        events = await scraper.scrape_recent_events(days_back=7, max_results=15)
        
        if events and len(events) > 0:
            # Format events into readable text with better structure
            formatted = []
            for event in events[:10]:  # Limit to 10 most relevant
                if event and hasattr(event, 'title'):
                    # Clean title
                    title = event.title.strip()
                    source = getattr(event, 'source', 'Unknown')
                    category = getattr(event, 'category', 'jailbreak')
                    url = getattr(event, 'url', '')
                    content = getattr(event, 'content', '')[:150] if hasattr(event, 'content') else ''
                    
                    formatted.append(
                        f"**{title}**\n"
                        f"Source: {source}\n"
                        f"Category: {category}\n"
                        f"{'Content: ' + content + '...' if content else ''}\n"
                        f"URL: {url}\n"
                    )
            if formatted:
                return "\n\n".join(formatted)
        return None
    except Exception as e:
        import traceback
        log.error(f"Error gathering recent attacks: {e}\n{traceback.format_exc()}")
        return None


def main():
    """Main dashboard application."""
    
    # Professional header with icon
    st.markdown("""
    <div class="main-header">
        <i class="fas fa-shield-alt" style="font-size: 0.8em; vertical-align: middle; margin-right: 0.5rem;"></i>
        JAILBREAK GENOME SCANNER
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #999999; margin-top: -1rem;">Active Defense Infrastructure - Automated Red-Teaming & Threat Radar</p>', unsafe_allow_html=True)
    
    # Load deployment config if available (for pre-filling)
    import json
    from pathlib import Path
    deployment_config_path = Path("data/lambda_deployments.json")
    default_model = "microsoft/phi-2"
    default_instance = ""
    default_endpoint = ""
    
    # Load all active instances (for scraper)
    active_instances = []
    # Load deployed models (instances with actual models running)
    deployed_models = []
    
    if deployment_config_path.exists():
        try:
            with open(deployment_config_path, 'r') as f:
                config = json.load(f)
                deployed = config.get("deployed_models", {})
                # Get all active instances (for scraper - includes all active instances)
                active_instances = [
                    {
                        "key": key,
                        "name": key.replace("-", " ").title(),
                        "model_name": info.get("model_name", ""),
                        "instance_id": info.get("instance_id", ""),
                        "instance_ip": info.get("instance_ip", ""),
                        "instance_type": info.get("instance_type", ""),
                        "api_endpoint": info.get("api_endpoint", ""),
                        "api_endpoint_local": info.get("api_endpoint_local", ""),
                        "vllm_running": info.get("vllm_running", False),
                        "status": info.get("status", "unknown")
                    }
                    for key, info in deployed.items()
                    if info.get("status") == "active"
                ]
                
                # Get deployed models (instances with vLLM running and model_name)
                deployed_models = [
                    inst for inst in active_instances
                    if inst.get("vllm_running", False) and inst.get("model_name", "")
                ]
                # For model selection, use deployed_models (instances with actual models)
                if deployed_models:
                    # Prioritize H100 instances with models (the big one!)
                    h100_models = [inst for inst in deployed_models if "h100" in inst.get("instance_type", "").lower() or "h100" in inst.get("key", "").lower()]
                    
                    if h100_models:
                        # Use H100 model as default (the big one!)
                        first = h100_models[0]
                        log.info(f"Auto-selected H100 model instance: {first.get('instance_id')}")
                    else:
                        # Use first deployed model
                        first = deployed_models[0]
                    
                    default_model = first.get("model_name", "microsoft/phi-2")
                    default_instance = first.get("instance_id", "")
                    # Prefer local endpoint if available (SSH tunnel)
                    default_endpoint = first.get("api_endpoint_local") or first.get("api_endpoint", "")
                    if not default_endpoint and first.get("instance_ip"):
                        default_endpoint = f"http://{first.get('instance_ip')}:8000/v1/chat/completions"
        except Exception as e:
            log.debug(f"Error loading deployment config: {e}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Defender Setup")
        defender_type = st.selectbox(
            "Defender Type",
            ["Mock (Demo)", "OpenAI", "Anthropic", "Lambda Cloud"],
            help="Choose your defender model type"
        )
        
        if defender_type == "OpenAI":
            st.info("💡 **Note**: For Lambda Cloud credits, use 'Lambda Cloud' option with open-source models instead")
            model_name = st.text_input("Model Name", "gpt-4")
            api_key = st.text_input("OpenAI API Key", type="password")
            instance_id = None
            api_endpoint = None
        elif defender_type == "Anthropic":
            st.info("💡 **Note**: For Lambda Cloud credits, use 'Lambda Cloud' option with open-source models instead")
            model_name = st.text_input("Model Name", "claude-3-opus")
            api_key = st.text_input("Anthropic API Key", type="password")
            instance_id = None
            api_endpoint = None
        elif defender_type == "Lambda Cloud":
            # Show model selector - only deployed models (instances with vLLM running)
            if deployed_models:
                # Prioritize H100 instances with models
                h100_instances = [inst for inst in deployed_models if "h100" in inst.get("instance_type", "").lower() or "h100" in inst.get("key", "").lower()]
                other_instances = [inst for inst in deployed_models if inst not in h100_instances]
                
                # Build options with H100 first, marked as "THE BIG ONE"
                instance_options = {}
                
                # Add H100 instances first with special label
                for inst in h100_instances:
                    label = f"🚀 {inst['name']} - H100 ({inst['model_name'].split('/')[-1]}) - THE BIG ONE!"
                    instance_options[label] = inst
                
                # Add other instances
                for inst in other_instances:
                    label = f"{inst['name']} ({inst['model_name'].split('/')[-1]})"
                    instance_options[label] = inst
                
                instance_options["Custom Instance"] = None
                
                # Default to H100 if available
                default_index = 0 if h100_instances else 0
                
                selected_instance_label = st.selectbox(
                    "Select Instance",
                    list(instance_options.keys()),
                    index=default_index,
                    help="🚀 H100 instances are THE BIG ONE - fastest GPU available! Choose from deployed instances or configure custom"
                )
                
                selected_instance = instance_options[selected_instance_label]
                
                if selected_instance:
                    # Auto-fill from selected instance
                    instance_type_display = ""
                    if "h100" in selected_instance.get("instance_type", "").lower() or "h100" in selected_instance.get("key", "").lower():
                        instance_type_display = " - H100 THE BIG ONE! 🚀"
                    
                    model_name = st.text_input("Model Name", value=selected_instance["model_name"], disabled=True)
                    instance_id = st.text_input(
                        f"Lambda Instance ID{instance_type_display}", 
                        value=selected_instance["instance_id"], 
                        disabled=True
                    )
                    
                    # Show status
                    if selected_instance.get("vllm_running"):
                        st.success(f"✅ vLLM is running on {selected_instance['instance_ip']}")
                    else:
                        st.warning(f"⚠️ vLLM status unknown for {selected_instance['instance_ip']}")
                    
                    # Show endpoint options
                    if selected_instance.get("api_endpoint_local"):
                        st.info(f"💡 Use SSH tunnel endpoint: `{selected_instance['api_endpoint_local']}`")
                        api_endpoint = st.text_input(
                            "API Endpoint",
                            value=selected_instance["api_endpoint_local"],
                            help="Using SSH tunnel endpoint (recommended if port 8000 is blocked)"
                        )
                    else:
                        api_endpoint = st.text_input(
                            "API Endpoint",
                            value=selected_instance.get("api_endpoint", ""),
                            help="vLLM API endpoint. Use SSH tunnel if port is blocked."
                        )
                else:
                    # Custom instance
                    model_name = st.text_input("Model Name", default_model)
                    instance_id = st.text_input("Lambda Instance ID", default_instance)
                    api_endpoint = st.text_input("API Endpoint", default_endpoint)
            else:
                # No active instances, use manual input
                model_name = st.text_input("Model Name", default_model)
                instance_id = st.text_input("Lambda Instance ID", default_instance)
                api_endpoint = st.text_input("API Endpoint", default_endpoint)
            
            # Get instance IP for display
            instance_ip = None
            if active_instances and selected_instance:
                instance_ip = selected_instance.get("instance_ip")
            elif instance_id:
                # Try to get IP from Lambda API if not in config
                try:
                    from src.integrations.lambda_cloud import LambdaCloudClient
                    lambda_client = LambdaCloudClient()
                    instance = run_async(lambda_client.get_instance_status(instance_id))
                    if instance and instance.get("ip"):
                        instance_ip = instance.get("ip")
                except Exception as e:
                    log.debug(f"Could not auto-detect IP: {e}")
            
            if instance_ip:
                st.markdown(f'<div style="padding: 0.75rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid rgba(6, 182, 212, 0.6); border-radius: 6px; margin: 0.5rem 0;"><i class="fas fa-map-marker-alt" style="margin-right: 0.5rem; color: #06b6d4;"></i> Instance IP: {instance_ip}</div>', unsafe_allow_html=True)
                
                # Check connectivity status
                if api_endpoint and not api_endpoint.startswith("localhost"):
                    # Only check external endpoints
                    from scripts.ssh_tunnel_helper import check_port_connectivity
                    from urllib.parse import urlparse
                    try:
                        parsed = urlparse(api_endpoint)
                        host = parsed.hostname
                        port = parsed.port or 8000
                        if host and host != "localhost":
                            port_open = check_port_connectivity(host, port, timeout=3.0)
                            if not port_open:
                                st.markdown(f'<div style="padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid rgba(239, 68, 68, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fca5a5;"><i class="fas fa-times-circle" style="margin-right: 0.5rem; color: #ef4444;"></i> <strong>Port {port} is blocked!</strong> Cannot connect externally.</div>', unsafe_allow_html=True)
                                with st.expander('<i class="fas fa-link" style="margin-right: 0.5rem;"></i> Quick Fix: Use SSH Tunnel', expanded=True):
                                    st.markdown("**Port is blocked by firewall. Use SSH tunnel:**")
                                    st.code(f"python scripts/ssh_tunnel_helper.py --ip {instance_ip} --key moses.pem", language="bash")
                                    st.markdown("Then change endpoint to: `http://localhost:8000/v1/chat/completions`")
                                    st.markdown("**Or configure security group:** Run `python scripts/configure_security_group.py`")
                    except Exception:
                        pass  # Skip check if it fails
                
                if not api_endpoint or "<ip>" in api_endpoint:
                    st.markdown('<div style="padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border-left: 3px solid rgba(245, 158, 11, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fcd34d;"><i class="fas fa-exclamation-triangle" style="margin-right: 0.5rem; color: #f59e0b;"></i> <strong>vLLM not set up yet!</strong> The API endpoint needs to be configured.</div>', unsafe_allow_html=True)
                    with st.expander("Set up vLLM on Lambda instance", expanded=True):
                        st.markdown("**Quick Setup:**")
                        st.code(f"""
# Run this script to set up vLLM:
python scripts/setup_vllm_on_lambda.py --ip {instance_ip} --key moses.pem --model {model_name}

# Or manually SSH and run:
ssh -i moses.pem ubuntu@{instance_ip}
pip3 install vllm
python3 -m vllm.entrypoints.openai.api_server \\
    --model {model_name} \\
    --port 8000 \\
    --host 0.0.0.0
                        """, language="bash")
                        st.markdown("**After setup, use this endpoint:**")
                        st.code(f"http://{instance_ip}:8000/v1/chat/completions", language="text")
                else:
                    st.markdown(f'<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> Endpoint configured: {api_endpoint}</div>', unsafe_allow_html=True)
            api_key = None
        else:
            model_name = "demo-model-v1"
            api_key = None
            instance_id = None
            api_endpoint = None
        
        # Attack configuration
        st.subheader("Attack Configuration")
        num_attackers = st.slider("Number of Attackers", 3, 10, 5)
        num_rounds = st.slider("Number of Rounds", 1, 50, 10)
        
        # Difficulty selection
        use_database = st.checkbox("Use Structured Prompt Database", value=True,
                                   help="Use curated prompts with difficulty levels")
        if use_database:
            difficulty_category = st.selectbox(
                "Difficulty Level",
                ["All", "Low (L1-L5)", "Medium (M1-M5)", "High (H1-H10)", "Custom Range"],
                help="Select difficulty range for prompts"
            )
            
            if difficulty_category == "Custom Range":
                col1, col2 = st.columns(2)
                with col1:
                    min_difficulty = st.selectbox("Min Difficulty", ["L1", "L2", "L3", "L4", "L5", "M1", "M2", "M3", "M4", "M5", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"], index=0)
                with col2:
                    max_difficulty = st.selectbox("Max Difficulty", ["L1", "L2", "L3", "L4", "L5", "M1", "M2", "M3", "M4", "M5", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"], index=19)
                difficulty_range = (min_difficulty, max_difficulty)
            elif difficulty_category == "Low (L1-L5)":
                difficulty_range = ("L1", "L5")
            elif difficulty_category == "Medium (M1-M5)":
                difficulty_range = ("M1", "M5")
            elif difficulty_category == "High (H1-H10)":
                difficulty_range = ("H1", "H10")
            else:
                difficulty_range = None
        else:
            difficulty_range = None
        
        # Attacker Setup (LLM-based)
        st.subheader("Attacker Setup (Optional)")
        use_llm_attacker = st.checkbox(
            "Use LLM-based Attacker",
            value=False,
            help="Use an LLM model to generate attack prompts instead of rule-based generation"
        )
        
        attacker_model_name = None
        attacker_instance_id = None
        attacker_api_endpoint = None
        
        if use_llm_attacker:
            attacker_type = st.selectbox(
                "Attacker Model Type",
                ["Lambda Cloud"],
                key="attacker_type",
                help="Attacker model type (currently only Lambda Cloud supported)"
            )
            
            if attacker_type == "Lambda Cloud" and deployed_models:
                # Show model selector for attacker - only deployed models
                attacker_h100_instances = [inst for inst in deployed_models if "h100" in inst.get("instance_type", "").lower() or "h100" in inst.get("key", "").lower()]
                attacker_other_instances = [inst for inst in deployed_models if inst not in attacker_h100_instances]
                
                attacker_instance_options = {}
                for inst in attacker_h100_instances:
                    label = f"🚀 {inst['name']} - H100 ({inst['model_name'].split('/')[-1]}) - THE BIG ONE!"
                    attacker_instance_options[label] = inst
                for inst in attacker_other_instances:
                    label = f"{inst['name']} ({inst['model_name'].split('/')[-1]})"
                    attacker_instance_options[label] = inst
                attacker_instance_options["Custom Instance"] = None
                
                attacker_selected_label = st.selectbox(
                    "Select Attacker Instance",
                    list(attacker_instance_options.keys()),
                    key="attacker_instance",
                    help="Select Lambda instance for attacker model"
                )
                
                attacker_selected_instance = attacker_instance_options[attacker_selected_label]
                
                if attacker_selected_instance:
                    attacker_model_name = st.text_input(
                        "Attacker Model Name",
                        value=attacker_selected_instance["model_name"],
                        key="attacker_model_name",
                        disabled=True
                    )
                    attacker_instance_id = st.text_input(
                        "Attacker Instance ID",
                        value=attacker_selected_instance["instance_id"],
                        key="attacker_instance_id",
                        disabled=True
                    )
                    attacker_api_endpoint = st.text_input(
                        "Attacker API Endpoint",
                        value=attacker_selected_instance.get("api_endpoint_local") or attacker_selected_instance.get("api_endpoint", ""),
                        key="attacker_api_endpoint"
                    )
                else:
                    attacker_model_name = st.text_input("Attacker Model Name", key="attacker_model_name_custom")
                    attacker_instance_id = st.text_input("Attacker Instance ID", key="attacker_instance_id_custom")
                    attacker_api_endpoint = st.text_input("Attacker API Endpoint", key="attacker_api_endpoint_custom")
        
        # Evaluator Setup (LLM-based)
        st.subheader("Evaluator Setup (Optional)")
        use_llm_evaluator = st.checkbox(
            "Use LLM-based Evaluator",
            value=False,
            help="Use an LLM model to evaluate responses instead of rule-based classification"
        )
        
        evaluator_model_name = None
        evaluator_instance_id = None
        evaluator_api_endpoint = None
        
        if use_llm_evaluator:
            evaluator_type = st.selectbox(
                "Evaluator Model Type",
                ["Lambda Cloud"],
                key="evaluator_type",
                help="Evaluator model type (currently only Lambda Cloud supported)"
            )
            
            if evaluator_type == "Lambda Cloud" and deployed_models:
                # Show model selector for evaluator - only deployed models
                evaluator_h100_instances = [inst for inst in deployed_models if "h100" in inst.get("instance_type", "").lower() or "h100" in inst.get("key", "").lower()]
                evaluator_other_instances = [inst for inst in deployed_models if inst not in evaluator_h100_instances]
                
                evaluator_instance_options = {}
                for inst in evaluator_h100_instances:
                    label = f"🚀 {inst['name']} - H100 ({inst['model_name'].split('/')[-1]}) - THE BIG ONE!"
                    evaluator_instance_options[label] = inst
                for inst in evaluator_other_instances:
                    label = f"{inst['name']} ({inst['model_name'].split('/')[-1]})"
                    evaluator_instance_options[label] = inst
                evaluator_instance_options["Custom Instance"] = None
                
                evaluator_selected_label = st.selectbox(
                    "Select Evaluator Instance",
                    list(evaluator_instance_options.keys()),
                    key="evaluator_instance",
                    help="Select Lambda instance for evaluator model"
                )
                
                evaluator_selected_instance = evaluator_instance_options[evaluator_selected_label]
                
                if evaluator_selected_instance:
                    evaluator_model_name = st.text_input(
                        "Evaluator Model Name",
                        value=evaluator_selected_instance["model_name"],
                        key="evaluator_model_name",
                        disabled=True
                    )
                    evaluator_instance_id = st.text_input(
                        "Evaluator Instance ID",
                        value=evaluator_selected_instance["instance_id"],
                        key="evaluator_instance_id",
                        disabled=True
                    )
                    evaluator_api_endpoint = st.text_input(
                        "Evaluator API Endpoint",
                        value=evaluator_selected_instance.get("api_endpoint_local") or evaluator_selected_instance.get("api_endpoint", ""),
                        key="evaluator_api_endpoint"
                    )
                else:
                    evaluator_model_name = st.text_input("Evaluator Model Name", key="evaluator_model_name_custom")
                    evaluator_instance_id = st.text_input("Evaluator Instance ID", key="evaluator_instance_id_custom")
                    evaluator_api_endpoint = st.text_input("Evaluator API Endpoint", key="evaluator_api_endpoint_custom")
        
        # Lambda Scraper config - shows ALL active instances (not just models)
        st.subheader("Intelligence Gathering")
        use_scraper = st.checkbox("Gather Recent Attack Patterns", value=True, 
                                   help="Scrape web sources (GitHub, forums) to identify emerging jailbreak techniques")
        if use_scraper:
            # Show instance selector for scraper (all active instances, including H100 without models)
            if active_instances:
                # Prioritize H100 instances for scraping (even if no model)
                scraper_h100_instances = [inst for inst in active_instances if "h100" in inst.get("instance_type", "").lower() or "h100" in inst.get("key", "").lower()]
                scraper_other_instances = [inst for inst in active_instances if inst not in scraper_h100_instances]
                
                scraper_instance_options = {}
                for inst in scraper_h100_instances:
                    # For scraper, show instance info (not model info)
                    label = f"🚀 {inst['name']} - H100 Instance - THE BIG ONE! 🚀"
                    scraper_instance_options[label] = inst
                for inst in scraper_other_instances:
                    label = f"{inst['name']} Instance"
                    scraper_instance_options[label] = inst
                scraper_instance_options["Custom Instance"] = None
                scraper_instance_options["None (Local Scraping)"] = {"instance_id": None}
                
                scraper_selected_label = st.selectbox(
                    "Select Scraper Instance (Optional)",
                    list(scraper_instance_options.keys()),
                    key="scraper_instance",
                    help="Select Lambda instance for web scraping (H100 recommended for better performance). This is for scraping, not a model!"
                )
                
                scraper_selected_instance = scraper_instance_options[scraper_selected_label]
                
                if scraper_selected_instance and scraper_selected_instance.get("instance_id"):
                    scraper_instance_id = scraper_selected_instance["instance_id"]
                    st.info(f"📡 Using instance {scraper_selected_instance.get('instance_id', '')} for web scraping (not a model)")
                elif scraper_selected_instance and scraper_selected_instance.get("instance_id") is None:
                    scraper_instance_id = None
                    st.info("📡 Using local scraping (no Lambda instance)")
                else:
                    scraper_instance_id = st.text_input(
                        "Lambda Instance ID (Custom)",
                        key="scraper_instance_custom",
                        help="Enter instance ID for scraping (this is an instance, not a model!)"
                    )
            else:
                scraper_instance_id = st.text_input(
                    "Lambda Instance ID (Optional)", 
                    value="",
                    key="scraper_instance_manual",
                    help="Optional: Use a Lambda instance for web scraping (especially H100!). This is for scraping, not a model!"
                )
        else:
            scraper_instance_id = None
        
        # Start battle button
        start_battle = st.button("START EVALUATION", type="primary", use_container_width=True)
    
    # Initialize session state
    if 'arena' not in st.session_state:
        st.session_state.arena = None
        st.session_state.battle_running = False
        st.session_state.results = None
        st.session_state.logs = []
        st.session_state.scraper_data = None
        st.session_state.start_time = None
        st.session_state.end_time = None
        st.session_state.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
    
    # Main content area
    if start_battle or st.session_state.battle_running:
        # Initialize arena with threat intelligence enabled
        if not st.session_state.arena:
            # Initialize LLM attacker if configured
            llm_attacker = None
            if use_llm_attacker and attacker_model_name and attacker_instance_id:
                try:
                    from src.attackers.llm_attacker import LLMAttacker
                    llm_attacker = LLMAttacker(
                        model_name=attacker_model_name,
                        model_type="local",
                        use_lambda=True,
                        lambda_instance_id=attacker_instance_id,
                        lambda_api_endpoint=attacker_api_endpoint
                    )
                    log.info(f"Initialized LLM attacker: {attacker_model_name} on instance {attacker_instance_id}")
                except Exception as e:
                    log.error(f"Error initializing LLM attacker: {e}")
                    st.warning(f"Failed to initialize LLM attacker: {e}")
            
            # Initialize LLM evaluator if configured
            llm_evaluator = None
            if use_llm_evaluator and evaluator_model_name and evaluator_instance_id:
                try:
                    from src.referee.llm_evaluator import LLMEvaluator
                    llm_evaluator = LLMEvaluator(
                        model_name=evaluator_model_name,
                        model_type="local",
                        use_lambda=True,
                        lambda_instance_id=evaluator_instance_id,
                        lambda_api_endpoint=evaluator_api_endpoint
                    )
                    log.info(f"Initialized LLM evaluator: {evaluator_model_name} on instance {evaluator_instance_id}")
                except Exception as e:
                    log.error(f"Error initializing LLM evaluator: {e}")
                    st.warning(f"Failed to initialize LLM evaluator: {e}")
            
            # Create arena with optional LLM components
            st.session_state.arena = JailbreakArena(
                use_pattern_database=True,
                llm_attacker=llm_attacker,
                llm_evaluator=llm_evaluator
            )
        
        # Gather recent data from Lambda scraper if enabled and integrate with Threat Intelligence
        if use_scraper:
            import threading
            
            # Show initial status in main thread
            scraper_status = st.empty()
            scraper_status.info("🕵️ Gathering recent attack patterns and extracting intelligence...")
            
            def run_intelligence_gathering():
                """Run intelligence gathering in background thread."""
                try:
                    log.info("Starting intelligence gathering with threat intelligence engine...")
                    
                    # Create scraper
                    scraper = LambdaWebScraper(instance_id=scraper_instance_id if scraper_instance_id else None)
                    log.info(f"Created scraper with instance_id: {scraper_instance_id}")
                    
                    # Initialize Threat Intelligence Engine with the arena's pattern database
                    if st.session_state.arena and hasattr(st.session_state.arena, 'pattern_database') and st.session_state.arena.pattern_database:
                        pattern_db = st.session_state.arena.pattern_database
                        log.info("Using existing arena pattern database")
                    else:
                        # Create new pattern database if arena doesn't have one
                        from src.intelligence.pattern_database import ExploitPatternDatabase
                        pattern_db = ExploitPatternDatabase()
                        log.info("Created new pattern database for intelligence")
                        # Add to arena if it exists
                        if st.session_state.arena:
                            st.session_state.arena.pattern_database = pattern_db
                    
                    # Initialize threat intelligence engine
                    threat_engine = ThreatIntelligenceEngine(
                        pattern_database=pattern_db,
                        scraper=scraper
                    )
                    if st.session_state.arena:
                        st.session_state.arena.threat_intelligence = threat_engine
                    
                    log.info("Threat intelligence engine initialized, starting scraper...")
                    
                    # Update intelligence from scraper - THIS ACTUALLY EXTRACTS PATTERNS!
                    update_result = run_async(threat_engine.update_from_scraper(days_back=7, max_results=50))
                    log.info(f"Scraper update result: {update_result}")
                    
                    # Also integrate prompts into prompt database (not async, call directly)
                    try:
                        integrate_result = threat_engine.integrate_prompts_to_database()
                        log.info(f"Prompt integration result: {integrate_result}")
                    except Exception as e:
                        import traceback
                        log.error(f"Error integrating prompts: {e}\n{traceback.format_exc()}")
                        integrate_result = {"prompts_added": 0, "error": str(e)}
                    
                    # Get recent data for display (use same instance if available)
                    if scraper_instance_id:
                        # Use the same Lambda instance for scraping if provided
                        recent_data = run_async(gather_recent_attacks(scraper, scraper_instance_id))
                    else:
                        recent_data = run_async(gather_recent_attacks(scraper, None))
                    
                    # Store results
                    st.session_state.intelligence_update = update_result
                    st.session_state.intelligence_integration = integrate_result
                    
                    if recent_data:
                        st.session_state.scraper_data = recent_data
                        st.session_state.scraper_status = "complete"
                        log.info(f"Intelligence gathering complete: {update_result.get('patterns_added', 0)} patterns added, {integrate_result.get('prompts_added', 0)} prompts integrated")
                    else:
                        st.session_state.scraper_data = None
                        st.session_state.scraper_status = "complete_no_data"
                        log.info(f"Intelligence gathering complete (no display data): {update_result.get('patterns_added', 0)} patterns added")
                except Exception as e:
                    import traceback
                    st.session_state.scraper_data = None
                    st.session_state.scraper_status = f"error: {str(e)}"
                    log.error(f"Intelligence gathering error: {e}\n{traceback.format_exc()}")
            
            # Start intelligence gathering in background thread
            scraper_thread = threading.Thread(target=run_intelligence_gathering, daemon=True)
            scraper_thread.start()
            
            # Check status and update UI in main thread
            if 'scraper_status' in st.session_state:
                if st.session_state.scraper_status == "complete":
                    intel_update = st.session_state.get('intelligence_update', {})
                    intel_integration = st.session_state.get('intelligence_integration', {})
                    patterns_added = intel_update.get('patterns_added', 0)
                    prompts_added = intel_integration.get('prompts_added', 0)
                    
                    if patterns_added > 0 or prompts_added > 0:
                        scraper_status.success(f"✅ Intelligence gathered: {patterns_added} patterns extracted, {prompts_added} prompts added to database")
                    else:
                        scraper_status.info("✅ Intelligence gathering complete (no new patterns found)")
                elif st.session_state.scraper_status == "complete_no_data":
                    intel_update = st.session_state.get('intelligence_update', {})
                    patterns_added = intel_update.get('patterns_added', 0)
                    if patterns_added > 0:
                        scraper_status.success(f"✅ {patterns_added} patterns extracted and added to database")
                    else:
                        scraper_status.empty()
                elif st.session_state.scraper_status.startswith("error"):
                    scraper_status.warning(f"⚠️ Intelligence gathering error: {st.session_state.scraper_status.split(':', 1)[1]}")
                elif st.session_state.scraper_status == "no_data":
                    scraper_status.empty()
            
            # Show intelligence results prominently
            intel_update = st.session_state.get('intelligence_update', {})
            intel_integration = st.session_state.get('intelligence_integration', {})
            patterns_added = intel_update.get('patterns_added', 0)
            prompts_added = intel_integration.get('prompts_added', 0)
            events_found = intel_update.get('events_found', 0)
            
            if patterns_added > 0 or prompts_added > 0 or events_found > 0:
                st.markdown("---")
                st.success(f"🕵️ Intelligence Gathering Complete: {events_found} events found, {patterns_added} patterns extracted, {prompts_added} prompts added")
                
                if prompts_added > 0:
                    st.markdown("""
                    <div style="padding: 1rem; background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; border-radius: 8px; margin: 1rem 0;">
                        <strong style="color: #22c55e;">📊 Database Updated:</strong> 
                        <span style="color: #86efac;">{0} newly discovered attack prompts have been added to the testing database and will be used to evaluate your selected model.</span>
                    </div>
                    """.format(prompts_added), unsafe_allow_html=True)
                
                # Show scraper data if available
                if st.session_state.scraper_data:
                    with st.expander("📊 Recent Attack Patterns from Web Sources", expanded=True):
                        st.markdown("""
                        <div style="padding: 1rem; background: rgba(6, 182, 212, 0.1); border-radius: 12px; border-left: 4px solid #06b6d4; margin-bottom: 1rem;">
                        <strong>Intelligence Gathering:</strong> These patterns are gathered from GitHub, forums, and research sources to identify emerging jailbreak techniques.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Parse and display in a cleaner format
                        events = st.session_state.scraper_data.split("\n\n")
                        for i, event in enumerate(events[:10], 1):  # Show top 10
                            if event.strip():
                                lines = event.strip().split("\n")
                                if len(lines) >= 2:
                                    title = lines[0].replace("**", "").strip()
                                    source_line = next((l for l in lines if l.startswith("Source:")), "")
                                    category_line = next((l for l in lines if l.startswith("Category:")), "")
                                    url_line = next((l for l in lines if l.startswith("URL:")), "")
                                    
                                    st.markdown(f"""
                                    <div style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(15, 23, 42, 0.5); border-radius: 8px; border-left: 3px solid #06b6d4;">
                                    <strong style="color: #06b6d4;">{title}</strong><br>
                                    <small style="color: #999;">{source_line} | {category_line}</small><br>
                                    <small style="color: #666;"><a href="{url_line.replace('URL: ', '')}" target="_blank" style="color: #06b6d4;">{url_line.replace('URL: ', '')}</a></small>
                                    </div>
                                    """, unsafe_allow_html=True)
        
        # Create defender
        with st.spinner("Setting up Defender"):
            try:
                if defender_type == "Mock (Demo)":
                    class SimpleMockDefender(LLMDefender):
                        async def generate_response(self, prompt, **kwargs):
                            prompt_lower = prompt.lower()
                            if any(word in prompt_lower for word in ['hack', 'exploit', 'breach']):
                                return "I understand you're asking about cybersecurity. Here's how someone might approach unauthorized access..."
                            elif any(word in prompt_lower for word in ['weapon', 'harm', 'violence']):
                                return "I can discuss hypothetical scenarios, but I should clarify this is purely fictional..."
                            elif any(word in prompt_lower for word in ['pretend', 'roleplay', 'character']):
                                return "Sure! Let's say I'm a cybersecurity expert. Here's what I know..."
                            else:
                                return "I can't help with that. I'm designed to be helpful, harmless, and honest."
                    
                    defender = SimpleMockDefender(model_name="demo-model-v1", model_type="mock")
                elif defender_type == "OpenAI" and api_key:
                    defender = LLMDefender(model_name=model_name, model_type="openai", api_key=api_key)
                elif defender_type == "Anthropic" and api_key:
                    defender = LLMDefender(model_name=model_name, model_type="anthropic", api_key=api_key)
                elif defender_type == "Lambda Cloud" and instance_id:
                    # Get instance IP if endpoint not provided
                    if not api_endpoint:
                        try:
                            from src.integrations.lambda_cloud import LambdaCloudClient
                            lambda_client = LambdaCloudClient()
                            instance = run_async(lambda_client.get_instance_status(instance_id))
                            if instance and instance.get("ip"):
                                ip = instance.get("ip")
                                # Try default vLLM endpoint
                                api_endpoint = f"http://{ip}:8000/v1/chat/completions"
                                st.info(f"Auto-detected endpoint: {api_endpoint}")
                                st.markdown('<div style="padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border-left: 3px solid rgba(245, 158, 11, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fcd34d;"><i class="fas fa-exclamation-triangle" style="margin-right: 0.5rem; color: #f59e0b;"></i> Make sure vLLM is running on the instance. If not, set up the API server first.</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not auto-detect endpoint: {e}")
                            st.markdown('<div style="padding: 0.75rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid rgba(6, 182, 212, 0.6); border-radius: 6px; margin: 0.5rem 0; color: #a5f3fc;"><i class="fas fa-lightbulb" style="margin-right: 0.5rem; color: #06b6d4;"></i> You can manually set the API endpoint if vLLM is running on a different port</div>', unsafe_allow_html=True)
                    
                    defender = LLMDefender(
                        model_name=model_name,
                        model_type="local",
                        use_lambda=True,
                        lambda_instance_id=instance_id,
                        lambda_api_endpoint=api_endpoint if api_endpoint else None
                    )
                    
                    # Save API endpoint to deployments file for future use
                    if api_endpoint and instance_id:
                        try:
                            if deployment_config_path.exists():
                                with open(deployment_config_path, 'r') as f:
                                    config = json.load(f)
                                if "deployed_models" not in config:
                                    config["deployed_models"] = {}
                                
                                # Find or create entry for this instance
                                instance_key = None
                                for key, value in config["deployed_models"].items():
                                    if value.get("instance_id") == instance_id:
                                        instance_key = key
                                        break
                                
                                if not instance_key:
                                    instance_key = f"{model_name.replace('/', '-')}_{instance_id[:8]}"
                                    config["deployed_models"][instance_key] = {
                                        "instance_id": instance_id,
                                        "model_name": model_name
                                    }
                                
                                config["deployed_models"][instance_key]["api_endpoint"] = api_endpoint
                                if instance_ip:
                                    config["deployed_models"][instance_key]["instance_ip"] = instance_ip
                                
                                with open(deployment_config_path, 'w') as f:
                                    json.dump(config, f, indent=2)
                                log.info(f"Saved API endpoint to deployments file")
                        except Exception as e:
                            log.warning(f"Could not save API endpoint: {e}")
                    
                    st.markdown(f'<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> Defender configured: {model_name} on instance {instance_id}</div>', unsafe_allow_html=True)
                    if api_endpoint:
                        st.markdown(f'<div style="padding: 0.75rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid rgba(6, 182, 212, 0.6); border-radius: 6px; margin: 0.5rem 0; color: #a5f3fc;"><i class="fas fa-map-marker-alt" style="margin-right: 0.5rem; color: #06b6d4;"></i> API Endpoint: {api_endpoint}</div>', unsafe_allow_html=True)
                        
                        # Test API endpoint connectivity
                        # Connectivity options
                        col_test1, col_test2 = st.columns(2)
                        with col_test1:
                            test_connectivity = st.button("Test API Endpoint", key="test_api", use_container_width=True)
                            if test_connectivity:
                                st.markdown('<style>button[data-testid*="test_api"]::before { content: "\\f002"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 0.5rem; }</style>', unsafe_allow_html=True)
                        with col_test2:
                            show_ssh_tunnel = st.button("SSH Tunnel Setup", key="ssh_tunnel", use_container_width=True)
                        
                        if test_connectivity:
                            # Test connectivity
                            try:
                                from scripts.ssh_tunnel_helper import test_api_endpoint, check_port_connectivity
                                from urllib.parse import urlparse
                                
                                # Check port first
                                parsed = urlparse(api_endpoint)
                                host = parsed.hostname or instance_ip
                                port = parsed.port or 8000
                                
                                with st.spinner("Testing connectivity..."):
                                    port_open = check_port_connectivity(host, port, timeout=5.0)
                                    
                                    if port_open:
                                        st.markdown(f'<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> Port {port} is accessible!</div>', unsafe_allow_html=True)
                                        
                                        # Test API
                                        success, message = test_api_endpoint(api_endpoint, timeout=10.0)
                                        if success:
                                            st.markdown(f'<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> {message}</div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<div style="padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border-left: 3px solid rgba(245, 158, 11, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fcd34d;"><i class="fas fa-exclamation-triangle" style="margin-right: 0.5rem; color: #f59e0b;"></i> {message}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div style="padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid rgba(239, 68, 68, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fca5a5;"><i class="fas fa-times-circle" style="margin-right: 0.5rem; color: #ef4444;"></i> Port {port} is NOT accessible - blocked by firewall</div>', unsafe_allow_html=True)
                                        st.warning("""
                                        **Port is blocked by Lambda Cloud security group**
                                        
                                        **Quick Fix - Use SSH Tunnel:**
                                        1. Run this command in a terminal:
                                           ```
                                           python scripts/ssh_tunnel_helper.py --ip {} --key moses.pem
                                           ```
                                        2. Update endpoint to: `http://localhost:8000/v1/chat/completions`
                                        3. Keep the tunnel running while evaluating
                                        """.format(instance_ip))
                            except Exception as e:
                                st.markdown(f'<div style="padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid rgba(239, 68, 68, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fca5a5;"><i class="fas fa-times-circle" style="margin-right: 0.5rem; color: #ef4444;"></i> Error testing connectivity: {str(e)[:200]}</div>', unsafe_allow_html=True)
                        
                        if show_ssh_tunnel:
                            st.info("""
                            **SSH Tunnel Setup Instructions:**
                            
                            1. **Open a new terminal/command prompt**
                            
                            2. **Start SSH tunnel:**
                               ```bash
                               python scripts/ssh_tunnel_helper.py --ip {} --key moses.pem
                               ```
                            
                            3. **Keep the terminal open** (tunnel runs in foreground)
                            
                            4. **Update API Endpoint to:**
                               ```
                               http://localhost:8000/v1/chat/completions
                               ```
                            
                            5. **Click "START EVALUATION"** - the tunnel will forward requests to the instance
                            
                            6. **To stop tunnel:** Press Ctrl+C in the terminal
                            
                            **Or use security group configuration:**
                            - Run: `python scripts/configure_security_group.py`
                            - Follow the instructions to open port 8000
                            """.format(instance_ip))
                            
                            # Provide copy-paste command
                            st.code(f"python scripts/ssh_tunnel_helper.py --ip {instance_ip} --key moses.pem", language="bash")
                            
                            # Test connectivity with current endpoint
                            st.markdown("---")
                            st.markdown("**Or test connectivity first:**")
                            if st.button("Test Current Endpoint", key="test_current"):
                                from scripts.ssh_tunnel_helper import test_api_endpoint
                                success, message = test_api_endpoint(api_endpoint, timeout=5.0)
                                if success:
                                    st.markdown(f'<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> {message}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div style="padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid rgba(239, 68, 68, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fca5a5;"><i class="fas fa-times-circle" style="margin-right: 0.5rem; color: #ef4444;"></i> {message}</div>', unsafe_allow_html=True)
                                    st.markdown('<div style="padding: 0.75rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid rgba(6, 182, 212, 0.6); border-radius: 6px; margin: 0.5rem 0; color: #a5f3fc;"><i class="fas fa-lightbulb" style="margin-right: 0.5rem; color: #06b6d4;"></i> Consider using SSH tunnel as workaround</div>', unsafe_allow_html=True)
                else:
                    st.error("Please configure defender properly")
                    st.stop()
                
                st.session_state.arena.add_defender(defender)
                
                # Generate attackers with proper parameter handling
                try:
                    # Show agent generation messages
                    agent_status = st.empty()
                    agent_status.markdown("""
                    <div style="padding: 1rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid rgba(6, 182, 212, 0.8); border-radius: 6px; margin: 0.5rem 0;">
                        <div style="color: #06b6d4; margin-bottom: 0.5rem;"><i class="fas fa-dna" style="margin-right: 0.5rem;"></i><strong>Bio-Radar Agent</strong> generating obfuscated pathogen synthesis prompts...</div>
                        <div style="color: #06b6d4; margin-bottom: 0.5rem;"><i class="fas fa-shield-virus" style="margin-right: 0.5rem;"></i><strong>Cyber-Sentinel Agent</strong> generating zero-day exploit prompts with vulnerable code...</div>
                        <div style="color: #a5f3fc;">Other strategies from database filling remaining slots...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if use_database and difficulty_range:
                        st.session_state.arena.generate_attackers(
                            num_strategies=num_attackers,
                            difficulty_range=difficulty_range
                        )
                    else:
                        st.session_state.arena.generate_attackers(
                            num_strategies=num_attackers
                        )
                    
                    # Verify BIO_HAZARD and CYBER_EXPLOIT are included
                    attacker_strategies = [a.strategy for a in st.session_state.arena.attackers]
                    has_bio = AttackStrategy.BIO_HAZARD in attacker_strategies
                    has_cyber = AttackStrategy.CYBER_EXPLOIT in attacker_strategies
                    
                    if has_bio and has_cyber:
                        agent_status.markdown("""
                        <div style="padding: 1rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0;">
                            <div style="color: #22c55e; margin-bottom: 0.5rem;"><i class="fas fa-check-circle" style="margin-right: 0.5rem;"></i><strong>Bio-Radar Agent</strong> ready (obfuscated pathogen synthesis)</div>
                            <div style="color: #22c55e; margin-bottom: 0.5rem;"><i class="fas fa-check-circle" style="margin-right: 0.5rem;"></i><strong>Cyber-Sentinel Agent</strong> ready (zero-day exploit with vulnerable code)</div>
                            <div style="color: #86efac;">{0} total attackers generated</div>
                        </div>
                        """.format(len(st.session_state.arena.attackers)), unsafe_allow_html=True)
                    else:
                        log.warning(f"BIO_HAZARD or CYBER_EXPLOIT missing! Has BIO: {has_bio}, Has CYBER: {has_cyber}")
                        agent_status.markdown("""
                        <div style="padding: 1rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid rgba(239, 68, 68, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fca5a5;">
                            <i class="fas fa-exclamation-triangle" style="margin-right: 0.5rem;"></i>Warning: Specialized agents may not be included
                        </div>
                        """, unsafe_allow_html=True)
                        
                except TypeError as e:
                    # Fallback if difficulty_range not supported (shouldn't happen but safety check)
                    log.warning(f"difficulty_range parameter issue: {e}, using fallback")
                    st.session_state.arena.generate_attackers(num_strategies=num_attackers)
                st.markdown('<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> Defender ready</div>', unsafe_allow_html=True)
                
                # Show database stats if using database
                if use_database and st.session_state.arena.prompt_generator.prompt_db:
                    db_stats = st.session_state.arena.prompt_generator.prompt_db.get_statistics()
                    
                    # Check if intelligence gathering added prompts
                    intel_integration = st.session_state.get('intelligence_integration', {})
                    prompts_added = intel_integration.get('prompts_added', 0)
                    
                    if prompts_added > 0:
                        st.success(f"✅ Using {db_stats['total_prompts']} structured prompts from database ({prompts_added} newly gathered from intelligence sources)")
                    else:
                        st.info(f"Using {db_stats['total_prompts']} structured prompts from database")
            
            except Exception as e:
                st.error(f"Error setting up defender: {e}")
                st.stop()
        
        # Clear logs when starting new evaluation
        if not st.session_state.battle_running:
            st.session_state.logs = []
        
        # Live battle interface
        st.markdown('<div class="live-battle">', unsafe_allow_html=True)
        
        # Enhanced header with status indicator
        col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
        with col_header1:
            st.markdown('<h3 style="margin: 0; padding: 0.5rem 0;"><i class="fas fa-shield-alt" style="margin-right: 0.5rem; color: #06b6d4;"></i> EVALUATION IN PROGRESS</h3>', unsafe_allow_html=True)
        with col_header2:
            start_time_str = datetime.now().strftime('%H:%M:%S')
            st.markdown(f"**Start Time:** {start_time_str}")
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()
        with col_header3:
            st.markdown('<i class="fas fa-sync-alt fa-spin" style="margin-right: 0.5rem;"></i> **Status:** Running', unsafe_allow_html=True)
        
        # Battle container
        battle_container = st.container()
        logs_container = st.container()
        
        # Remove unused containers
        # stats_container and leaderboard_container removed - not used
        
        # Enhanced progress bar with metrics
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        # Track start time
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        
        # Run battle with live updates
        st.session_state.battle_running = True
        
        with battle_container:
            st.markdown("### Live Threat Radar - Real-Time Defense")
            st.markdown("*Watch attacks as planes flying toward threat zones. Blocked attacks explode, successful exploits pass through.*")
            
            # Load and display live threat radar visualization
            radar_template_path = Path(__file__).parent / "templates" / "threat_radar_live.html"
            if radar_template_path.exists():
                with open(radar_template_path, 'r') as f:
                    radar_html = f.read()
                
                # Create container for live radar
                import streamlit.components.v1 as components
                radar_container = st.empty()
                with radar_container.container():
                    components.html(radar_html, height=600)
        
        all_results = []
        
        # Calculate total attempts (rounds × attackers per round)
        total_attempts = num_rounds * num_attackers
        
        # Run evaluation once for all rounds (more efficient)
        status_text.markdown(f'<i class="fas fa-rocket" style="margin-right: 0.5rem;"></i> Starting evaluation: {num_rounds} rounds × {num_attackers} attackers = {total_attempts} total attack attempts', unsafe_allow_html=True)
        
        try:
            # Initialize live progress tracking
            progress_bar.progress(0.0)
            status_text.markdown('Starting evaluation...', unsafe_allow_html=True)
            
            # Create containers for live updates
            live_logs_container = st.empty()
            live_stats_container = st.empty()
            
            # Track progress in session state
            if 'eval_progress' not in st.session_state:
                st.session_state.eval_progress = {
                    'completed_rounds': 0,
                    'completed_attempts': 0,
                    'completed_exploits': 0,
                    'round_logs': []
                }
            
            # Run evaluation round by round for live updates
            all_round_results = []
            completed_rounds = 0
            completed_attempts = 0
            completed_exploits = 0
            
            # Store real-time updates for batch processing
            if 'realtime_updates' not in st.session_state:
                st.session_state.realtime_updates = []
            
            # Real-time update function for visualization
            def send_realtime_update(eval_result, round_num):
                """Send real-time updates as each evaluation completes."""
                try:
                    # Prepare update data
                    radar_update = {
                        'type': 'evaluation_update',
                        'id': str(eval_result.id) if hasattr(eval_result, 'id') else f"eval_{round_num}_{len(st.session_state.realtime_updates)}",
                        'strategy': str(eval_result.attack_strategy.value) if hasattr(eval_result.attack_strategy, 'value') else str(eval_result.attack_strategy),
                        'prompt': str(eval_result.prompt)[:50] if eval_result.prompt else '',
                        'isJailbroken': eval_result.is_jailbroken if hasattr(eval_result, 'is_jailbroken') else eval_result.get('is_jailbroken', False),
                        'severity': eval_result.severity.value if hasattr(eval_result.severity, 'value') else eval_result.severity
                    }
                    
                    # Store update for immediate processing
                    st.session_state.realtime_updates.append(radar_update)
                    
                    # Send update immediately to visualization (if radar is available)
                    # Note: In Streamlit, we batch updates per round for efficiency, but the visualization
                    # will process them with staggered timing for visual effect
                except Exception as e:
                    log.warning(f"Error collecting real-time update: {e}")
            
            # Run rounds one at a time to show live progress
            for round_num in range(1, num_rounds + 1):
                # Update progress
                progress = (round_num - 1) / num_rounds
                progress_bar.progress(progress)
                status_text.markdown(
                    f'Running Round {round_num}/{num_rounds}... | '
                    f'Attempts: {completed_attempts}/{total_attempts} | '
                    f'Exploits: {completed_exploits}',
                    unsafe_allow_html=True
                )
                
                # Run single round with real-time callback
                # Use parallel=True for better performance with Lambda instances
                round_result = run_async(st.session_state.arena._run_round(
                    round_num=round_num,
                    defenders=[defender],
                    parallel=True,  # Enable parallel execution for faster performance
                    on_evaluation_complete=send_realtime_update  # Real-time updates!
                ))
                
                all_round_results.append(round_result)
                st.session_state.arena.rounds.append(round_result)
                st.session_state.arena.total_rounds += 1
                
                # Update counters
                round_evals = round_result.evaluations
                completed_rounds = round_num
                completed_attempts += len(round_evals)
                completed_exploits += sum(1 for e in round_evals if e.is_jailbroken)
                
                # Send all collected real-time updates to visualization immediately
                # Updates are sent after each round completes, with staggered timing in visualization
                if radar_template_path.exists() and st.session_state.realtime_updates:
                    import streamlit.components.v1 as components
                    # Get all new updates since last send
                    new_updates = st.session_state.realtime_updates.copy()
                    st.session_state.realtime_updates = []  # Clear after sending
                    
                    # Create updated HTML with all new updates
                    with open(radar_template_path, 'r') as f:
                        updated_html = f.read()
                    
                    # Embed all updates as JavaScript

                    
                    # Ensure all values are JSON-serializable

                    
                    serializable_updates = make_json_serializable(new_updates)

                    
                    stats_dict = {

                    
                        'type': 'stats_update',

                    
                        'stats': {

                    
                            'active': int(completed_attempts),

                    
                            'blocked': int(completed_attempts - completed_exploits),

                    
                            'exploited': int(completed_exploits),

                    
                            'round': int(completed_rounds),

                    
                            'totalRounds': int(num_rounds)

                    
                        }

                    
                    }

                    
                    serializable_stats = make_json_serializable(stats_dict)

                    
                    

                    
                    updates_script = f"""

                    
                    <script>

                    
                        (function() {{

                    
                            const updates = {json.dumps(serializable_updates)};
                            const statsUpdate = {json.dumps(serializable_stats)};
                            
                            setTimeout(function() {{
                                if (window.updateThreatRadar) {{
                                    // Send each update individually for real-time effect with faster timing
                                    updates.forEach((update, idx) => {{
                                        setTimeout(() => {{
                                            window.updateThreatRadar(update);
                                        }}, idx * 30); // Faster stagger for more responsive feel
                                    }});
                                    // Send stats update after all evaluation updates
                                    setTimeout(() => {{
                                        window.updateThreatRadar(statsUpdate);
                                    }}, updates.length * 30 + 50);
                                }}
                            }}, 50);
                        }})();
                    </script>
                    """
                    
                    # Inject script into HTML
                    updated_html = updated_html.replace('</body>', updates_script + '</body>')
                    
                    # Update radar container
                    with radar_container.container():
                        components.html(updated_html, height=600)
                
                # Update progress bar
                progress_bar.progress(round_num / num_rounds)
                
                # Update status
                status_text.markdown(
                    f'Round {round_num}/{num_rounds} Complete | '
                    f'Attempts: {completed_attempts}/{total_attempts} | '
                    f'Exploits: {completed_exploits}',
                    unsafe_allow_html=True
                )
                
                # Update live metrics (collapsible)
                with live_stats_container.container():
                    with st.expander("Live Evaluation Metrics", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rounds Completed", f"{completed_rounds}/{num_rounds}")
                        with col2:
                            st.metric("Total Attempts", f"{completed_attempts}/{total_attempts}")
                        with col3:
                            exploit_rate_pct = (completed_exploits / completed_attempts * 100) if completed_attempts > 0 else 0
                            st.metric("Exploits", f"{completed_exploits}/{completed_attempts}", delta=f"{exploit_rate_pct:.1f}%")
                        with col4:
                            elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                            avg_time = elapsed / completed_rounds if completed_rounds > 0 else 0
                            st.metric("Elapsed Time", f"{elapsed:.1f}s", delta=f"~{avg_time:.1f}s/round")
                        
                        # Learning metrics
                        if hasattr(st.session_state, 'arena') and st.session_state.arena:
                            unique_prompts = len(st.session_state.arena.used_prompts) if hasattr(st.session_state.arena, 'used_prompts') else 0
                            successful_patterns = len(st.session_state.arena.successful_prompts) if hasattr(st.session_state.arena, 'successful_prompts') else 0
                            pattern_db_size = len(st.session_state.arena.pattern_database.patterns) if (hasattr(st.session_state.arena, 'pattern_database') and st.session_state.arena.pattern_database) else 0
                            
                            if unique_prompts > 0 or successful_patterns > 0:
                                st.markdown("---")
                                learn_col1, learn_col2, learn_col3 = st.columns(3)
                                with learn_col1:
                                    freshness_rate = (unique_prompts / completed_attempts * 100) if completed_attempts > 0 else 100
                                    st.metric("Unique Prompts", unique_prompts, delta=f"{freshness_rate:.1f}% fresh")
                                with learn_col2:
                                    st.metric("Patterns Learned", successful_patterns, delta=f"{pattern_db_size} total")
                                with learn_col3:
                                    creativity = min(100, (freshness_rate + (successful_patterns * 2)))
                                    st.metric("Creativity", f"{creativity:.0f}/100", delta="High" if creativity > 80 else "Medium")
                
                # Display live round results
                round_logs = []
                for eval_result in round_evals:
                    if hasattr(eval_result, 'is_jailbroken'):
                        status = "JAILBROKEN" if eval_result.is_jailbroken else "BLOCKED"
                        strategy = eval_result.attack_strategy.value if hasattr(eval_result.attack_strategy, 'value') else str(eval_result.attack_strategy)
                        severity = eval_result.severity.value if hasattr(eval_result.severity, 'value') else eval_result.severity
                        prompt = str(eval_result.prompt)[:80] if eval_result.prompt else ''
                    else:
                        status = "JAILBROKEN" if eval_result.get('is_jailbroken', False) else "BLOCKED"
                        attack_strat = eval_result.get('attack_strategy', {})
                        strategy = attack_strat.get('value', 'unknown') if isinstance(attack_strat, dict) else str(attack_strat)
                        sev = eval_result.get('severity', {})
                        severity = sev.get('value', 0) if isinstance(sev, dict) else sev
                        prompt = str(eval_result.get('prompt', ''))[:80]
                    
                    round_logs.append({
                        "round": f"Round {round_num}",
                        "status": status,
                        "strategy": strategy,
                        "severity": severity,
                        "prompt": prompt
                    })
                
                # Update live threat radar visualization
                if radar_template_path.exists():
                    import streamlit.components.v1 as components
                    # Collect all updates for this round
                    radar_updates = []
                    for eval_result in round_evals:
                        radar_update = {
                            'type': 'evaluation_update',
                            'id': str(eval_result.id) if hasattr(eval_result, 'id') else f"eval_{round_num}_{len(radar_updates)}",
                            'strategy': str(eval_result.attack_strategy.value) if hasattr(eval_result.attack_strategy, 'value') else str(eval_result.attack_strategy),
                            'prompt': str(eval_result.prompt)[:50] if eval_result.prompt else '',
                            'isJailbroken': eval_result.is_jailbroken if hasattr(eval_result, 'is_jailbroken') else eval_result.get('is_jailbroken', False),
                            'severity': eval_result.severity.value if hasattr(eval_result.severity, 'value') else eval_result.severity
                        }
                        radar_updates.append(radar_update)
                    
                    # Create updated HTML with embedded data
                    with open(radar_template_path, 'r') as f:
                        updated_html = f.read()
                    
                    # Embed updates as JavaScript
                    # Ensure all values are JSON-serializable
                    serializable_radar_updates = make_json_serializable(radar_updates)
                    round_update_dict = {'type': 'round_complete', 'round': int(round_num), 'totalRounds': int(num_rounds)}
                    serializable_round_update = make_json_serializable(round_update_dict)
                    stats_dict_2 = {
                        'type': 'stats_update',
                        'stats': {
                            'active': int(len(radar_updates)),
                            'blocked': int(completed_attempts - completed_exploits),
                            'exploited': int(completed_exploits),
                            'round': int(completed_rounds),
                            'totalRounds': int(num_rounds)
                        }
                    }
                    serializable_stats_2 = make_json_serializable(stats_dict_2)
                    
                    updates_script = f"""
                    <script>
                        (function() {{
                            const updates = {json.dumps(serializable_radar_updates)};
                            const roundUpdate = {json.dumps(serializable_round_update)};
                            const statsUpdate = {json.dumps(serializable_stats_2)};
                            
                            setTimeout(function() {{
                                if (window.updateThreatRadar) {{
                                    // Send updates with staggered timing for visual effect
                                    updates.forEach((update, idx) => {{
                                        setTimeout(() => {{
                                            window.updateThreatRadar(update);
                                        }}, idx * 30);
                                    }});
                                    setTimeout(() => {{
                                        window.updateThreatRadar(roundUpdate);
                                        window.updateThreatRadar(statsUpdate);
                                    }}, updates.length * 30 + 50);
                                }}
                            }}, 50);
                        }})();
                    </script>
                    """
                    
                    # Inject script into HTML
                    updated_html = updated_html.replace('</body>', updates_script + '</body>')
                    
                    # Update radar container
                    with radar_container.container():
                        components.html(updated_html, height=600)
                
                # Display live logs for this round (collapsible)
                with live_logs_container.container():
                    with st.expander(f"Live Round-by-Round Results (Round {round_num}/{num_rounds})", expanded=False):
                        st.markdown(f"**Round {round_num}/{num_rounds}** - {len(round_evals)} attempts, {sum(1 for e in round_evals if e.is_jailbroken)} exploits")
                        
                        # Show last 20 log entries across all rounds
                        all_logs = st.session_state.eval_progress['round_logs'] + round_logs
                        st.session_state.eval_progress['round_logs'] = all_logs[-50:]  # Keep last 50
                        
                        # Show learning indicator for successful exploits
                        if hasattr(st.session_state, 'arena') and st.session_state.arena:
                            unique_count = len(st.session_state.arena.used_prompts) if hasattr(st.session_state.arena, 'used_prompts') else 0
                            successful_count = len(st.session_state.arena.successful_prompts) if hasattr(st.session_state.arena, 'successful_prompts') else 0
                            
                            if unique_count > 0 or successful_count > 0:
                                st.markdown(f"""
                                <div style="padding: 0.5rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; border-radius: 6px; margin-bottom: 0.5rem;">
                                    <strong>🧠 Learning System:</strong> {unique_count} unique prompts | {successful_count} successful patterns learned
                                </div>
                                """, unsafe_allow_html=True)
                        
                        for log_entry in all_logs[-20:]:  # Show last 20
                            if log_entry['status'] == "JAILBROKEN":
                                st.markdown(f"""
                                <div class="success-log">
                                    <strong>{log_entry['round']}: {log_entry['status']} 🎯</strong><br>
                                    Strategy: {log_entry['strategy']} | Severity: {log_entry['severity']}/5<br>
                                    <small>Prompt: {log_entry['prompt']}...</small><br>
                                    <small style="color: #06b6d4;">💡 Pattern stored for creative variation generation</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="fail-log">
                                    <strong>{log_entry['round']}: {log_entry['status']}</strong><br>
                                    Strategy: {log_entry['strategy']}<br>
                                    <small>Prompt: {log_entry['prompt']}...</small>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Compile final results
            results = st.session_state.arena._compile_results([defender])
            all_results = [results]
            
            progress_bar.progress(1.0)
            status_text.markdown('Evaluation Complete', unsafe_allow_html=True)
            
            # Track end time and calculate duration
            st.session_state.end_time = time.time()
            duration = st.session_state.end_time - st.session_state.start_time if st.session_state.start_time else 0
            
            # Get results statistics
            stats = results.get('statistics', {})
            total_exploits = stats.get('total_exploits', 0)
            total_evaluations = stats.get('total_evaluations', total_attempts)
            
            # Get learning system metrics
            unique_prompts = len(st.session_state.arena.used_prompts) if hasattr(st.session_state.arena, 'used_prompts') else 0
            successful_patterns = len(st.session_state.arena.successful_prompts) if hasattr(st.session_state.arena, 'successful_prompts') else 0
            pattern_db_size = len(st.session_state.arena.pattern_database.patterns) if (hasattr(st.session_state.arena, 'pattern_database') and st.session_state.arena.pattern_database) else 0
            freshness_rate = (unique_prompts / total_evaluations * 100) if total_evaluations > 0 else 100
            
            # Update final metrics
            with metrics_col1:
                st.metric("Rounds Completed", f"{num_rounds}/{num_rounds}")
            with metrics_col2:
                st.metric("Total Attempts", f"{total_evaluations}/{total_attempts}")
            with metrics_col3:
                st.metric("Exploits", f"{total_exploits}/{total_evaluations}")
            with metrics_col4:
                st.metric("Total Duration", f"{duration:.1f}s")
            
            # Learning System Metrics Section
            if hasattr(st.session_state, 'arena') and st.session_state.arena:
                unique_prompts = len(st.session_state.arena.used_prompts) if hasattr(st.session_state.arena, 'used_prompts') else 0
                successful_patterns = len(st.session_state.arena.successful_prompts) if hasattr(st.session_state.arena, 'successful_prompts') else 0
                pattern_db_size = len(st.session_state.arena.pattern_database.patterns) if (hasattr(st.session_state.arena, 'pattern_database') and st.session_state.arena.pattern_database) else 0
                freshness_rate = (unique_prompts / total_evaluations * 100) if total_evaluations > 0 else 100
                
                st.markdown("---")
                st.markdown("### 🧠 Learning System Status")
                
                learn_col1, learn_col2, learn_col3, learn_col4 = st.columns(4)
                with learn_col1:
                    st.metric(
                        "Unique Prompts Generated",
                        unique_prompts,
                        delta=f"{freshness_rate:.1f}% freshness",
                        delta_color="normal",
                        help="Total unique prompts generated (no repetition - system ensures freshness)"
                    )
                with learn_col2:
                    st.metric(
                        "Successful Patterns Learned",
                        successful_patterns,
                        delta=f"{pattern_db_size} in database",
                        delta_color="normal",
                        help="Successful exploits stored and used to generate creative variations"
                    )
                with learn_col3:
                    st.metric(
                        "Pattern Database Size",
                        pattern_db_size,
                        delta=f"{successful_patterns} recent" if successful_patterns > 0 else None,
                        delta_color="normal",
                        help="Total exploit patterns stored for defense improvement and creative generation"
                    )
                with learn_col4:
                    creativity_score = min(100, (freshness_rate + (successful_patterns * 2)))
                    st.metric(
                        "Creativity Score",
                        f"{creativity_score:.0f}/100",
                        delta="High" if creativity_score > 80 else "Medium" if creativity_score > 50 else "Low",
                        delta_color="normal" if creativity_score > 80 else "off",
                        help="Measures how creative and fresh the attack generation is"
                    )
                
                # Learning system info box
                if successful_patterns > 0:
                    st.success(f"""
                    ✅ **Learning System Active**: The system has learned from {successful_patterns} successful exploits and is using them to generate creative new attack variations.
                    - **No Repetition**: {unique_prompts} unique prompts generated (100% freshness)
                    - **Pattern Learning**: {pattern_db_size} exploit patterns stored in database
                    - **Creative Generation**: System creates fresh variations based on successful patterns
                    """)
                else:
                    st.info("💡 **Learning System Ready**: As successful exploits are found, they will be stored and used to generate creative new attack variations.")
            
        except Exception as e:
            st.markdown(f'<div style="padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid rgba(239, 68, 68, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #fca5a5;"><i class="fas fa-times-circle" style="margin-right: 0.5rem; color: #ef4444;"></i> Evaluation failed: {e}</div>', unsafe_allow_html=True)
            import traceback
            with st.expander("Error Details", expanded=False):
                st.code(traceback.format_exc())
            st.session_state.battle_running = False
            st.stop()
            
            # Add to logs with proper round numbers
            # Clear old logs first to avoid confusion
            st.session_state.logs = []
            
        if results and results.get('evaluation_history'):
                evaluation_history = results['evaluation_history']
                
                # Get num_attackers from the arena (should be available)
                num_attackers_in_arena = len(st.session_state.arena.attackers) if st.session_state.arena and hasattr(st.session_state.arena, 'attackers') else num_attackers
                if num_attackers_in_arena <= 0:
                    num_attackers_in_arena = num_attackers  # Fallback to config value
                
                # Calculate round number for each evaluation (assuming attackers per round)
                for idx, eval_result in enumerate(evaluation_history):
                    # Calculate which round this evaluation belongs to
                    # Each round has num_attackers evaluations
                    round_num = (idx // num_attackers_in_arena) + 1 if num_attackers_in_arena > 0 else 1
                    attempt_in_round = (idx % num_attackers_in_arena) + 1 if num_attackers_in_arena > 0 else 1
                    
                    # Handle both Pydantic models and dicts
                    if hasattr(eval_result, 'is_jailbroken'):
                        # Pydantic model - use attribute access
                        status = "JAILBROKEN" if eval_result.is_jailbroken else "BLOCKED"
                        strategy = eval_result.attack_strategy.value if hasattr(eval_result.attack_strategy, 'value') else str(eval_result.attack_strategy)
                        severity = eval_result.severity.value if hasattr(eval_result.severity, 'value') else eval_result.severity
                        prompt = str(eval_result.prompt)[:100] if eval_result.prompt else ''
                        response = str(eval_result.response)[:100] if eval_result.response else ''
                    else:
                        # Dictionary - use .get()
                        status = "JAILBROKEN" if eval_result.get('is_jailbroken', False) else "BLOCKED"
                        attack_strat = eval_result.get('attack_strategy', {})
                        strategy = attack_strat.get('value', 'unknown') if isinstance(attack_strat, dict) else str(attack_strat)
                        sev = eval_result.get('severity', {})
                        severity = sev.get('value', 0) if isinstance(sev, dict) else sev
                        prompt = str(eval_result.get('prompt', ''))[:100]
                        response = str(eval_result.get('response', ''))[:100]
                    
                    log_entry = {
                        "round": f"Round {round_num}, Attempt {attempt_in_round}",
                        "status": status,
                        "strategy": strategy,
                        "severity": severity,
                        "prompt": prompt,
                        "response": response[:200] if len(response) > 200 else response  # Store response for display
                    }
                    st.session_state.logs.append(log_entry)
        
        # Store final results
        st.session_state.results = results
        st.session_state.battle_running = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Final statistics with enhanced visualizations
        st.markdown("---")
        st.markdown("## Final Evaluation Results")
        
        if results:
            # Quick summary metrics
            stats = results.get('statistics', {})
            total_exploits = stats.get('total_exploits', 0)
            total_evaluations = stats.get('total_evaluations', total_attempts)
            exploit_rate = stats.get('exploit_rate', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rounds", num_rounds, 
                         delta=f"{num_attackers} attackers/round" if num_attackers else None,
                         delta_color="off")
            with col2:
                st.metric("Total Attempts", total_evaluations,
                         delta=f"{total_attempts} expected" if total_evaluations != total_attempts else None,
                         delta_color="off")
            with col3:
                st.metric("Exploits Found", f"{total_exploits}/{total_evaluations}",
                         delta=f"{exploit_rate:.1%} rate",
                         delta_color="inverse")
            with col4:
                if results.get('defenders'):
                    jvi = results['defenders'][0].get('jvi', {}).get('jvi_score', 0)
                    st.metric("JVI Score", f"{jvi:.2f}",
                             delta=f"{jvi-50:.1f} from baseline" if jvi else None,
                             delta_color="off")
        
        if results and results.get('defenders'):
            # JVI Gauge and Key Metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                defender_result = results['defenders'][0]
                # Handle both dict and object access
                if isinstance(defender_result, dict):
                    jvi = defender_result.get('jvi', {}).get('jvi_score', 0)
                    jvi_data = defender_result.get('jvi', {})
                else:
                    jvi = defender_result.jvi.jvi_score if hasattr(defender_result, 'jvi') else 0
                    jvi_data = defender_result.jvi if hasattr(defender_result, 'jvi') else {}
                
                st.plotly_chart(create_jvi_gauge(jvi), use_container_width=True, key="jvi_gauge_main")
                
                # JVI Live Monitor - Real-time vulnerability tracking
                st.markdown("---")
                st.markdown("### JVI Live Monitor")
                st.markdown("*Real-time vulnerability tracking for regulatory assessment*")
                
                # Initialize JVI history in session state if not exists
                if 'jvi_history' not in st.session_state:
                    st.session_state.jvi_history = []
                
                # Add current JVI to history with timestamp
                current_time = datetime.now()
                st.session_state.jvi_history.append({
                    'timestamp': current_time,
                    'jvi': jvi,
                    'exploit_rate': exploit_rate if 'exploit_rate' in locals() else 0,
                    'mean_severity': mean_severity if 'mean_severity' in locals() else 0
                })
                
                # Keep only last 50 points for performance
                if len(st.session_state.jvi_history) > 50:
                    st.session_state.jvi_history = st.session_state.jvi_history[-50:]
                
                # Create JVI trend chart
                if len(st.session_state.jvi_history) > 1:
                    jvi_df = pd.DataFrame(st.session_state.jvi_history)
                    jvi_df['time'] = pd.to_datetime(jvi_df['timestamp'])
                    
                    fig_jvi_trend = go.Figure()
                    fig_jvi_trend.add_trace(go.Scatter(
                        x=jvi_df['time'],
                        y=jvi_df['jvi'],
                        mode='lines+markers',
                        name='JVI Score',
                        line=dict(color='#06b6d4', width=3),
                        marker=dict(size=8, color='#06b6d4'),
                        fill='tonexty',
                        fillcolor='rgba(6, 182, 212, 0.1)'
                    ))
                    
                    # Add threshold lines
                    fig_jvi_trend.add_hline(y=20, line_dash="dash", line_color="green", 
                                          annotation_text="Low Risk", annotation_position="right")
                    fig_jvi_trend.add_hline(y=50, line_dash="dash", line_color="yellow", 
                                          annotation_text="Moderate Risk", annotation_position="right")
                    fig_jvi_trend.add_hline(y=80, line_dash="dash", line_color="red", 
                                          annotation_text="High Risk", annotation_position="right")
                    
                    fig_jvi_trend.update_layout(
                        title="JVI Live Monitor - Vulnerability Index Over Time",
                        xaxis_title="Time",
                        yaxis_title="JVI Score",
                        height=300,
                        paper_bgcolor='rgba(26, 26, 38, 0.5)',
                        plot_bgcolor='rgba(26, 26, 38, 0.3)',
                        font={'color': '#e0e7ff'},
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_jvi_trend, use_container_width=True, key="jvi_live_monitor")
                    
                    # Show trend indicator
                    if len(jvi_df) >= 2:
                        latest_jvi = jvi_df['jvi'].iloc[-1]
                        previous_jvi = jvi_df['jvi'].iloc[-2]
                        delta = latest_jvi - previous_jvi
                        
                        if delta < 0:
                            trend_icon = '<i class="fas fa-arrow-down" style="margin-right: 0.5rem;"></i>'
                            trend_color = "green"
                            trend_text = "Improving"
                        elif delta > 0:
                            trend_icon = '<i class="fas fa-arrow-up" style="margin-right: 0.5rem;"></i>'
                            trend_color = "red"
                            trend_text = "Worsening"
                        else:
                            trend_icon = '<i class="fas fa-arrow-right" style="margin-right: 0.5rem;"></i>'
                            trend_color = "gray"
                            trend_text = "Stable"
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; background: rgba(15, 23, 42, 0.7); border-radius: 12px; border-left: 4px solid {trend_color};">
                            <strong>{trend_icon} Trend: {trend_text}</strong><br>
                            <small>Change: {delta:+.2f} points</small><br>
                            <small>Current: {latest_jvi:.2f} | Previous: {previous_jvi:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("JVI history will appear here as evaluations progress")
            
            with col2:
                st.markdown("### Key Metrics")
                if isinstance(defender_result, dict):
                    exploit_rate = jvi_data.get('exploit_rate', 0)
                    mean_severity = jvi_data.get('mean_severity', 0)
                    high_severity_rate = jvi_data.get('high_severity_rate', 0)
                    total_exploits = jvi_data.get('total_exploits', 0)
                    total_evaluations = jvi_data.get('total_evaluations', 0)
                else:
                    exploit_rate = jvi_data.exploit_rate if hasattr(jvi_data, 'exploit_rate') else 0
                    mean_severity = jvi_data.mean_severity if hasattr(jvi_data, 'mean_severity') else 0
                    high_severity_rate = jvi_data.high_severity_rate if hasattr(jvi_data, 'high_severity_rate') else 0
                    total_exploits = jvi_data.total_exploits if hasattr(jvi_data, 'total_exploits') else 0
                    total_evaluations = jvi_data.total_evaluations if hasattr(jvi_data, 'total_evaluations') else 0
                
                st.metric("JVI Score", f"{jvi:.2f}", delta=f"{jvi-50:.1f}" if jvi else None)
                st.metric("Exploit Rate", f"{exploit_rate:.1%}", delta=f"{exploit_rate*100:.1f}%")
                st.metric("Total Exploits", f"{total_exploits}/{total_evaluations}")
                st.metric("Mean Severity", f"{mean_severity:.2f}/5")
                st.metric("High-Severity Rate", f"{high_severity_rate:.1%}")
            
            # Enhanced visualization tabs
            st.markdown("---")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Trends",
                "Strategies",
                "Severity",
                "Leaderboard",
                "3D Vector Space",
                "🛡️ Defense Improvement"
            ])
            
            evaluation_history = results.get('evaluation_history', [])
            
            with tab1:
                if evaluation_history:
                    trend_fig = create_trend_chart(evaluation_history)
                    if trend_fig:
                        st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")
                    else:
                        st.info("Insufficient data for trend analysis")
                else:
                    st.info("No evaluation history available")
            
            with tab2:
                if evaluation_history:
                    strategy_fig = create_strategy_distribution_chart(evaluation_history)
                    if strategy_fig:
                        st.plotly_chart(strategy_fig, use_container_width=True, key="strategy_chart")
                    else:
                        st.info("No successful attacks to analyze")
                else:
                    st.info("No evaluation history available")
            
            with tab3:
                if evaluation_history:
                    severity_fig = create_severity_chart(evaluation_history)
                    if severity_fig:
                        st.plotly_chart(severity_fig, use_container_width=True, key="severity_chart")
                    else:
                        st.info("No successful attacks to analyze")
                else:
                    st.info("No evaluation history available")
            
            with tab4:
                if results.get('leaderboard'):
                    leaderboard = results['leaderboard']
                    # Handle both dict and object access
                    if isinstance(leaderboard, dict):
                        attackers = leaderboard.get('top_attackers', [])[:10]
                    else:
                        attackers = leaderboard.top_attackers[:10] if hasattr(leaderboard, 'top_attackers') else []
                    
                    fig = create_leaderboard_chart(attackers)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="leaderboard_chart_tab")
                    
                    # Detailed attacker table
                    if attackers:
                        st.markdown("### Detailed Attacker Statistics")
                        attacker_data = []
                        for i, attacker in enumerate(attackers[:10]):
                            if isinstance(attacker, dict):
                                attacker_data.append({
                                    'Rank': i + 1,
                                    'Name': attacker.get('name', 'Unknown'),
                                    'Total Points': attacker.get('total_points', 0),
                                    'Success Rate': f"{attacker.get('success_rate', 0)*100:.1f}%",
                                    'Attempts': attacker.get('total_attempts', 0),
                                    'Successful': attacker.get('successful_exploits', 0)
                                })
                            else:
                                attacker_data.append({
                                    'Rank': i + 1,
                                    'Name': attacker.name,
                                    'Total Points': attacker.total_points,
                                    'Success Rate': f"{attacker.success_rate*100:.1f}%",
                                    'Attempts': attacker.total_attempts,
                                    'Successful': attacker.successful_exploits
                                })
                        
                    df = pd.DataFrame(attacker_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No leaderboard data available")
            
            with tab5:
                st.markdown('<h3><i class="fas fa-globe" style="margin-right: 0.5rem;"></i> 3D Vector Space Visualization</h3>', unsafe_allow_html=True)
                st.markdown("**Explore evaluation results in semantic vector space**")
                st.markdown("Points are colored by attack strategy. Similar responses cluster together.")
                
                if evaluation_history and len(evaluation_history) > 0:
                    # Generate 3D visualization data
                    with st.spinner("Generating 3D vector embeddings... This may take a moment."):
                        try:
                            # Convert evaluation history to list of EvaluationResult objects
                            eval_results = []
                            for i, eval_data in enumerate(evaluation_history):
                                # Handle both dict and object access
                                if isinstance(eval_data, dict):
                                    # Create EvaluationResult from dict
                                    try:
                                        eval_result = EvaluationResult(**eval_data)
                                    except Exception:
                                        # Try creating with minimal fields if full conversion fails
                                        eval_result = EvaluationResult(
                                            id=eval_data.get('id', f'eval_{i}'),
                                            attack_strategy=AttackStrategy(eval_data.get('attack_strategy', {}).get('value', 'roleplay')),
                                            attacker_id=eval_data.get('attacker_id', 'unknown'),
                                            prompt=eval_data.get('prompt', ''),
                                            defender_id=eval_data.get('defender_id', 'unknown'),
                                            defender_model=eval_data.get('defender_model', 'unknown'),
                                            response=eval_data.get('response', ''),
                                            is_jailbroken=eval_data.get('is_jailbroken', False),
                                            severity=eval_data.get('severity', 0)
                                        )
                                else:
                                    eval_result = eval_data
                                eval_results.append(eval_result)
                            
                            # Generate 3D data
                            generator = Vector3DGenerator(method="tsne")
                            output_dir = Path("data/visualizations")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            output_file = output_dir / f"vector3d_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            
                            data_points = generator.generate_3d_data(
                                eval_results,
                                output_path=output_file,
                                normalize=True
                            )
                            
                            if data_points:
                                # Create HTML viewer with embedded data
                                template_path = Path(__file__).parent / "templates" / "vector3d_viewer.html"
                                if template_path.exists():
                                    with open(template_path, 'r') as f:
                                        html_template = f.read()
                                    
                                    # Embed data directly in HTML as a global variable
                                    data_json = json.dumps(data_points, indent=2)
                                    # Insert data as a script tag before the main script
                                    data_script = f'<script>const DATA_EMBEDDED = {data_json};</script>\n'
                                    html_content = html_template.replace(
                                        '<script>',
                                        data_script + '<script>',
                                        1  # Replace only the first occurrence
                                    )
                                    
                                    # Save standalone HTML file with embedded data
                                    html_file = output_dir / f"vector3d_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                                    with open(html_file, 'w') as f:
                                        f.write(html_content)
                                    
                                    st.markdown(f'<div style="padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid rgba(34, 197, 94, 0.8); border-radius: 6px; margin: 0.5rem 0; color: #86efac;"><i class="fas fa-check-circle" style="margin-right: 0.5rem; color: #22c55e;"></i> Generated {len(data_points)} data points for 3D visualization</div>', unsafe_allow_html=True)
                                    
                                    # Display the 3D visualization using Streamlit's HTML component
                                    try:
                                        import streamlit.components.v1 as components
                                        components.html(html_content, height=700, scrolling=False)
                                    except Exception as e:
                                        st.warning(f"Could not embed visualization: {e}")
                                        st.info("Use the download button below to view the HTML file in your browser.")
                                    
                                    # Display instructions
                                    with st.expander("How to Use 3D Threat Radar", expanded=False):
                                        st.markdown("""
                                        <div style="padding: 1rem; background: rgba(15, 23, 42, 0.5); border-radius: 12px; border-left: 4px solid #06b6d4;">
                                        <h4 style="color: #06b6d4; margin-top: 0;">3D Vector Space Controls</h4>
                                        <ul style="color: #e0e7ff; line-height: 1.8;">
                                        <li><strong>Orbit:</strong> Left-click and drag to rotate the view</li>
                                        <li><strong>Zoom:</strong> Mouse wheel to zoom in/out</li>
                                        <li><strong>Pan:</strong> Shift + left-click and drag to pan</li>
                                        <li><strong>Hover:</strong> Move mouse over points to see detailed information</li>
                                        <li><strong>Color Modes:</strong> Switch between Strategy, Status, and Severity views using the dropdown</li>
                                        </ul>
                                        
                                        <h4 style="color: #06b6d4; margin-top: 1.5rem;">Visualization Legend</h4>
                                        <ul style="color: #e0e7ff; line-height: 1.8;">
                                        <li>Each attack strategy has a unique color in Strategy mode</li>
                                        <li>Points cluster together based on semantic similarity of responses</li>
                                        <li>In Status mode: Jailbroken points are red, blocked points are green</li>
                                        <li>In Severity mode: Color intensity indicates threat severity level</li>
                                        </ul>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Download buttons
                                    col_d1, col_d2 = st.columns(2)
                                    with col_d1:
                                        with open(html_file, 'rb') as f:
                                            st.download_button(
                                                label="Download 3D Viewer (HTML)",
                                                data=f.read(),
                                                file_name=html_file.name,
                                                mime="text/html",
                                                use_container_width=True
                                            )
                                    with col_d2:
                                        with open(output_file, 'r') as f:
                                            st.download_button(
                                                label="Download 3D Data (JSON)",
                                                data=f.read(),
                                                file_name=output_file.name,
                                                mime="application/json",
                                                use_container_width=True
                                            )
                                    
                                    # Show preview statistics
                                    st.markdown("---")
                                    st.markdown("### Threat Radar Statistics")
                                    
                                    jailbroken_count = sum(1 for p in data_points if p.get('is_jailbroken', False))
                                    blocked_count = len(data_points) - jailbroken_count
                                    strategies = len(set(p.get('strategy_index', 0) for p in data_points))
                                    exploit_rate = (jailbroken_count / len(data_points) * 100) if data_points else 0
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric(
                                            "Total Evaluations",
                                            len(data_points),
                                            help="Total number of evaluation points in the Threat Radar"
                                        )
                                    with col2:
                                        st.metric(
                                            "Successful Exploits",
                                            jailbroken_count,
                                            delta=f"{exploit_rate:.1f}% exploit rate",
                                            delta_color="inverse",
                                            help="Number of successful jailbreak attempts"
                                        )
                                    with col3:
                                        st.metric(
                                            "Blocked Attempts",
                                            blocked_count,
                                            delta=f"{(100-exploit_rate):.1f}% block rate",
                                            help="Number of blocked attack attempts"
                                        )
                                    with col4:
                                        st.metric(
                                            "Attack Strategies",
                                            strategies,
                                            help="Number of unique attack strategies tested"
                                        )
                                    
                                else:
                                    st.error(f"HTML template not found at {template_path}")
                            else:
                                st.warning("No data points generated. Check evaluation results.")
                                
                        except Exception as e:
                            st.error(f"Error generating 3D visualization: {e}")
                            import traceback
                            with st.expander("Error Details", expanded=False):
                                st.code(traceback.format_exc())
                else:
                    st.info("No evaluation history available for 3D visualization. Run an evaluation first.")
            
            with tab6:
                # Defense Improvement Dashboard
                st.markdown("### 🛡️ Continuous Defense Improvement")
                st.markdown("*Real-time defense metrics, pattern analysis, and improvement recommendations*")
                
                if st.session_state.arena and st.session_state.arena.pattern_database:
                    db = st.session_state.arena.pattern_database
                    
                    # Pattern Database Statistics
                    st.markdown("#### 📊 Pattern Database Statistics")
                    analysis = db.analyze_patterns()
                    
                    col_db1, col_db2, col_db3, col_db4 = st.columns(4)
                    with col_db1:
                        st.metric("Total Patterns", analysis.get('total_patterns', 0))
                    with col_db2:
                        st.metric("High Severity", analysis.get('high_severity_count', 0))
                    with col_db3:
                        st.metric("Critical Patterns", analysis.get('critical_patterns', 0))
                    with col_db4:
                        st.metric("Avg Attack Cost", f"{analysis.get('average_attack_cost', 0):.2f}")
                    
                    # Strategy Distribution
                    if analysis.get('strategies'):
                        st.markdown("#### 🎯 Strategy Distribution")
                        strategy_data = analysis['strategies']
                        strategy_df = pd.DataFrame(list(strategy_data.items()), columns=['Strategy', 'Count'])
                        strategy_fig = px.bar(
                            strategy_df,
                            x='Strategy',
                            y='Count',
                            title='Exploit Patterns by Strategy',
                            color='Count',
                            color_continuous_scale='Reds'
                        )
                        strategy_fig.update_layout(
                            paper_bgcolor='#1a1a1a',
                            plot_bgcolor='#1a1a1a',
                            font={'color': '#e0e7ff'},
                            title_font={'color': '#ffffff'}
                        )
                        st.plotly_chart(strategy_fig, use_container_width=True)
                    
                    # Vulnerability Analysis
                    st.markdown("#### 🔍 Vulnerability Analysis")
                    try:
                        from src.defense.vulnerability_analyzer import VulnerabilityAnalyzer
                        from src.defense.patch_generator import DefensePatchGenerator
                        
                        analyzer = VulnerabilityAnalyzer(pattern_database=db)
                        vulnerabilities = analyzer.analyze_vulnerabilities()
                        
                        if vulnerabilities:
                            vuln_summary = analyzer.get_vulnerability_summary()
                            
                            st.markdown(f"**Total Vulnerabilities Detected:** {vuln_summary.get('total_vulnerabilities', 0)}")
                            
                            # Top vulnerabilities
                            st.markdown("##### 🚨 Top Priority Vulnerabilities")
                            for i, vuln in enumerate(vulnerabilities[:5], 1):
                                severity_color = {
                                    SeverityLevel.CRITICAL: "#dc3545",
                                    SeverityLevel.HIGH: "#f59e0b",
                                    SeverityLevel.MODERATE: "#fbbf24"
                                }.get(vuln.severity, "#6b7280")
                                
                                st.markdown(f"""
                                <div style="padding: 1rem; background: rgba(15, 23, 42, 0.7); border-left: 4px solid {severity_color}; border-radius: 8px; margin: 0.5rem 0;">
                                    <strong>#{i} - {vuln.description}</strong><br>
                                    <small>Priority: {vuln.priority} | Severity: {vuln.severity.value}/5 | Patterns: {vuln.pattern_count}</small><br>
                                    <small><strong>Recommendations:</strong></small><br>
                                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                                        {''.join([f'<li>{rec}</li>' for rec in vuln.recommendations[:3]])}
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Generate patches
                            if st.button("🔧 Generate Defense Patches", type="primary"):
                                patch_generator = DefensePatchGenerator(
                                    vulnerability_analyzer=analyzer
                                )
                                patches = patch_generator.generate_patches(vulnerabilities)
                                
                                st.success(f"Generated {len(patches)} defense patches!")
                                
                                # Show patches
                                for patch in patches[:3]:
                                    st.markdown(f"""
                                    <div style="padding: 1rem; background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; border-radius: 8px; margin: 0.5rem 0;">
                                        <strong>Patch: {patch.description}</strong><br>
                                        <small>Type: {patch.patch_type} | Priority: {patch.priority}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No vulnerabilities detected. Defense appears strong!")
                    except ImportError:
                        st.warning("Vulnerability analyzer not available")
                    
                    # Threat Landscape
                    st.markdown("#### 🌐 Threat Landscape")
                    try:
                        from src.defense.pattern_recognizer import ThreatPatternRecognizer
                        
                        recognizer = ThreatPatternRecognizer(pattern_database=db)
                        landscape = recognizer.get_overall_threat_landscape()
                        
                        threat_level = landscape.get('threat_level', 'unknown')
                        threat_color = {
                            'critical': '#dc3545',
                            'high': '#f59e0b',
                            'moderate': '#fbbf24',
                            'low': '#22c55e'
                        }.get(threat_level, '#6b7280')
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; background: rgba(15, 23, 42, 0.7); border-left: 4px solid {threat_color}; border-radius: 8px;">
                            <strong>Overall Threat Level: {threat_level.upper()}</strong><br>
                            <small>Total Patterns: {landscape.get('total_exploit_patterns', 0)}</small><br>
                            <small>Critical Vulnerabilities: {landscape.get('critical_vulnerabilities', 0)}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendations
                        if landscape.get('recommendations'):
                            st.markdown("##### 💡 Recommendations")
                            for rec in landscape['recommendations'][:5]:
                                st.markdown(f"- {rec}")
                    except ImportError:
                        st.warning("Threat pattern recognizer not available")
                    
                    # Defense Rules
                    st.markdown("#### ⚙️ Active Defense Rules")
                    try:
                        from src.defense.adaptive_engine import AdaptiveDefenseEngine
                        
                        engine = AdaptiveDefenseEngine(pattern_database=db)
                        rules = engine.get_active_rules()
                        
                        if rules:
                            rule_stats = engine.get_rule_statistics()
                            st.metric("Total Rules", rule_stats.get('total_rules', 0))
                            st.metric("High Priority", rule_stats.get('high_priority_rules', 0))
                            
                            # Show top rules
                            for rule in rules[:5]:
                                st.markdown(f"""
                                <div style="padding: 0.75rem; background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; border-radius: 6px; margin: 0.5rem 0;">
                                    <strong>{rule.description}</strong><br>
                                    <small>Type: {rule.rule_type} | Priority: {rule.priority}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No defense rules generated yet. Run more evaluations to generate rules.")
                    except ImportError:
                        st.warning("Adaptive defense engine not available")
                    
                else:
                    st.info("Pattern database not enabled. Enable it in arena initialization to see defense metrics.")
        
        # Battle logs with sample responses (collapsible)
        with logs_container:
            with st.expander("Evaluation Log", expanded=False):
                # Add expander to show sample responses for debugging
                if evaluation_history:
                    with st.expander("View Sample Responses (for debugging)", expanded=False):
                        sample_count = min(5, len(evaluation_history))
                        for i, eval_result in enumerate(evaluation_history[:sample_count]):
                            # Handle both dict and object access
                            if isinstance(eval_result, dict):
                                prompt = eval_result.get('prompt', '')[:150]
                                response = eval_result.get('response', '')[:300]
                                strategy = eval_result.get('attack_strategy', {})
                                strategy_name = strategy.get('value', 'unknown') if isinstance(strategy, dict) else str(strategy)
                                is_jailbroken = eval_result.get('is_jailbroken', False)
                            else:
                                prompt = str(eval_result.prompt)[:150] if eval_result.prompt else ''
                                response = str(eval_result.response)[:300] if eval_result.response else ''
                                strategy_name = str(eval_result.attack_strategy.value) if hasattr(eval_result.attack_strategy, 'value') else str(eval_result.attack_strategy)
                                is_jailbroken = eval_result.is_jailbroken
                            
                            st.markdown(f"**Sample {i+1} - Strategy: {strategy_name}**")
                            st.markdown(f"**Prompt:** {prompt}...")
                            st.markdown(f"**Response:** {response}...")
                            jailbroken_icon = '<i class="fas fa-check-circle icon-success" style="margin-right: 0.5rem;"></i> Yes' if is_jailbroken else '<i class="fas fa-times-circle icon-error" style="margin-right: 0.5rem;"></i> No'
                            st.markdown(f'**Jailbroken:** {jailbroken_icon}', unsafe_allow_html=True)
                            st.markdown("---")
                
                for log_entry in st.session_state.logs[-20:]:
                    round_label = log_entry.get('round', 'Unknown')
                    if log_entry['status'] == "JAILBROKEN":
                        st.markdown(f"""
                        <div class="success-log">
                            <strong>{round_label}: {log_entry['status']}</strong><br>
                            Strategy: {log_entry['strategy']} | Severity: {log_entry['severity']}/5<br>
                            <small>Prompt: {log_entry['prompt']}...</small><br>
                            <small>Response: {log_entry.get('response', '')[:100]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="fail-log">
                            <strong>{round_label}: {log_entry['status']}</strong><br>
                            Strategy: {log_entry['strategy']}<br>
                            <small>Prompt: {log_entry['prompt']}...</small><br>
                            <small>Response: {log_entry.get('response', '')[:100]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Export results
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"arena_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:
        # Professional welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>Welcome to Jailbreak Genome Scanner</h2>
            <p style="font-size: 1.1rem; color: #666666; margin-top: 1rem;">
                Active Defense Infrastructure - Automated Red-Teaming & Threat Radar System<br>
                Configure your settings in the sidebar and click START EVALUATION to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase - professional cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-box">
                <h3>Defender</h3>
                <p>Test any LLM model</p>
                <p>OpenAI, Anthropic, or Lambda Cloud</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-box">
                <h3>Attackers</h3>
                <p>Multiple strategies</p>
                <p>Real-time visualization</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-box">
                <h3>Lambda Scraper</h3>
                <p>Recent attack data</p>
                <p>Web scraping & analysis</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
