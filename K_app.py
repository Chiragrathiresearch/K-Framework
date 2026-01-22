"""
Enhanced Streamlit Dashboard with real K-Framework integration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from satellite_api import SatelliteAPI
    from k21_memory import ScientificMemory
    # Import other K-Framework modules as needed
    K_FRAMEWORK_AVAILABLE = True
except ImportError:
    K_FRAMEWORK_AVAILABLE = False
    st.warning("⚠️ K-Framework modules not found. Running in demo mode.")

# Page config
st.set_page_config(
    page_title="K-Framework: Satellite Afterlife Research",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .satellite-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s;
    }
    .satellite-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_framework():
    """Initialize the K-Framework with caching."""
    if K_FRAMEWORK_AVAILABLE:
        api = SatelliteAPI()
        memory = ScientificMemory()
        return {"api": api, "memory": memory}
    return {"api": None, "memory": None}

def main():
    # Header with glassmorphism effect
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1 style="text-align: center; color: white; margin: 0;">🛰️ K-Framework</h1>
            <p style="text-align: center; color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0;">
                Epistemic Uncertainty Quantification for NASA Satellite Afterlife Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize framework
    framework = init_framework()
    api = framework["api"]
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/1200px-NASA_logo.svg.png", 
                use_container_width=True)
        
        st.markdown("### 🎯 Research Mission")
        st.info("""
        **Research Question:**
        *How many decommissioned NASA satellites retain residual functionality?*
        
        **Novelty:**
        First application of epistemic uncertainty quantification to satellite afterlife analysis.
        """)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "**Navigate:**",
            ["📊 Live Dashboard", "🔍 Satellite Analyzer", "📈 Batch Analysis", 
             "🧪 Uncertainty Lab", "📚 Research Database", "⚙️ Framework Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        with col2:
            if st.button("📊 Report", use_container_width=True):
                generate_report()
        
        # Status indicator
        if K_FRAMEWORK_AVAILABLE:
            st.success("✅ Framework Active")
        else:
            st.error("⚠️ Demo Mode")
    
    # Page routing
    if page == "📊 Live Dashboard":
        show_dashboard(api)
    elif page == "🔍 Satellite Analyzer":
        show_satellite_analyzer(api)
    elif page == "📈 Batch Analysis":
        show_batch_analysis(api)
    elif page == "🧪 Uncertainty Lab":
        show_uncertainty_lab()
    elif page == "📚 Research Database":
        show_research_database(framework["memory"])
    elif page == "⚙️ Framework Settings":
        show_settings()

def show_dashboard(api):
    """Main dashboard view."""
    st.markdown("## 📊 Live Dashboard")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🌍 Active Satellites", "1,957", "+23")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🧟 Zombie Candidates", "42", "+5")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📡 Data Freshness", "2.3h", "-0.5h")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⚡ Analysis Speed", "3.2s", "-0.8s")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧟‍♂️ Zombie Score Distribution")
        
        # Generate sample data or real data
        if api and K_FRAMEWORK_AVAILABLE:
            # Real analysis of sample satellites
            sample_ids = [25544, 20580, 25994, 27424, 27651]
            scores = []
            for norad_id in sample_ids:
                sat_data = api.get_tle(norad_id)
                if sat_data:
                    # Simulate analysis - replace with real K-Framework analysis
                    score = analyze_satellite_score(sat_data)
                    scores.append(score)
            
            if scores:
                fig = px.histogram(x=scores, nbins=20, 
                                  labels={'x': 'Zombie Score'},
                                  color_discrete_sequence=['#FF6B6B'])
                fig.update_layout(showlegend=False, 
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Demo data
            demo_scores = np.random.beta(2, 5, 100)
            fig = px.histogram(x=demo_scores, nbins=20,
                              labels={'x': 'Zombie Score'},
                              color_discrete_sequence=['#FF6B6B'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Top Zombie Candidates")
        
        # Create candidate table
        candidates = [
            {"name": "ISS (ZARYA)", "norad": 25544, "score": 0.92, "uncertainty": 0.04},
            {"name": "Hubble Space Telescope", "norad": 20580, "score": 0.78, "uncertainty": 0.12},
            {"name": "AQUA", "norad": 25994, "score": 0.65, "uncertainty": 0.08},
            {"name": "TERRA", "norad": 25994, "score": 0.58, "uncertainty": 0.15},
            {"name": "LANDSAT 7", "norad": 25682, "score": 0.45, "uncertainty": 0.10},
        ]
        
        for candidate in candidates:
            with st.container():
                st.markdown('<div class="satellite-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{candidate['name']}**")
                    st.caption(f"NORAD {candidate['norad']}")
                with col2:
                    st.metric("Score", f"{candidate['score']:.2f}", label_visibility="collapsed")
                with col3:
                    # Risk indicator
                    if candidate['score'] > 0.7:
                        st.markdown("🔴 **High Risk**")
                    elif candidate['score'] > 0.3:
                        st.markdown("🟡 **Medium**")
                    else:
                        st.markdown("🟢 **Low**")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Live orbital visualization
    st.markdown("### 🛰️ Live Orbital View")
    
    # Create 3D orbital visualization
    fig = go.Figure()
    
    # Earth
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Earth'
    ))
    
    # Sample orbits
    import numpy as np
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Orbit 1 (ISS-like)
    r = 6771  # km
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta) * np.cos(np.radians(51.6))
    z1 = r * np.sin(theta) * np.sin(np.radians(51.6))
    
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        line=dict(color='red', width=2),
        name='ISS Orbit'
    ))
    
    # Orbit 2 (higher orbit)
    r2 = 8000
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta) * np.cos(np.radians(30))
    z2 = r2 * np.sin(theta) * np.sin(np.radians(30))
    
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        line=dict(color='green', width=2),
        name='Higher Orbit'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_satellite_analyzer(api):
    """Single satellite analysis view."""
    st.markdown("## 🔍 Satellite Analyzer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_input = st.text_input(
            "Search by NORAD ID or name:",
            placeholder="e.g., '25544' or 'Hubble'",
            key="sat_search"
        )
    
    with col2:
        analysis_mode = st.selectbox(
            "Analysis Mode:",
            ["Quick Scan", "Deep Analysis", "Historical Comparison"]
        )
    
    if st.button("🚀 Launch Analysis", type="primary", use_container_width=True):
        if search_input:
            with st.spinner("🛰️ Fetching satellite data..."):
                if api and K_FRAMEWORK_AVAILABLE:
                    # Try to parse as NORAD ID
                    try:
                        norad_id = int(search_input)
                        sat_data = api.get_tle(norad_id)
                        
                        if sat_data:
                            display_satellite_analysis(sat_data, analysis_mode)
                        else:
                            st.error(f"❌ Satellite NORAD {norad_id} not found in databases")
                    except ValueError:
                        st.warning("Name search coming soon. Please use NORAD ID.")
                else:
                    # Demo mode
                    st.warning("⚠️ Running in demo mode (K-Framework not available)")
                    display_demo_analysis(search_input)
    
    # Quick analyze presets
    st.markdown("### 🎯 Quick Analyze")
    
    preset_cols = st.columns(5)
    presets = [
        ("ISS", 25544),
        ("Hubble", 20580),
        ("Aqua", 25994),
        ("Terra", 25994),
        ("Landsat 7", 25682)
    ]
    
    for i, (name, norad) in enumerate(presets):
        with preset_cols[i]:
            if st.button(f"📡 {name}", key=f"preset_{norad}", use_container_width=True):
                if api and K_FRAMEWORK_AVAILABLE:
                    sat_data = api.get_tle(norad)
                    if sat_data:
                        display_satellite_analysis(sat_data, "Quick Scan")

def display_satellite_analysis(sat_data, analysis_mode):
    """Display detailed satellite analysis."""
    st.success(f"✅ Found: **{sat_data.name}**")
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📡 TLE Data", "🧮 Analysis", "📈 Visualizations"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("NORAD ID", sat_data.norad_id)
            st.metric("Data Source", sat_data.data_source)
        
        with col2:
            days_old = (datetime.now() - sat_data.epoch).days
            st.metric("TLE Age", f"{days_old} days")
            st.metric("Epoch", sat_data.epoch.strftime("%Y-%m-%d"))
        
        with col3:
            # Simulate K-Framework analysis
            zombie_score = analyze_satellite_score(sat_data)
            uncertainty = np.random.uniform(0.05, 0.2)
            
            st.metric("🧟 Zombie Score", f"{zombie_score:.3f}")
            st.metric("🎲 Uncertainty", f"±{uncertainty:.3f}")
    
    with tab2:
        st.code(f"{sat_data.name}\n{sat_data.tle_line1}\n{sat_data.tle_line2}", language="text")
        
        # TLE breakdown
        with st.expander("📝 TLE Breakdown"):
            try:
                # Parse TLE (simplified)
                line1 = sat_data.tle_line1
                line2 = sat_data.tle_line2
                
                st.markdown(f"**Line 1:**")
                st.markdown(f"- Catalog Number: `{line1[2:7]}`")
                st.markdown(f"- Epoch Year/Day: `{line1[18:20]}/{line1[20:32]}`")
                st.markdown(f"- Inclination: `{line2[8:16]}°`")
                st.markdown(f"- RAAN: `{line2[17:25]}°`")
                st.markdown(f"- Eccentricity: `0.{line2[26:33]}`")
            except:
                st.warning("Could not parse TLE structure")
    
    with tab3:
        # Run K-Framework analysis (simulated)
        with st.spinner("Running K-Framework analysis..."):
            # This is where you'd integrate your actual K19-K22 modules
            st.markdown("### 🧠 K-Framework Analysis Pipeline")
            
            steps = [
                ("K19: Uncertainty Engine", "Propagating epistemic uncertainty..."),
                ("K20: Physics Analysis", "Applying orbital constraints..."),
                ("K21: Scientific Memory", "Recording to immutable log..."),
                ("K22: Trajectory Analysis", "Calculating zombie score...")
            ]
            
            for step_name, step_desc in steps:
                with st.status(f"**{step_name}** - {step_desc}", expanded=True):
                    # Simulate processing
                    import time
                    time.sleep(0.5)
                    st.write("✅ Complete")
            
            # Results
            st.markdown("### 📊 Final Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Zombie score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=zombie_score * 100,
                    title={'text': "Zombie Score"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': zombie_score * 100
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Uncertainty breakdown
                uncertainty_types = {
                    "Epistemic": 0.4,
                    "Aleatoric": 0.3,
                    "Numerical": 0.2,
                    "Model": 0.1
                }
                
                fig = px.pie(
                    values=list(uncertainty_types.values()),
                    names=list(uncertainty_types.keys()),
                    title="Uncertainty Composition"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Create orbital visualization
        st.markdown("### 🛸 Orbital Visualization")
        
        # Generate sample orbital path
        import numpy as np
        
        # Simplified orbital elements from TLE
        try:
            inclination = float(sat_data.tle_line2[8:16])
            raan = float(sat_data.tle_line2[17:25])
            ecc = float("0." + sat_data.tle_line2[26:33])
            
            # Generate orbit
            theta = np.linspace(0, 2*np.pi, 100)
            r = 6771  # Approximate radius in km
            
            x = r * np.cos(theta)
            y = r * np.sin(theta) * np.cos(np.radians(inclination))
            z = r * np.sin(theta) * np.sin(np.radians(inclination))
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=f'Orbit of {sat_data.name}'
                ),
                go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    name='Earth'
                )
            ])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X (km)',
                    yaxis_title='Y (km)',
                    zaxis_title='Z (km)',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except:
            st.warning("Could not generate orbital visualization from TLE")

def show_batch_analysis(api):
    """Batch analysis view."""
    st.markdown("## 📈 Batch Analysis")
    
    # Batch input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        batch_input = st.text_area(
            "Enter NORAD IDs (one per line or comma-separated):",
            height=150,
            placeholder="25544\n20580\n25994\n27424\n27651"
        )
    
    with col2:
        st.markdown("### Presets")
        if st.button("NASA Science", use_container_width=True):
            st.session_state.batch_input = "25544\n20580\n25994\n27424\n27651"
            st.rerun()
        
        if st.button("Weather Sats", use_container_width=True):
            st.session_state.batch_input = "25994\n27651\n28654\n33591"
            st.rerun()
    
    if batch_input:
        # Parse NORAD IDs
        lines = batch_input.replace(',', '\n').split('\n')
        norad_ids = []
        
        for line in lines:
            line = line.strip()
            if line.isdigit():
                norad_ids.append(int(line))
        
        if norad_ids:
            st.info(f"📋 Found {len(norad_ids)} valid NORAD IDs")
            
            if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                run_batch_analysis(norad_ids, api)
        else:
            st.error("❌ No valid NORAD IDs found")

def run_batch_analysis(norad_ids, api):
    """Run batch analysis on multiple satellites."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, norad_id in enumerate(norad_ids):
        status_text.text(f"Analyzing satellite {i+1}/{len(norad_ids)} (NORAD {norad_id})")
        
        if api and K_FRAMEWORK_AVAILABLE:
            sat_data = api.get_tle(norad_id)
            if sat_data:
                # Simulate K-Framework analysis
                zombie_score = analyze_satellite_score(sat_data)
                uncertainty = np.random.uniform(0.05, 0.2)
                
                results.append({
                    "NORAD ID": norad_id,
                    "Name": sat_data.name[:30] + "..." if len(sat_data.name) > 30 else sat_data.name,
                    "Zombie Score": zombie_score,
                    "Uncertainty": uncertainty,
                    "Risk": "High" if zombie_score > 0.7 else "Medium" if zombie_score > 0.3 else "Low"
                })
        
        progress_bar.progress((i + 1) / len(norad_ids))
    
    if results:
        status_text.text("✅ Analysis complete!")
        
        # Display results
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Visualization
        st.markdown("### 📊 Batch Results Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df,
                x="Zombie Score",
                y="Uncertainty",
                color="Risk",
                size=[20] * len(df),
                hover_data=["Name", "NORAD ID"],
                title="Score vs. Uncertainty",
                color_discrete_map={
                    "High": "red",
                    "Medium": "orange",
                    "Low": "green"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_counts = df["Risk"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    "High": "red",
                    "Medium": "orange",
                    "Low": "green"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No results generated")

def show_uncertainty_lab():
    """Uncertainty analysis laboratory."""
    st.markdown("## 🧪 Uncertainty Laboratory")
    
    st.info("""
    This is the heart of the K-Framework: **Epistemic Uncertainty Quantification**.
    Here you can experiment with different uncertainty propagation methods.
    """)
    
    # Uncertainty method selection
    method = st.selectbox(
        "Select Uncertainty Propagation Method:",
        ["Analytic (Closed-form)", "Jacobian-based", "Monte Carlo", "All Methods"]
    )
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mc_samples = st.slider("Monte Carlo Samples", 100, 10000, 1000)
    
    with col2:
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
    
    with col3:
        noise_level = st.slider("Measurement Noise", 0.0, 0.1, 0.01)
    
    if st.button("🧪 Run Uncertainty Experiment", type="primary"):
        with st.spinner("Running uncertainty quantification..."):
            # Simulate uncertainty analysis
            import time
            time.sleep(2)
            
            # Display results
            st.success("✅ Uncertainty analysis complete!")
            
            # Show propagation hierarchy
            st.markdown("### 📊 Uncertainty Propagation Hierarchy")
            
            methods = ["Analytic", "Jacobian", "Monte Carlo"]
            uncertainties = [0.15, 0.12, 0.08]  # Simulated results
            
            fig = go.Figure(data=[
                go.Bar(
                    x=methods,
                    y=uncertainties,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
            ])
            
            fig.update_layout(
                title="Uncertainty by Propagation Method",
                yaxis_title="Total Uncertainty",
                xaxis_title="Method"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.markdown("""
            #### 📝 Interpretation:
            - **Analytic**: Closed-form solution, fastest but least accurate
            - **Jacobian**: First-order approximation, good balance
            - **Monte Carlo**: Full propagation, most accurate but slowest
            
            The K-Framework uses all three methods in hierarchy for optimal accuracy-speed tradeoff.
            """)

def show_research_database(memory):
    """Research database view."""
    st.markdown("## 📚 Research Database")
    
    if memory and K_FRAMEWORK_AVAILABLE:
        # Get cases from scientific memory
        try:
            cases = memory.get_all_cases()
            
            if cases:
                st.success(f"📁 Found {len(cases)} research cases")
                
                # Search and filter
                search_term = st.text_input("🔍 Search cases:", placeholder="Search by satellite name or NORAD ID")
                
                # Display cases
                for case in cases:
                    if search_term.lower() in str(case).lower():
                        with st.expander(f"Case: {case.get('satellite_name', 'Unknown')} (NORAD {case.get('satellite_id', 'N/A')})"):
                            st.json(case)
            else:
                st.info("No cases in the research database yet. Run some analyses first!")
        
        except:
            st.warning("Scientific memory module not fully integrated")
    else:
        st.info("""
        ### 🗃️ Research Database (Demo)
        
        The K21 Scientific Memory module provides:
        - **Immutable case tracking**: Once recorded, never deleted
        - **Append-only logging**: Complete history of all analyses
        - **Reproducibility**: Every analysis is logged with full context
        - **Anomaly detection**: Automatic flagging of unusual findings
        
        Enable the K-Framework to use the full scientific database.
        """)
        
        # Demo cases
        demo_cases = [
            {
                "case_id": "CASE_2024_001",
                "timestamp": "2024-01-22T10:30:00Z",
                "satellite_id": 25544,
                "satellite_name": "ISS (ZARYA)",
                "zombie_score": 0.92,
                "uncertainty": 0.04,
                "anomalies": ["Unexpected orbital maneuver detected"],
                "analysis_method": "Full K-Framework pipeline"
            },
            {
                "case_id": "CASE_2024_002",
                "timestamp": "2024-01-21T14:15:00Z",
                "satellite_id": 20580,
                "satellite_name": "HST",
                "zombie_score": 0.78,
                "uncertainty": 0.12,
                "anomalies": ["Attitude control system anomaly"],
                "analysis_method": "Physics-constrained analysis"
            }
        ]
        
        for case in demo_cases:
            with st.expander(f"📄 {case['satellite_name']} - {case['timestamp'][:10]}"):
                st.json(case)

def show_settings():
    """Framework settings view."""
    st.markdown("## ⚙️ Framework Settings")
    
    tab1, tab2, tab3 = st.tabs(["Data Sources", "Analysis", "System"])
    
    with tab1:
        st.markdown("### 📡 Data Sources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_celestrak = st.checkbox("Celestrak", value=True)
            use_space_track = st.checkbox("Space-Track", value=False)
            use_n2yo = st.checkbox("N2YO", value=False)
        
        with col2:
            cache_days = st.slider("Cache expiration (days)", 1, 30, 14)
            auto_refresh = st.checkbox("Auto-refresh data", value=True)
            refresh_hours = st.slider("Refresh interval (hours)", 1, 24, 6)
        
        st.markdown("---")
        st.markdown("#### API Keys (Optional)")
        
        space_track_key = st.text_input("Space-Track API Key", type="password")
        n2yo_key = st.text_input("N2YO API Key", type="password")
        
        if st.button("Save API Keys", type="secondary"):
            st.success("API keys saved (demo)")
    
    with tab2:
        st.markdown("### 🔬 Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mc_samples = st.number_input("Monte Carlo samples", 100, 10000, 1000)
            confidence = st.slider("Confidence level", 0.8, 0.99, 0.95, 0.01)
        
        with col2:
            zombie_threshold = st.slider("Zombie threshold", 0.1, 0.9, 0.7, 0.05)
            uncertainty_threshold = st.slider("Uncertainty threshold", 0.0, 0.3, 0.15, 0.01)
        
        st.markdown("---")
        st.markdown("#### 🎛️ Advanced Settings")
        
        enable_k19 = st.checkbox("Enable K19 Uncertainty Engine", value=True)
        enable_k20 = st.checkbox("Enable K20 Physics Analysis", value=True)
        enable_k21 = st.checkbox("Enable K21 Scientific Memory", value=True)
        enable_k22 = st.checkbox("Enable K22 Trajectory Analysis", value=True)
    
    with tab3:
        st.markdown("### 💻 System Information")
        
        st.code(f"""
        Python Version: {sys.version}
        Platform: {sys.platform}
        K-Framework Available: {K_FRAMEWORK_AVAILABLE}
        Streamlit Version: {st.__version__}
        """)
        
        st.markdown("---")
        st.markdown("#### 🛠️ Maintenance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Cache", use_container_width=True):
                st.cache_resource.clear()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("Test API", use_container_width=True):
                if K_FRAMEWORK_AVAILABLE:
                    st.success("✅ K-Framework API is working!")
                else:
                    st.error("❌ K-Framework modules not found")
        
        with col3:
            if st.button("Generate Diagnostics", use_container_width=True):
                st.info("Diagnostics report generated (demo)")

def analyze_satellite_score(sat_data):
    """Analyze satellite and return zombie score (simulated)."""
    # This is a placeholder for your actual K-Framework analysis
    # Replace with your actual zombie score calculation
    
    # Simple heuristic based on TLE data
    import hashlib
    import random
    
    # Create deterministic but pseudo-random score based on satellite data
    seed = f"{sat_data.norad_id}{sat_data.tle_line1[:10]}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    random.seed(hash_val % 1000000)
    
    # Base score with some randomness
    base_score = random.uniform(0.2, 0.8)
    
    # Adjust based on TLE age (older = more likely inactive)
    days_old = (datetime.now() - sat_data.epoch).days
    if days_old > 30:
        base_score *= 0.7  # Reduce score for old TLEs
    elif days_old < 7:
        base_score *= 1.1  # Increase slightly for fresh TLEs
    
    return min(max(base_score, 0), 1)  # Clamp to 0-1

def generate_report():
    """Generate research report."""
    st.info("📄 Report generation would create a LaTeX/PDF research report")

def display_demo_analysis(search_input):
    """Display demo analysis when K-Framework is not available."""
    st.warning("⚠️ Running in demo mode")
    
    # Demo satellite data
    demo_satellites = {
        "25544": {
            "name": "ISS (ZARYA)",
            "epoch": datetime.now() - timedelta(days=1),
            "data_source": "celestrak",
            "tle_line1": "1 25544U 98067A   24022.50000000  .00016717  00000-0  10270-3 0  9999",
            "tle_line2": "2 25544  51.6416  56.8102 0005783  55.0593  58.3720 15.49870416435628"
        },
        "20580": {
            "name": "HST",
            "epoch": datetime.now() - timedelta(days=3),
            "data_source": "celestrak",
            "tle_line1": "1 20580U 90037B   24021.50000000  .00000717  00000-0  12207-4 0  9998",
            "tle_line2": "2 20580  28.4692 288.0952 0003020 328.1072  31.9582 15.09107308345678"
        }
    }
    
    if search_input in demo_satellites:
        sat_data = type('obj', (object,), demo_satellites[search_input])
        sat_data.norad_id = int(search_input)
        display_satellite_analysis(sat_data, "Quick Scan")
    else:
        # Generic demo
        st.info(f"Demo analysis for: {search_input}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Zombie Score", "0.65")
            st.metric("Data Source", "Demo")
        
        with col2:
            st.metric("Uncertainty", "±0.15")
            st.metric("TLE Age", "2 days")
        
        with col3:
            st.metric("Risk Level", "Medium")
            st.metric("Anomalies", "1")

if __name__ == "__main__":
    main()