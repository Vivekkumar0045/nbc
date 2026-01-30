import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import os
import google.generativeai as genai
from datetime import datetime

# API Config
API_URL = "https://vivek45537-nbc.hf.space"

# Page Config
st.set_page_config(
    page_title="NBC Nexus | AI Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models & Data
# Models are now served via API

@st.cache_data
def load_data():
    # Use absolute path relative to this file to ensure it works regardless of where the script is run from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bearings = pd.read_csv(os.path.join(base_dir, "bearings_data.csv"))
    dealers = pd.read_csv(os.path.join(base_dir, "dealer_network.csv"))
    return bearings, dealers

try:
    bearings_df, dealers_df = load_data()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Helper Functions
def predict_rul(inputs):
    # inputs: [Hours, RPM, Temp, Vib, Lub, Load_Factor]
    payload = {
        "Operating_Hours": inputs[0],
        "RPM": inputs[1],
        "Temperature_C": inputs[2],
        "Vibration_mm_s": inputs[3],
        "Lubrication_Level_Pct": inputs[4],
        "Load_Factor": inputs[5] if len(inputs) > 5 else 1.0 # Handle potential missing load factor
    }
    try:
        response = requests.post(f"{API_URL}/predict/rul", json=payload)
        if response.status_code == 200:
            return response.json()['predicted_rul_days']
        else:
            st.error(f"API Error: {response.text}")
            return 0
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return 0

def segment_dealer_row(row):
    payload = {
        "Inventory_Level": int(row['Inventory_Level']),
        "Service_Responsiveness_Score": float(row['Service_Responsiveness_Score']),
        "Turnaround_Time_Hrs": float(row['Turnaround_Time_Hrs']),
        "Customer_Satisfaction_Index": float(row['Customer_Satisfaction_Index'])
    }
    try:
        response = requests.post(f"{API_URL}/predict/dealer-segment", json=payload)
        if response.status_code == 200:
            return response.json()['segment_label']
    except:
        pass
    return "Unknown"

def get_best_dealer(dealers_df):
    # Apply clustering via API
    # Note: For efficiency in production, implement a batch API endpoint.
    # For now, we iterate (acceptable for small datasets)
    
    # We'll use a progress bar if the dataset is large, but for now just map
    dealers_df['Segment'] = dealers_df.apply(segment_dealer_row, axis=1)
    
    # Filter for Premium Partners with stock
    best_dealers = dealers_df[
        (dealers_df['Segment'] == 'Premium Partner') & 
        (dealers_df['Inventory_Level'] > 0)
    ].sort_values(by='Service_Responsiveness_Score', ascending=False)
    
    return best_dealers

# --- UI Layout ---

# Sidebar
st.sidebar.title("NBC Nexus Control")
st.sidebar.markdown("The Autonomous Circular Lifecycle Engine")
page = st.sidebar.radio("Module", ["IoT & Predictive Maintenance", "Dealer Opportunity Engine", "Circular Economy Hub"])

st.sidebar.markdown("---")
st.sidebar.info("Connected to NBC Nexus Core\nStatus: Online 🟢")

# --- AI Chatbot ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Bearing Bot 🤖. I can help with predictive diagnostics, inventory routing, and sustainability reports."}
    ]

# Sidebar Chat Interface
st.sidebar.divider()
st.sidebar.markdown("### 💬 AI Assistant")

# Display chat messages in the sidebar (scrollable container usually)
with st.sidebar.container(height=300):
    for message in st.session_state.messages:
        # Use avatar for better UI
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Chat Input (Pinned to sidebar bottom)
if prompt := st.sidebar.chat_input("Ask Bearing Bot..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process with Gemini
    try:
        if "gemini" in st.secrets:
            genai.configure(api_key=st.secrets["gemini"]["api_key"])
            model = genai.GenerativeModel('gemini-3-flash-preview')
            
            # Build Context from Dashboard State
            context = f"""
            You are Bearing Bot, the AI engine for NBC Nexus.
            User is currently viewing the '{page}' module.
            System Date: {datetime.now().strftime('%Y-%m-%d')}
            Topic: Bearings (Ball, Roller), Predictive Maintenance, Dealer Fulfillment, Circular Economy.
            
            Be concise, professional, and helpful. 
            """
            
            # Simple stateless call for demo (or maintain history string)
            history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
            full_prompt = f"{context}\n\nChat History:\n{history_context}\n\nUser: {prompt}\nAssistant:"
            
            response = model.generate_content(full_prompt)
            reply = response.text
        else:
            reply = "⚠️ API Key not found. Please check secrets.toml."
    except Exception as e:
        reply = f"System Error: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# --- Page 1: Predictive Maintenance ---
if page == "IoT & Predictive Maintenance":
    st.title("🏭 Predictive Pulse (Digital Twin)")
    st.markdown("Real-time telemetry and RUL prediction for field assets.")

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Assets", len(bearings_df))
    critical_count = len(bearings_df[bearings_df['Status'] == 'Critical'])
    col2.metric("Critical Alerts", critical_count, delta=f"{critical_count}", delta_color="inverse")
    avg_rul = round(bearings_df['RUL_Days'].mean(), 1)
    col3.metric("Fleet Avg RUL", f"{avg_rul} Days")
    col4.metric("AI Accuracy", "94.2%")

    st.markdown("### Asset Inspector")
    
    # Selection
    selected_unit = st.selectbox("Select Bearing Unit for Analysis", bearings_df['Unit_ID'].unique())
    unit_data = bearings_df[bearings_df['Unit_ID'] == selected_unit].iloc[0]
    
    # Simulation Interface (What-If Analysis)
    st.subheader(f"Digital Twin: {selected_unit}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Live Telemetry Replay**")
        input_rpm = st.slider("RPM", 0, 3000, int(unit_data['RPM']))
        input_temp = st.slider("Temperature (°C)", 20.0, 150.0, float(unit_data['Temperature_C']))
        input_vib = st.slider("Vibration (mm/s)", 0.0, 10.0, float(unit_data['Vibration_mm_s']))
        input_hours = st.slider("Operating Hours", 0, 20000, int(unit_data['Operating_Hours']))
        input_lub = st.slider("Lubrication %", 0, 100, int(unit_data['Lubrication_Level_Pct']))
        input_load = st.slider("Load Factor", 0.0, 2.0, float(unit_data.get('Load_Factor', 1.0)))
        
        # Real-time Prediction
        pred_rul = predict_rul([input_hours, input_rpm, input_temp, input_vib, input_lub, input_load])
        
        status_color = "green"
        if pred_rul < 30: status_color = "red"
        elif pred_rul < 90: status_color = "orange"
        
        st.markdown(f"### AI Predicted RUL: :{status_color}[{int(pred_rul)} Days]")
        if pred_rul < 30:
            st.error("⚠️ CRITICAL FAILURE IMMINENT - REPLACEMENT TRIGGERED")
            st.session_state['trigger_order'] = True
            st.session_state['failed_unit'] = selected_unit
        else:
            st.success("Asset Healthy")
            st.session_state['trigger_order'] = False

    with col2:
        # Gauge Chart for RUL
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_rul,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Remaining Useful Life (Days)"},
            gauge = {
                'axis': {'range': [None, 365]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 90], 'color': "orange"},
                    {'range': [90, 365], 'color': "lightgreen"}],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Historical Vibration Trend (Simulated)**")
        # Generate dummy historical data for the chart
        dates = pd.date_range(end=datetime.now(), periods=30)
        hist_vib = [input_vib * (1 - (i/100)) for i in range(30)][::-1] # Increasing trend
        trend_df = pd.DataFrame({"Date": dates, "Vibration": hist_vib})
        fig_trend = px.line(trend_df, x="Date", y="Vibration", title="30-Day Vibration History")
        st.plotly_chart(fig_trend, use_container_width=True)

# --- Page 2: Dealer Engine ---
elif page == "Dealer Opportunity Engine":
    st.title("🤝 Opportunity Engine")
    st.markdown("Automated fulfillment routing based on Dealer performance.")
    
    if 'trigger_order' in st.session_state and st.session_state['trigger_order']:
        st.warning(f"🚨 ORDER TRIGGER RECEIVED: Replacement Bearing for Unit {st.session_state.get('failed_unit', 'Unknown')}")
        
        st.markdown("### AI Dealer Recommendation")
        st.write("Analyzing network for: **Stock Availability**, **Responsiveness**, and **Performance Score**...")
        
        best_dealers = get_best_dealer(dealers_df.copy())
        
        # Display Best Options
        for idx, row in best_dealers.head(3).iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                c1.subheader(row['Name'])
                c1.caption(f"{row['Location']} • {row['Segment']}")
                
                c2.metric("Turnaround", f"{row['Turnaround_Time_Hrs']} hrs")
                c2.caption("Est. Delivery")
                
                c3.metric("Inventory", row['Inventory_Level'])
                c3.caption("Units in Stock")
                
                c4.button(f"Dispatch Order #{np.random.randint(1000,9999)}", key=idx)
            st.divider()
            
        st.markdown("### Network Performance Map")
        fig = px.scatter_mapbox(dealers_df, 
                                lat=[20.5937 + np.random.randn() for _ in range(len(dealers_df))], # Dummy coords for demo
                                lon=[78.9629 + np.random.randn() for _ in range(len(dealers_df))],
                                color="Service_Responsiveness_Score",
                                size="Inventory_Level",
                                hover_name="Name",
                                zoom=4,
                                height=400,
                                title="Dealer Network Heatmap")
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No active replacement triggers. Select a bearing in 'Predictive Maintenance' and simulate a failure (low RUL) to see the engine in action.")
        
        st.subheader("Dealer Network Segmentation")
        # Apply segmentation for view
        if 'Segment' not in dealers_df.columns:
             dealers_df['Segment'] = dealers_df.apply(segment_dealer_row, axis=1)

        fig = px.scatter(dealers_df, x="Turnaround_Time_Hrs", y="Customer_Satisfaction_Index", 
                         color="Segment", size="Inventory_Level", hover_data=['Name'],
                         title="Partner Segmentation Analysis")
        st.plotly_chart(fig, use_container_width=True)

# --- Page 3: Circular Economy ---
elif page == "Circular Economy Hub":
    st.title("♻️ Circular Economy Handshake")
    st.markdown("Core exchange and sustainability tracking.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Recovery Impact")
        
        # KPI Cards
        st.metric("Bearings Remanufactured (YTD)", "1,248 Units")
        st.metric("Steel Saved", "4,120 kg")
        st.metric("CO2 Abitement", "842 Tons", delta="12% vs last year")
        
    with col2:
        st.subheader("Remanufacturability Assessment")
        
        st.write("Upload Bearing Image for Visual Inspection AI (Mock)")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Bearing Core', width=300)
            st.success("✅ Assessment Complete: **Grade B Core**")
            st.write("Recommendation: **Remanufacture** (Surface Grinding Required)")
            st.button("Generate Return Label")

    st.markdown("### Sustainability Funnel")
    fig = go.Figure(go.Funnel(
        y = ["Sold", "Returned Cores", "Inspectable", "Remanufacturable", "Back to Market"],
        x = [10000, 4500, 4200, 3100, 3100],
        textinfo = "value+percent initial"))
    st.plotly_chart(fig, use_container_width=True)
