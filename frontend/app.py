import streamlit as st
import plotly.express as px
import numpy as np
import websockets
import json
import asyncio

# CORRECT PLACEMENT: st.set_page_config() is the first Streamlit command
st.set_page_config(page_title="IAE-Transformer", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .stTextInput input {font-size: 16px; padding: 12px;}
    .stSelectbox [data-baseweb=select] {border-radius: 8px;}
    .stButton button {background-color: #4CAF50; color: white;}
    .css-1aumxhk {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🧠 Interactive Attention Explorer")
    
    # Session state initialization
    if "attention" not in st.session_state:
        st.session_state.attention = None
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.selectbox(
            "Select Model",
            ["bert-base-uncased", "gpt2", "roberta-base", "t5-small"],
            index=0
        )
        
        st.markdown("---")
        st.header("📤 Custom Models")
        uploaded_model = st.file_uploader(
            "Upload Model Weights",
            type=["bin", "pt"],
            accept_multiple_files=False
        )
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_input(
            "Enter text to analyze:",
            "The cat sat on the mat while watching birds outside.",
            key="input_text"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Attention")
    
    if analyze_btn or text_input:
        with st.spinner("Computing attention patterns..."):
            try:
                # WebSocket connection
                async def get_analysis():
                    async with websockets.connect(f"ws://localhost:8000/ws/analyze") as ws:
                        await ws.send(json.dumps({
                            "text": text_input,
                            "model": model_name
                        }))
                        return await ws.recv()
                
                response = asyncio.run(get_analysis())
                data = json.loads(response)
                
                st.session_state.attention = {
                    "tokens": data["tokens"],
                    "attentions": np.array(data["attentions"]),
                    "saliency": np.array(data["saliency"])
                }
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    if st.session_state.attention:
        st.markdown("---")
        st.header("🔍 Attention Visualization")
        
        # Layer/Head selection
        col_a, col_b = st.columns(2)
        with col_a:
            layer = st.slider(
                "Select Layer",
                0, len(st.session_state.attention["attentions"])-1, 0
            )
        with col_b:
            head = st.slider(
                "Select Attention Head",
                0, st.session_state.attention["attentions"][0].shape[1]-1, 0
            )
        
        # Plot attention heatmap
        fig = px.imshow(
            st.session_state.attention["attentions"][layer][0][head],
            x=st.session_state.attention["tokens"],
            y=st.session_state.attention["tokens"],
            color_continuous_scale="Viridis",
            labels={"x": "Output Token", "y": "Input Token"},
            title=f"Attention Pattern - Layer {layer+1}, Head {head+1}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Saliency visualization
        st.markdown("---")
        st.header("📊 Token Saliency Scores")
        saliency_fig = px.bar(
            x=st.session_state.attention["tokens"],
            y=st.session_state.attention["saliency"],
            labels={"x": "Token", "y": "Saliency"},
            color=st.session_state.attention["saliency"],
            color_continuous_scale="Blues"
        )
        st.plotly_chart(saliency_fig, use_container_width=True)

if __name__ == "__main__":
    main()