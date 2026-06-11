import streamlit as st
import pandas as pd
import io
import os
from backend.agent import DataAgent
import json

st.set_page_config(page_title="Data Preprocessing Agent", layout="wide")

def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'agent' not in st.session_state:
        st.session_state['agent'] = DataAgent()
    if 'profile' not in st.session_state:
        st.session_state['profile'] = None
    if 'strategy' not in st.session_state:
        st.session_state['strategy'] = None
    if 'cleaned_data' not in st.session_state:
        st.session_state['cleaned_data'] = None


initialize_session_state()

st.title("🧹 Autonomous Data Preprocessing Agent")

# Sidebar
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Messy CSV", type=["csv"])
    if uploaded_file is not None:
        if st.session_state['data'] is None:
            import io
            from pandas.errors import ParserError

            def _try_read(file_obj, **kwargs):
                file_obj.seek(0)
                return pd.read_csv(file_obj, **kwargs)

            df = None
            warnings_list = []

            # Stage 1: UTF-8, strict
            try:
                df = _try_read(uploaded_file)
            except UnicodeDecodeError:
                pass
            except ParserError:
                pass

            # Stage 2: Latin-1, skip bad lines
            if df is None:
                try:
                    df = _try_read(uploaded_file, encoding='latin1', on_bad_lines='skip')
                    warnings_list.append("⚠️ File used Latin-1 encoding (not UTF-8).")
                except (UnicodeDecodeError, ParserError):
                    pass

            # Stage 3: UTF-8 skip bad lines
            if df is None:
                try:
                    df = _try_read(uploaded_file, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')
                    warnings_list.append("⚠️ Some unreadable characters or malformed rows were dropped.")
                except ParserError:
                    pass

            # Stage 4: Auto-detect delimiter (handles semicolons, tabs, pipes, etc.)
            if df is None:
                try:
                    df = _try_read(uploaded_file, sep=None, engine='python', encoding_errors='ignore', on_bad_lines='skip')
                    warnings_list.append("⚠️ Non-standard delimiter detected and auto-resolved.")
                except Exception:
                    pass

            if df is None:
                st.error("❌ Could not parse this file. Please check that it is a valid CSV and try again.")
            else:
                for w in warnings_list:
                    st.warning(w)
                if not warnings_list and 'on_bad_lines' not in str(warnings_list):
                    pass  # clean read, no warnings needed
                st.session_state['data'] = df
                st.success("File uploaded successfully!")
            
            # Reset downstreams
            st.session_state['profile'] = None
            st.session_state['strategy'] = None
            st.session_state['cleaned_data'] = None
            st.session_state['agent'] = DataAgent()
            
    if st.session_state['data'] is not None:
        st.header("2. Global Settings")
        
        columns = st.session_state['data'].columns.tolist()
        target_col = st.selectbox("Select Target Variable", options=columns, index=len(columns)-1)
        task_type = st.selectbox("Task Type", ["Classification", "Regression"])
        
        if 'prev_target_col' not in st.session_state:
            st.session_state['prev_target_col'] = target_col
        if 'prev_task_type' not in st.session_state:
            st.session_state['prev_task_type'] = task_type
            
        if target_col != st.session_state['prev_target_col'] or task_type != st.session_state['prev_task_type']:
            st.session_state['profile'] = None
            st.session_state['strategy'] = None
            st.session_state['cleaned_data'] = None
            st.session_state['prev_target_col'] = target_col
            st.session_state['prev_task_type'] = task_type
            
        if st.button("Generate Strategy"):
            with st.spinner("Agent is profiling data..."):
                st.session_state['profile'] = st.session_state['agent'].profile(st.session_state['data'], target_col=target_col, task_type=task_type)
                st.session_state['strategy'] = st.session_state['agent'].propose_strategy(st.session_state['profile'])
            st.success("Strategy generated! Review in Tab 2.")

# Main View
if st.session_state['data'] is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Raw Data & Profiling", "⚙️ Interactive Strategy", "✨ Cleaned Data & Logs"])
    
    with tab1:
        st.subheader("Raw Dataset")
        st.dataframe(st.session_state['data'].head(50))
        
        if st.session_state['profile']:
            st.subheader("Profiling Metrics & Anomalies")
            profile_df = pd.DataFrame.from_dict(st.session_state['profile'], orient='index')
            st.dataframe(profile_df)
            
    with tab2:
        if st.session_state['strategy']:
            st.subheader("Review and Adjust Preprocessing Strategy")
            st.info("The agent has proposed the following steps based on data profiling. You can override them before execution.")
            
            edited_strategy = {}
            for col, strat in st.session_state['strategy'].items():
                resolved = st.session_state['profile'][col]['resolved_type']
                if resolved == 'ignore':
                    continue
                    
                with st.expander(f"Column: {col} (Resolved as {resolved})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        type_cast_options = ["None", "to_numeric", "to_string"]
                        current_type = strat['type_cast'] if strat['type_cast'] else "None"
                        new_type = st.selectbox("Type Cast", type_cast_options, index=type_cast_options.index(current_type), key=f"{col}_type")

                    with col2:
                        impute_options = ["None", "mean", "median", "most_frequent", "constant", "knn"]
                        current_impute = strat['impute'] if strat['impute'] else "None"
                        new_impute = st.selectbox("Imputation", impute_options, index=impute_options.index(current_impute), key=f"{col}_impute")

                    edited_strategy[col] = {
                        'type_cast': new_type if new_type != "None" else None,
                        'impute': new_impute if new_impute != "None" else None,
                    }
            
            st.session_state['strategy'] = edited_strategy
            
            if st.button("🚀 Execute Preprocessing", type="primary"):
                with st.spinner("Fitting models and transforming data..."):
                    # We create a fresh agent instance for execution to keep logs clean
                    execution_agent = DataAgent()
                    execution_agent.profile(st.session_state['data'], target_col=st.session_state['agent'].target_col, task_type=st.session_state['agent'].task_type)
                    execution_agent.fit(st.session_state['data'], st.session_state['strategy'])
                    st.session_state['cleaned_data'] = execution_agent.transform(st.session_state['data'])
                    st.session_state['agent'] = execution_agent
                st.success("Preprocessing Complete! Check Tab 3.")

    with tab3:
        if st.session_state['cleaned_data'] is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Data (Snippet)")
                st.dataframe(st.session_state['data'].head(20))
            with col2:
                st.subheader("Cleaned Data (Snippet)")
                st.dataframe(st.session_state['cleaned_data'].head(20))
                
            st.subheader("Agent Execution Logs")
            for log in getattr(st.session_state['agent'], 'logs', []):
                st.code(log, language="text")
                
            st.subheader("Downloads")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = st.session_state['cleaned_data'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Cleaned CSV",
                    data=csv,
                    file_name='cleaned_data.csv',
                    mime='text/csv',
                )
            with col2:
                import joblib, io as _io
                pkl_buf = _io.BytesIO()
                joblib.dump(st.session_state['agent'], pkl_buf)
                st.download_button(
                    label="⬇️ Download Preprocessor (.pkl)",
                    data=pkl_buf.getvalue(),
                    file_name='preprocessor.pkl',
                    mime='application/octet-stream',
                )
        else:
            st.info("Execute preprocessing in Tab 2 to view cleaned data and logs.")
else:
    st.info("Please upload a CSV file from the sidebar to begin.")
