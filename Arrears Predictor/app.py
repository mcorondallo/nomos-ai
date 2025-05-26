import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from model import ArrearsPredictor
from utils import generate_pdf_report, load_dummy_data, generate_excel_report, generate_sample_template
import io
import os
import base64
from datetime import datetime
import time
from PIL import Image

# Page config
st.set_page_config(
    page_title="Nomos AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {display: flex; align-items: center;}
    .logo-img {float: left; margin-right: 20px; height: 100px;}
    .main-txt {float: right;}
    .highlight {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .metric-card {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .stButton>button {background-color: #0066cc; color: white;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size: 1.2rem;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ArrearsPredictor()
    
if 'historical_data' not in st.session_state:
    # Load dummy data initially
    st.session_state.historical_data = load_dummy_data()
    
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
    
if 'confidence_intervals' not in st.session_state:
    st.session_state.confidence_intervals = None
    
if 'feature_correlation' not in st.session_state:
    st.session_state.feature_correlation = None

# Title and description with logo
logo_col, title_col = st.columns([1, 4])

with logo_col:
    try:
        image = Image.open('assets/Nomos_AI.png')
        st.image(image, width=180)
    except FileNotFoundError:
        st.warning("Logo not found. Please place the Nomos_AI.png file in the assets folder.")

with title_col:
    st.title("Nomos AI")
    st.markdown("""
    <div style='font-size: 1.2em;'>
    Advanced arrears payment prediction powered by machine learning. Analyze payment likelihood 
    based on historical patterns, client behavior, and geographical context.
    </div>
    """, unsafe_allow_html=True)

# Add date and time info
st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%B %d, %Y %H:%M')}")

# Version info
st.sidebar.markdown("**Version:** 1.0.0")

# Add sidebar info
with st.sidebar.expander("About Nomos AI"):
    st.markdown("""
    **Nomos AI** is an advanced predictive analytics platform designed for financial risk assessment. 
    
    Key features:
    - Payment likelihood prediction
    - Risk categorization
    - Geographical analysis
    - Feature importance visualization
    - Customizable reporting
    
    Leverage historical data to make informed decisions about arrears collection and risk management.
    """)

st.sidebar.markdown("---")

# Add a quick tutorial in the sidebar
with st.sidebar.expander("Quick Tutorial"):
    st.markdown("""
    1. **Train Model**: Upload historical payment data or use the provided dummy data
    2. **Make Predictions**: Upload new data or enter details manually
    3. **View Results**: Analyze predictions and generate custom reports
    
    Start by training a model with the dummy data to see how it works!
    """)
    
# Add benchmark configuration
with st.sidebar.expander("Configure Benchmark"):
    benchmark = st.slider("Payment Likelihood Benchmark (%)", 0, 100, 75)
    st.session_state.benchmark = benchmark / 100
    
    # Risk thresholds
    st.markdown("**Risk Category Thresholds**")
    col1, col2 = st.columns(2)
    with col1:
        low_risk = st.number_input("Low Risk Threshold", 0.0, 1.0, 0.7, 0.05)
    with col2:
        high_risk = st.number_input("High Risk Threshold", 0.0, 1.0, 0.3, 0.05)
    
    st.session_state.risk_thresholds = (high_risk, low_risk)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📊 Train Model", "🔮 Make Predictions", "📈 Results & Reports", "📋 Data Insights"])

with tab1:
    st.header("Train Model with Historical Data")
    
    train_col1, train_col2 = st.columns([2, 1])
    
    with train_col1:
        # Option to upload historical data
        with st.expander("Upload or Select Data", expanded=True):
            data_source = st.radio(
                "Select Data Source",
                ["Use Dummy Data", "Upload CSV File", "Download and Edit Template"],
                horizontal=True
            )
            
            if data_source == "Upload CSV File":
                historical_file = st.file_uploader("Upload CSV file with historical payment data", type="csv")
                
                if historical_file is not None:
                    try:
                        historical_data = pd.read_csv(historical_file)
                        st.success(f"Successfully loaded {len(historical_data)} records!")
                        st.session_state.historical_data = historical_data
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            elif data_source == "Download and Edit Template":
                template_buffer = generate_sample_template()
                st.download_button(
                    label="Download CSV Template",
                    data=template_buffer,
                    file_name="nomos_ai_template.csv",
                    mime="text/csv"
                )
                st.info("Download the template, fill it with your historical data, and upload it using the 'Upload CSV File' option.")
            else:
                st.info("Using dummy data for model training (200 sample records)")
        
        # Data preview
        st.subheader("Data Preview")
        with st.container():
            if len(st.session_state.historical_data) > 0:
                # Add tabs for different data views
                data_view_tab1, data_view_tab2, data_view_tab3 = st.tabs(["Table View", "Summary Statistics", "Data Visualization"])
                
                with data_view_tab1:
                    st.dataframe(st.session_state.historical_data.head(10), height=300, use_container_width=True)
                    st.caption(f"Showing 10 of {len(st.session_state.historical_data)} records")
                
                with data_view_tab2:
                    st.write("Numerical Features Summary")
                    st.dataframe(st.session_state.historical_data.describe().round(2), use_container_width=True)
                    
                    categorical_cols = st.session_state.historical_data.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        st.write("Categorical Features")
                        for col in categorical_cols:
                            st.write(f"**{col}** distribution:")
                            st.bar_chart(st.session_state.historical_data[col].value_counts())
                
                with data_view_tab3:
                    if 'was_paid' in st.session_state.historical_data.columns:
                        # Payment distribution
                        fig = px.pie(
                            st.session_state.historical_data, 
                            names='was_paid', 
                            title='Payment Distribution (0=Not Paid, 1=Paid)',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature distribution by payment status
                        feature_to_plot = st.selectbox(
                            "Select feature to visualize by payment status:",
                            st.session_state.model.numeric_features
                        )
                        
                        fig = px.histogram(
                            st.session_state.historical_data,
                            x=feature_to_plot,
                            color="was_paid",
                            barmode="overlay",
                            title=f"{feature_to_plot} Distribution by Payment Status",
                            labels={"was_paid": "Payment Status (1=Paid)"},
                            histnorm="probability density"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Target variable 'was_paid' not found in the data. Basic visualizations shown.")
                        
                        # Feature distributions
                        feature_to_plot = st.selectbox(
                            "Select feature to visualize:",
                            st.session_state.model.numeric_features
                        )
                        
                        fig = px.histogram(
                            st.session_state.historical_data,
                            x=feature_to_plot,
                            title=f"{feature_to_plot} Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
    with train_col2:
        st.subheader("Model Configuration")
        with st.expander("Training Parameters", expanded=True):
            n_estimators = st.slider("Number of Estimators", 50, 500, 100, 50)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
                value=0.1
            )
            max_depth = st.slider("Max Tree Depth", 2, 10, 4, 1)
            random_state = st.number_input("Random Seed", 0, 100, 42, 1)
            
            model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "random_state": random_state
            }
        
        st.subheader("Train Model")
        # Train model button
        if st.button("Train Model", key="train_model_btn"):
            with st.spinner("Training model..."):
                st.session_state.model.train(st.session_state.historical_data, model_params)
                st.success("Model trained successfully!")
                
                # Calculate feature correlation data for Data Insights tab
                numerical_features = st.session_state.historical_data[st.session_state.model.numeric_features]
                st.session_state.feature_correlation = numerical_features.corr()
                
                # Display training statistics
                st.metric("Model Accuracy", f"{st.session_state.model.model_accuracy:.2%}")
                
                # Show train/test split performance
                st.subheader("Model Performance")
                st.write(f"Training Score: {st.session_state.model.train_score:.4f}")
                st.write(f"Test Score: {st.session_state.model.test_score:.4f}")
                
                # Show feature importance
                if hasattr(st.session_state.model, 'feature_importance'):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.model.features,
                        'Importance': st.session_state.model.feature_importance
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        title='Feature Importance Analysis',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)  
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model option
                    if st.button("Save Trained Model"):
                        st.session_state.model.save_model("nomos_ai_model.pkl")
                        st.success("Model saved to 'nomos_ai_model.pkl'!")

with tab2:
    st.header("Make Predictions")
    
    # Check if model is trained
    if not st.session_state.model.trained:
        st.warning("⚠️ Model not yet trained. Please train the model first in the 'Train Model' tab or continue with the default model.")
        if st.button("Use Default Model"):
            with st.spinner("Training default model..."):
                st.session_state.model.train(st.session_state.historical_data)
                st.success("Default model trained successfully!")
    
    # Tabs for different input methods
    input_tab1, input_tab2, input_tab3 = st.tabs(["Upload CSV", "Manual Input", "Batch Analysis"])
    
    with input_tab1:
        st.subheader("Upload Data for Prediction")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            prediction_file = st.file_uploader("Upload CSV file with data to predict", type="csv")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("Download Template"):
                template_buffer = generate_sample_template(prediction_only=True)
                st.download_button(
                    label="⬇️ CSV Template",
                    data=template_buffer,
                    file_name="prediction_template.csv",
                    mime="text/csv",
                    key="pred_template_dl"
                )
        
        if prediction_file is not None:
            try:
                prediction_data = pd.read_csv(prediction_file)
                st.success(f"Successfully loaded {len(prediction_data)} records for prediction!")
                
                # Preview the data with column highlighting
                st.subheader("Data Preview")
                st.dataframe(prediction_data.head(5), height=230, use_container_width=True)
                
                missing_features = [f for f in st.session_state.model.features if f not in prediction_data.columns]
                if missing_features:
                    st.warning(f"⚠️ Warning: The following features are missing from your data: {', '.join(missing_features)}")
                    st.info("The model will use default values for these missing features.")
                
                # Run prediction with options
                with st.expander("Prediction Options", expanded=True):
                    include_confidence = st.checkbox("Include confidence intervals", value=True)
                    advanced_metrics = st.checkbox("Generate advanced analytics", value=True)
                    
                    if st.button("🔮 Run Prediction", key="run_csv_prediction", use_container_width=True):
                        with st.spinner("Analyzing data and generating predictions..."):
                            # Add a progress bar for better UX
                            progress_bar = st.progress(0)
                            for i in range(101):
                                time.sleep(0.01)
                                progress_bar.progress(i)
                            
                            results = st.session_state.model.predict(
                                prediction_data, 
                                include_confidence=include_confidence,
                                risk_thresholds=st.session_state.risk_thresholds if 'risk_thresholds' in st.session_state else None
                            )
                            st.session_state.prediction_results = results
                            
                            if include_confidence:
                                st.session_state.confidence_intervals = True
                            
                            st.success("✅ Predictions generated successfully! View details in the 'Results & Reports' tab.")
                            
                            # Show a quick summary
                            st.subheader("Quick Summary")
                            
                            risk_counts = results['risk_category'].value_counts()
                            avg_prob = results['payment_probability'].mean()
                            
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Average Payment Probability", f"{avg_prob:.2%}")
                            with metrics_col2:
                                high_risk = risk_counts.get('High Risk', 0)
                                st.metric("High Risk Cases", high_risk, delta=-high_risk, delta_color="inverse")
                            with metrics_col3:
                                low_risk = risk_counts.get('Low Risk', 0)
                                st.metric("Low Risk Cases", low_risk, delta=low_risk, delta_color="normal")
                                
                            # Button to jump to results tab
                            st.button("📊 View Detailed Results", key="goto_results_from_csv")
                            
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Make sure your CSV file contains all required features or use the provided template.")
    
    with input_tab2:
        st.subheader("Enter Data Manually")
        
        # Create a form for manual input with enhanced UI
        with st.form("manual_input_form"):
            st.markdown("<h4 style='text-align: center; color: #0066cc;'>Payment Prediction Form</h4>", unsafe_allow_html=True)
            
            # Get a sample record structure from the dummy data
            manual_input = {}
            
            # Use 3 columns for more compact layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Payment Information**")
                # Days overdue
                manual_input['days_overdue'] = st.slider("Days Overdue", 0, 180, 30)
                manual_input['previous_late_payments'] = st.slider("Previous Late Payments", 0, 10, 2)
                manual_input['average_days_late'] = st.slider("Average Days Late", 0, 90, 15)
                
            with col2:
                st.markdown("**Property & Lease Information**")
                # Lease information
                manual_input['lease_count'] = st.slider("Number of Leases", 1, 10, 1)
                manual_input['lease_amount'] = st.number_input("Lease Amount (€)", min_value=100, max_value=10000, value=1000)
                manual_input['property_size_sqm'] = st.number_input("Property Size (sqm)", min_value=10, max_value=1000, value=100)
            
            with col3:
                st.markdown("**Location & Payment Method**")
                # Payment method
                manual_input['payment_method'] = st.selectbox(
                    "Payment Method", 
                    options=["bank_transfer", "card"]
                )
                
                # Country
                manual_input['country'] = st.selectbox(
                    "Country", 
                    options=["France", "Germany", "Spain", "Italy", "UK",
                            "Netherlands", "Belgium", "Portugal", "Austria", "Switzerland"]
                )
                
                # Region
                manual_input['region'] = st.selectbox(
                    "Region", 
                    options=["North", "South", "East", "West", "Central"]
                )
            
            # Add an option for confidence intervals
            include_confidence = st.checkbox("Include confidence intervals in prediction", value=True)
                
            # Use a primary-colored button
            submitted = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)
        
        # Process the form submission outside the form context
        if submitted:
            with st.spinner("Analyzing data points and generating prediction..."):
                # Add a progress bar for better UX
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                
                # Convert the manual input to a DataFrame
                input_df = pd.DataFrame([manual_input])
                
                # Make the prediction
                results = st.session_state.model.predict(
                    input_df,
                    include_confidence=include_confidence,
                    risk_thresholds=st.session_state.risk_thresholds if 'risk_thresholds' in st.session_state else None
                )
                st.session_state.prediction_results = results
                
                if include_confidence:
                    st.session_state.confidence_intervals = True
                
                # Show the results inline
                st.success("✅ Prediction generated successfully!")
                
                # Display the predicted payment probability with a gauge chart
                payment_prob = results['payment_probability'].iloc[0]
                risk_category = results['risk_category'].iloc[0]
                
                # Determine color based on risk category
                if risk_category == 'High Risk':
                    color = "red"
                elif risk_category == 'Medium Risk':
                    color = "orange"
                else:
                    color = "green"
                
                st.markdown(f"<h3 style='text-align: center;'>Prediction Results</h3>", unsafe_allow_html=True)
                
                # Create columns for visualization
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=payment_prob * 100,
                        title={'text': "Payment Probability"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'ticksuffix': "%"},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "darkgray"}
                            ]
                        },
                        number={'suffix': "%", 'font': {'size': 24}}
                    ))
                    fig.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                
                with res_col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                    st.markdown(f"<div style='text-align: center; font-size: 22px; font-weight: bold;'>Risk Category: <span style='color: {color};'>{risk_category}</span></div>", unsafe_allow_html=True)
                    
                    if include_confidence:
                        lower_bound = results['lower_bound'].iloc[0] if 'lower_bound' in results.columns else payment_prob - 0.1
                        upper_bound = results['upper_bound'].iloc[0] if 'upper_bound' in results.columns else payment_prob + 0.1
                        
                        st.markdown(f"<div style='text-align: center; font-size: 16px;'>Confidence Interval:<br>{lower_bound:.2%} - {upper_bound:.2%}</div>", unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                    benchmark = st.session_state.benchmark if 'benchmark' in st.session_state else 0.75
                    delta = payment_prob - benchmark
                    delta_text = f"+{delta:.2%}" if delta > 0 else f"{delta:.2%}"
                    delta_color = "green" if delta > 0 else "red"
                    
                    st.markdown(f"<div style='text-align: center; font-size: 16px;'>Benchmark Comparison: <span style='color: {delta_color};'>{delta_text}</span></div>", unsafe_allow_html=True)
                
                # Add button to view full results (now outside the form)
                st.button("📊 View Detailed Results", key="goto_results_from_manual")
    
    with input_tab3:
        st.subheader("Batch Analysis & Simulation")
        
        # Introduction to batch analysis
        st.markdown("""
        Use this section to analyze how changing certain parameters affects payment probability 
        across different scenarios. This is useful for sensitivity analysis and what-if scenarios.
        """)
        
        # Create batch simulation form
        with st.form("batch_simulation_form"):
            st.markdown("**Select parameter to vary:**")
            
            # Choose which parameter to simulate
            param_to_vary = st.selectbox(
                "Parameter",
                options=[
                    "days_overdue", 
                    "previous_late_payments", 
                    "average_days_late", 
                    "lease_count", 
                    "lease_amount"
                ]
            )
            
            # Set base values for all parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Parameter range:**")
                # Set min/max/step values based on selected parameter
                if param_to_vary in ["days_overdue", "average_days_late"]:
                    min_val = st.number_input(f"Minimum {param_to_vary}", 0, 180, 0, 10)
                    max_val = st.number_input(f"Maximum {param_to_vary}", 0, 180, 60, 10)
                    step_val = st.number_input(f"Step size", 1, 30, 10, 1)
                elif param_to_vary == "previous_late_payments":
                    min_val = st.number_input(f"Minimum {param_to_vary}", 0, 10, 0, 1)
                    max_val = st.number_input(f"Maximum {param_to_vary}", 0, 10, 5, 1)
                    step_val = st.number_input(f"Step size", 1, 5, 1, 1)
                elif param_to_vary == "lease_count":
                    min_val = st.number_input(f"Minimum {param_to_vary}", 1, 10, 1, 1)
                    max_val = st.number_input(f"Maximum {param_to_vary}", 1, 10, 5, 1)
                    step_val = st.number_input(f"Step size", 1, 5, 1, 1)
                elif param_to_vary == "lease_amount":
                    min_val = st.number_input(f"Minimum {param_to_vary}", 100, 10000, 500, 500)
                    max_val = st.number_input(f"Maximum {param_to_vary}", 100, 10000, 3000, 500)
                    step_val = st.number_input(f"Step size", 100, 1000, 500, 100)
            
            with col2:
                st.markdown("**Fixed Parameters:**")
                # Set fixed values for other parameters
                fixed_params = {}
                
                for param in ["days_overdue", "previous_late_payments", "average_days_late", "lease_count", "lease_amount"]:
                    if param != param_to_vary:
                        if param == "days_overdue":
                            fixed_params[param] = st.slider(f"Fixed {param}", 0, 180, 30, 10)
                        elif param == "previous_late_payments":
                            fixed_params[param] = st.slider(f"Fixed {param}", 0, 10, 2, 1)
                        elif param == "average_days_late":
                            fixed_params[param] = st.slider(f"Fixed {param}", 0, 90, 15, 5)
                        elif param == "lease_count":
                            fixed_params[param] = st.slider(f"Fixed {param}", 1, 10, 1, 1)
                        elif param == "lease_amount":
                            fixed_params[param] = st.slider(f"Fixed {param}", 500, 5000, 1000, 500)
                
                # Additional fixed parameters
                fixed_params["payment_method"] = st.selectbox("Payment Method", options=["bank_transfer", "card"])
                fixed_params["country"] = st.selectbox("Country", options=["France", "Germany", "Spain", "Italy", "UK"])
                fixed_params["region"] = st.selectbox("Region", options=["North", "South", "East", "West", "Central"])
                fixed_params["property_size_sqm"] = st.slider("Property Size (sqm)", 50, 500, 100, 50)
            
            # Submit button
            submitted = st.form_submit_button("🔄 Run Simulation", use_container_width=True)
            
            if submitted:
                with st.spinner("Running batch simulation..."):
                    # Generate a range of values
                    param_values = list(range(min_val, max_val + 1, step_val))
                    
                    # Create a list of dictionaries with all combinations
                    simulation_data = []
                    for val in param_values:
                        sim_case = fixed_params.copy()
                        sim_case[param_to_vary] = val
                        simulation_data.append(sim_case)
                    
                    # Convert to DataFrame
                    sim_df = pd.DataFrame(simulation_data)
                    
                    # Run prediction
                    results = st.session_state.model.predict(sim_df)
                    
                    # Store simulation results
                    st.session_state.simulation_results = {
                        'parameter': param_to_vary,
                        'values': param_values,
                        'results': results
                    }
                    
                    # Show results
                    st.success("✅ Simulation completed successfully!")
                    
                    # Visualization
                    st.subheader("Simulation Results")
                    
                    # Line chart of parameter vs probability
                    fig = px.line(
                        results, 
                        x=param_to_vary, 
                        y="payment_probability",
                        title=f"Effect of {param_to_vary} on Payment Probability",
                        markers=True
                    )
                    
                    # Add benchmark line
                    benchmark = st.session_state.benchmark if 'benchmark' in st.session_state else 0.75
                    fig.add_hline(y=benchmark, line_dash="dash", line_color="red", annotation_text="Benchmark")
                    
                    fig.update_layout(
                        xaxis_title=param_to_vary,
                        yaxis_title="Payment Probability",
                        yaxis=dict(tickformat=".0%"),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table view
                    results_display = results[[param_to_vary, 'payment_probability', 'risk_category']].copy()
                    results_display['payment_probability'] = results_display['payment_probability'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(results_display, use_container_width=True)
                    
                    # Download option
                    st.download_button(
                        label="⬇️ Download Simulation Results",
                        data=results.to_csv(index=False).encode('utf-8'),
                        file_name=f"simulation_results_{param_to_vary}.csv",
                        mime="text/csv"
                    )

with tab3:
    st.header("Prediction Results & Reports")
    
    if st.session_state.prediction_results is not None:
        # Get benchmark value from session state or use default
        benchmark = st.session_state.benchmark if 'benchmark' in st.session_state else 0.75
        
        # Create tabs for different views
        results_tabs = st.tabs(["Summary Dashboard", "Detailed Results", "Risk Analysis", "Reports"])
        
        results_df = st.session_state.prediction_results
        
        with results_tabs[0]:  # Summary Dashboard
            st.subheader("Payment Prediction Summary")
            
            # Key metrics in cards
            avg_probability = results_df['payment_probability'].mean()
            risk_counts = results_df['risk_category'].value_counts()
            high_risk = risk_counts.get('High Risk', 0)
            medium_risk = risk_counts.get('Medium Risk', 0)
            low_risk = risk_counts.get('Low Risk', 0)
            
            # First row of metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "Average Probability", 
                    f"{avg_probability:.2%}", 
                    f"{avg_probability - benchmark:.2%}",
                    delta_color="normal" if avg_probability >= benchmark else "inverse"
                )
            with metric_cols[1]:
                st.metric("Total Records", len(results_df))
            with metric_cols[2]:
                above_benchmark = (results_df['payment_probability'] >= benchmark).sum()
                st.metric(
                    "Above Benchmark", 
                    f"{above_benchmark} ({above_benchmark/len(results_df):.1%})"
                )
            with metric_cols[3]:
                below_benchmark = (results_df['payment_probability'] < benchmark).sum()
                st.metric(
                    "Below Benchmark", 
                    f"{below_benchmark} ({below_benchmark/len(results_df):.1%})",
                    delta=f"-{below_benchmark}",
                    delta_color="inverse"
                )
            
            # Risk distribution visualization
            st.subheader("Risk Distribution")
            risk_row1, risk_row2 = st.columns([1, 1])
            
            with risk_row1:
                # Pie chart of risk categories
                risk_data = pd.DataFrame({
                    'Risk Category': ['High Risk', 'Medium Risk', 'Low Risk'],
                    'Count': [high_risk, medium_risk, low_risk]
                })
                
                fig = px.pie(
                    risk_data, 
                    names='Risk Category', 
                    values='Count',
                    color='Risk Category',
                    color_discrete_map={
                        'High Risk': 'red',
                        'Medium Risk': 'orange',
                        'Low Risk': 'green'
                    },
                    hole=0.4
                )
                fig.update_layout(
                    title="Risk Category Distribution",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with risk_row2:
                # Histogram of payment probabilities
                fig = px.histogram(
                    results_df, 
                    x='payment_probability',
                    nbins=20, 
                    title='Payment Probability Distribution',
                    color_discrete_sequence=['#0066cc']
                )
                
                # Add vertical line for benchmark
                fig.add_vline(
                    x=benchmark, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Benchmark ({benchmark:.0%})",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    xaxis_title="Payment Probability",
                    yaxis_title="Count",
                    xaxis=dict(tickformat=".0%"),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Geographic distribution if country data is available
            if 'country' in results_df.columns:
                st.subheader("Geographic Analysis")
                
                country_metrics = results_df.groupby('country').agg({
                    'payment_probability': ['mean', 'count'],
                    'risk_category': lambda x: (x == 'High Risk').sum()
                }).reset_index()
                
                country_metrics.columns = ['Country', 'Avg Probability', 'Count', 'High Risk Count']
                country_metrics['Avg Probability'] = country_metrics['Avg Probability'].round(4)
                
                # Sort by average probability
                country_metrics = country_metrics.sort_values('Avg Probability', ascending=False)
                
                # Calculate percentage of high risk
                country_metrics['High Risk %'] = (country_metrics['High Risk Count'] / country_metrics['Count']).round(4)
                
                # Bar chart for payment by country
                fig = px.bar(
                    country_metrics,
                    x='Country',
                    y='Avg Probability',
                    color='High Risk %',
                    text_auto='.1%',
                    color_continuous_scale='RdYlGn_r',
                    title="Average Payment Probability by Country"
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Average Payment Probability",
                    yaxis=dict(tickformat=".0%"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with results_tabs[1]:  # Detailed Results
            st.subheader("Detailed Prediction Results")
            
            # Filter options
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                min_prob = st.slider("Min Probability", 0.0, 1.0, 0.0, 0.05)
            with filter_col2:
                max_prob = st.slider("Max Probability", 0.0, 1.0, 1.0, 0.05)
            with filter_col3:
                risk_filter = st.multiselect(
                    "Risk Categories", 
                    options=["High Risk", "Medium Risk", "Low Risk"],
                    default=["High Risk", "Medium Risk", "Low Risk"]
                )
            
            # Apply filters
            filtered_df = results_df[
                (results_df['payment_probability'] >= min_prob) & 
                (results_df['payment_probability'] <= max_prob) &
                (results_df['risk_category'].isin(risk_filter))
            ]
            
            # Display data
            if not filtered_df.empty:
                # Format probabilities as percentages
                display_df = filtered_df.copy()
                for col in display_df.columns:
                    if 'probability' in col or 'bound' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    display_df,
                    height=400,
                    use_container_width=True
                )
                
                # Download options
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=filtered_df.to_csv(index=False).encode('utf-8'),
                        file_name="nomos_ai_predictions.csv",
                        mime="text/csv"
                    )
                
                with export_col2:
                    # Excel export with formatting
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        filtered_df.to_excel(writer, sheet_name='Predictions', index=False)
                        # Get the xlsxwriter workbook and worksheet objects
                        workbook = writer.book
                        worksheet = writer.sheets['Predictions']
                        # Add formats for percentages
                        percent_format = workbook.add_format({'num_format': '0.00%'})
                        # Apply formatting
                        for col_num, col_name in enumerate(filtered_df.columns):
                            if 'probability' in col_name or 'bound' in col_name:
                                worksheet.set_column(col_num, col_num, None, percent_format)
                    
                    excel_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=excel_buffer,
                        file_name="nomos_ai_predictions.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            else:
                st.warning("No records match the selected filters.")
        
        with results_tabs[2]:  # Risk Analysis
            st.subheader("Risk Analysis")
            
            # Create a column with both numeric and categorical data for analysis
            analysis_df = results_df.copy()
            
            # Select feature for analysis
            feature_options = [col for col in analysis_df.columns if col not in [
                'payment_probability', 'risk_category', 'lower_bound', 'upper_bound'
            ]]
            
            if feature_options:
                selected_feature = st.selectbox(
                    "Select feature for risk analysis:",
                    options=feature_options
                )
                
                # Check if feature is categorical or numeric
                is_categorical = analysis_df[selected_feature].dtype == 'object' or \
                                analysis_df[selected_feature].nunique() < 10
                
                if is_categorical:
                    # Group by the feature and calculate statistics
                    grouped = analysis_df.groupby(selected_feature).agg({
                        'payment_probability': ['mean', 'median', 'std', 'count'],
                        'risk_category': lambda x: (x == 'High Risk').mean() * 100  # % high risk
                    }).reset_index()
                    
                    # Flatten column names
                    grouped.columns = [selected_feature, 'Mean Probability', 'Median Probability', 
                                      'Std Deviation', 'Count', 'High Risk %']
                    
                    # Create charts
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        # Bar chart for mean probability by feature
                        fig = px.bar(
                            grouped,
                            x=selected_feature,
                            y='Mean Probability',
                            color='High Risk %',
                            text_auto='.1%',
                            color_continuous_scale='RdYlGn_r',
                            title=f"Mean Payment Probability by {selected_feature}"
                        )
                        fig.update_layout(
                            yaxis=dict(tickformat=".0%"),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with analysis_col2:
                        # Pie chart for record distribution
                        fig = px.pie(
                            grouped,
                            names=selected_feature,
                            values='Count',
                            title=f"Record Distribution by {selected_feature}"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Table with statistics
                    st.subheader(f"Statistical Analysis by {selected_feature}")
                    display_grouped = grouped.copy()
                    
                    # Format probability columns as percentages
                    for col in ['Mean Probability', 'Median Probability']:
                        display_grouped[col] = display_grouped[col].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(display_grouped, use_container_width=True)
                    
                else:  # Numeric feature
                    # Scatter plot of feature vs. probability
                    fig = px.scatter(
                        analysis_df,
                        x=selected_feature,
                        y='payment_probability',
                        color='risk_category',
                        color_discrete_map={
                            'High Risk': 'red',
                            'Medium Risk': 'orange',
                            'Low Risk': 'green'
                        },
                        title=f"Payment Probability vs. {selected_feature}"
                    )
                    
                    # Add trend line
                    fig.update_layout(
                        yaxis=dict(tickformat=".0%"),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    correlation = analysis_df[[selected_feature, 'payment_probability']].corr().iloc[0, 1]
                    
                    st.metric(
                        "Correlation with Payment Probability", 
                        f"{correlation:.4f}",
                        help="Measures how strongly the feature correlates with payment probability. Values close to 1 or -1 indicate strong correlation."
                    )
                    
                    # Group into bins for easier analysis
                    st.subheader(f"Binned Analysis of {selected_feature}")
                    bins = st.slider("Number of bins", 3, 10, 5)
                    
                    # Create bins
                    analysis_df[f"{selected_feature}_bin"] = pd.qcut(
                        analysis_df[selected_feature], 
                        bins, 
                        duplicates='drop'
                    )
                    
                    # Group by bins
                    binned = analysis_df.groupby(f"{selected_feature}_bin").agg({
                        'payment_probability': ['mean', 'count'],
                        'risk_category': lambda x: (x == 'High Risk').mean() * 100  # % high risk
                    }).reset_index()
                    
                    # Flatten column names
                    binned.columns = ['Bin', 'Mean Probability', 'Count', 'High Risk %']
                    
                    # Convert bins to strings for better display
                    binned['Bin'] = binned['Bin'].astype(str)
                    
                    # Create bar chart for binned data
                    fig = px.bar(
                        binned,
                        x='Bin',
                        y='Mean Probability',
                        text_auto='.1%',
                        title=f"Mean Payment Probability by {selected_feature} (Binned)",
                        color='High Risk %',
                        color_continuous_scale='RdYlGn_r',
                    )
                    fig.update_layout(
                        xaxis_title=selected_feature,
                        yaxis=dict(tickformat=".0%"),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No features available for analysis.")
        
        with results_tabs[3]:  # Reports
            st.subheader("Generate Reports")
            
            # Create different report options
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
                with st.container():
                    st.markdown("### PDF Report")
                    st.markdown("Generate a comprehensive PDF report with payment prediction analysis, risk distribution, and recommendations.")
                    
                    include_region = st.checkbox("Include regional analysis", value=True)
                    include_confidence = st.checkbox("Include confidence intervals", value='confidence_intervals' in st.session_state)
                    
                    if st.button("Generate PDF Report", key="gen_pdf_report", use_container_width=True):
                        with st.spinner("Generating PDF report..."):
                            # Add a progress bar
                            progress_bar = st.progress(0)
                            for i in range(101):
                                time.sleep(0.02)
                                progress_bar.progress(i)
                            
                            pdf_buffer = generate_pdf_report(
                                results_df, 
                                benchmark,
                                include_region=include_region,
                                include_confidence=include_confidence
                            )
                            
                            # Provide download link
                            st.success("PDF report generated successfully!")
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"nomos_ai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
            
            with report_col2:
                with st.container():
                    st.markdown("### Excel Report")
                    st.markdown("Generate a detailed Excel workbook with multiple tabs for different analyses and visualizations.")
                    
                    include_charts = st.checkbox("Include embedded charts", value=True)
                    include_detailed = st.checkbox("Include detailed data tables", value=True)
                    
                    if st.button("Generate Excel Report", key="gen_excel_report", use_container_width=True):
                        with st.spinner("Generating Excel report..."):
                            # Add a progress bar
                            progress_bar = st.progress(0)
                            for i in range(101):
                                time.sleep(0.02)
                                progress_bar.progress(i)
                            
                            excel_buffer = generate_excel_report(
                                results_df,
                                benchmark,
                                include_charts=include_charts,
                                include_detailed=include_detailed
                            )
                            
                            # Provide download link
                            st.success("Excel report generated successfully!")
                            st.download_button(
                                label="⬇️ Download Excel Report",
                                data=excel_buffer,
                                file_name=f"nomos_ai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                mime="application/vnd.ms-excel",
                                use_container_width=True
                            )
            
            # Batch report options
            if len(results_df) > 1:
                st.subheader("Batch Reporting")
                
                st.markdown("""
                Generate individual reports for each record or group by a specific category.
                This is useful for creating focused reports for different clients or regions.
                """)
                
                batch_cols = st.columns(2)
                with batch_cols[0]:
                    group_by = st.selectbox(
                        "Group reports by:",
                        options=["No grouping"] + [col for col in results_df.columns if col not in [
                            'payment_probability', 'risk_category', 'lower_bound', 'upper_bound'
                        ] and results_df[col].nunique() < 20]
                    )
                
                with batch_cols[1]:
                    report_type = st.radio(
                        "Report format:",
                        options=["PDF", "Excel"],
                        horizontal=True
                    )
                
                if st.button("Generate Batch Reports", use_container_width=True):
                    # This would be implemented in a real application
                    # Here we'll just show a message
                    st.info("Batch report generation would be implemented in the full version.")
                    
                    with st.spinner("Simulating batch report generation..."):
                        progress_bar = st.progress(0)
                        for i in range(101):
                            time.sleep(0.02)
                            progress_bar.progress(i)
                        
                        st.success("Batch reports would be generated and available for download.")
    else:
        st.info("📊 No prediction results available yet. Please make predictions in the 'Make Predictions' tab.")
        
        # Show a demo of what the results would look like
        if st.button("Load Demo Results"):
            with st.spinner("Loading demo results..."):
                # Generate some dummy prediction results
                dummy_data = load_dummy_data(50)  # 50 sample records
                dummy_predictions = st.session_state.model.predict(dummy_data)
                st.session_state.prediction_results = dummy_predictions
                st.session_state.confidence_intervals = True
                
                st.success("Demo results loaded! Refresh this section to view them.")
                st.rerun()

# Add Data Insights tab
with tab4:
    st.header("Data Insights & Analytics")
    
    insight_tabs = st.tabs(["Feature Correlation", "Trend Analysis", "Geographic Insights", "Payment Patterns"])
    
    with insight_tabs[0]:  # Feature Correlation
        st.subheader("Feature Correlation Analysis")
        
        if st.session_state.feature_correlation is not None:
            # Plot correlation heatmap
            st.markdown("""This analysis shows how different numerical features in the dataset correlate with each other. 
            Strong positive correlations (close to 1) appear in dark blue, while strong negative correlations (close to -1) appear in dark red.""")
            
            fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
            corr = st.session_state.feature_correlation
            
            # Generate heatmap
            sns.heatmap(
                corr, 
                annot=True, 
                fmt=".2f",
                cmap='coolwarm', 
                square=True,
                linewidths=.5, 
                ax=ax
            )
            plt.title('Feature Correlation Matrix', fontsize=15)
            st.pyplot(fig)
            
            # Feature pair analysis
            st.subheader("Feature Pair Analysis")
            if len(st.session_state.model.numeric_features) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox(
                        "Select X-axis feature:",
                        options=st.session_state.model.numeric_features,
                        index=0
                    )
                with col2:
                    y_feature = st.selectbox(
                        "Select Y-axis feature:",
                        options=st.session_state.model.numeric_features,
                        index=min(1, len(st.session_state.model.numeric_features) - 1)
                    )
                
                if x_feature != y_feature:
                    # Create scatter plot
                    if 'was_paid' in st.session_state.historical_data.columns:
                        fig = px.scatter(
                            st.session_state.historical_data, 
                            x=x_feature, 
                            y=y_feature,
                            color='was_paid',
                            title=f"Relationship between {x_feature} and {y_feature}",
                            labels={'was_paid': 'Payment Status'},
                            color_discrete_map={0: 'red', 1: 'green'}
                        )
                    else:
                        fig = px.scatter(
                            st.session_state.historical_data, 
                            x=x_feature, 
                            y=y_feature,
                            title=f"Relationship between {x_feature} and {y_feature}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display correlation
                    correlation = st.session_state.historical_data[[x_feature, y_feature]].corr().iloc[0, 1]
                    st.metric("Correlation Coefficient", f"{correlation:.4f}")
                    
                    # Interpretation
                    if abs(correlation) < 0.3:
                        st.info("These features have a weak correlation.")
                    elif abs(correlation) < 0.7:
                        st.info("These features have a moderate correlation.")
                    else:
                        st.warning("These features have a strong correlation. This may indicate multicollinearity in your model.")
                else:
                    st.warning("Please select different features for X and Y axes.")
            else:
                st.info("Need at least two numeric features to perform pair analysis.")
        else:
            st.info("Feature correlation data is not available. Train your model first in the 'Train Model' tab.")
    
    with insight_tabs[1]:  # Trend Analysis
        st.subheader("Payment Trend Analysis")
        
        st.markdown("""Analyze how different feature values affect payment probability. 
        This can help identify patterns and thresholds that influence payment likelihood.""")
        
        if st.session_state.model.trained:
            # Select feature to analyze
            feature = st.selectbox(
                "Select feature to analyze:",
                options=st.session_state.model.numeric_features,
                key="trend_feature"
            )
            
            # Create a range of values for the selected feature
            if feature in ['days_overdue', 'average_days_late']:
                min_val, max_val, step = 0, 90, 5
            elif feature == 'previous_late_payments':
                min_val, max_val, step = 0, 10, 1
            elif feature == 'lease_count':
                min_val, max_val, step = 1, 10, 1
            elif feature == 'lease_amount':
                min_val, max_val, step = 500, 5000, 500
            elif feature == 'property_size_sqm':
                min_val, max_val, step = 50, 500, 50
            else:
                min_val, max_val, step = 0, 100, 10
                
            range_values = list(range(min_val, max_val + 1, step))
            
            # Create dataframe with varying feature values
            trend_data = []
            
            # Get the median values for other features
            base_record = {}
            for f in st.session_state.model.features:
                if f in st.session_state.model.numeric_features:
                    if f != feature:
                        base_record[f] = st.session_state.historical_data[f].median()
                else:  # Categorical feature
                    # Use most common value
                    base_record[f] = st.session_state.historical_data[f].mode()[0]
            
            # Create records with varying feature values
            for val in range_values:
                record = base_record.copy()
                record[feature] = val
                trend_data.append(record)
            
            # Create dataframe and make predictions
            trend_df = pd.DataFrame(trend_data)
            trend_results = st.session_state.model.predict(trend_df)
            
            # Plot trend
            fig = px.line(
                trend_results, 
                x=feature, 
                y="payment_probability",
                title=f"Effect of {feature} on Payment Probability",
                markers=True
            )
            
            # Add reference line at benchmark
            if 'benchmark' in st.session_state:
                benchmark = st.session_state.benchmark
                fig.add_hline(
                    y=benchmark, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Benchmark ({benchmark:.0%})"
                )
            
            # Format axes
            fig.update_layout(
                xaxis_title=feature,
                yaxis_title="Payment Probability",
                yaxis=dict(tickformat=".0%"),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Identify critical points
            critical_points = []
            
            # Find where trend crosses benchmark if available
            if 'benchmark' in st.session_state:
                benchmark = st.session_state.benchmark
                crosses = []
                for i in range(1, len(trend_results)):
                    prev = trend_results['payment_probability'].iloc[i-1]
                    curr = trend_results['payment_probability'].iloc[i]
                    if (prev < benchmark and curr >= benchmark) or (prev >= benchmark and curr < benchmark):
                        crosses.append(i)
                
                for idx in crosses:
                    critical_points.append({
                        'type': 'Benchmark Cross',
                        'value': trend_results[feature].iloc[idx],
                        'probability': trend_results['payment_probability'].iloc[idx]
                    })
            
            # Find max and min points
            max_idx = trend_results['payment_probability'].idxmax()
            min_idx = trend_results['payment_probability'].idxmin()
            
            critical_points.append({
                'type': 'Maximum',
                'value': trend_results[feature].iloc[max_idx],
                'probability': trend_results['payment_probability'].iloc[max_idx]
            })
            
            critical_points.append({
                'type': 'Minimum',
                'value': trend_results[feature].iloc[min_idx],
                'probability': trend_results['payment_probability'].iloc[min_idx]
            })
            
            # Display critical points
            if critical_points:
                st.subheader("Critical Points")
                cp_df = pd.DataFrame(critical_points)
                cp_df['probability'] = cp_df['probability'].apply(lambda x: f"{x:.2%}")
                st.dataframe(cp_df, use_container_width=True)
                
                # Insights based on critical points
                st.subheader("Key Insights")
                if feature in ['days_overdue', 'previous_late_payments', 'average_days_late']:
                    st.markdown(f"**Insight**: Higher values of {feature} generally result in lower payment probability.")
                elif feature in ['lease_count', 'lease_amount']:
                    st.markdown(f"**Insight**: Higher values of {feature} appear to correlate with {'higher' if trend_results['payment_probability'].iloc[-1] > trend_results['payment_probability'].iloc[0] else 'lower'} payment probability.")
                
                # Risk threshold suggestions if available
                if 'benchmark' in st.session_state and crosses:
                    st.markdown(f"**Suggested Threshold**: {feature} values {'below' if trend_results['payment_probability'].iloc[crosses[0]] > trend_results['payment_probability'].iloc[crosses[0]-1] else 'above'} {trend_results[feature].iloc[crosses[0]]} are more likely to meet or exceed the benchmark payment probability.")
        else:
            st.info("Please train the model first to enable trend analysis.")
    
    with insight_tabs[2]:  # Geographic Insights
        st.subheader("Geographic Payment Analysis")
        
        if not st.session_state.historical_data.empty and 'country' in st.session_state.historical_data.columns:
            # Group by country and region
            if 'was_paid' in st.session_state.historical_data.columns:
                # Group by country
                country_data = st.session_state.historical_data.groupby('country').agg({
                    'was_paid': ['mean', 'count']
                }).reset_index()
                
                country_data.columns = ['Country', 'Payment Rate', 'Count']
                country_data['Payment Rate'] = country_data['Payment Rate'].round(4)
                
                # Sort by payment rate
                country_data = country_data.sort_values('Payment Rate', ascending=False)
                
                # Bar chart for payment rate by country
                fig = px.bar(
                    country_data,
                    x='Country',
                    y='Payment Rate',
                    color='Payment Rate',
                    text_auto='.1%',
                    title="Historical Payment Rate by Country",
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Payment Rate",
                    yaxis=dict(tickformat=".0%"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If region is available, show region breakdown
                if 'region' in st.session_state.historical_data.columns:
                    st.subheader("Regional Analysis")
                    
                    # Select country to analyze
                    country_list = st.session_state.historical_data['country'].unique().tolist()
                    selected_country = st.selectbox("Select country to analyze:", country_list)
                    
                    # Filter for selected country
                    country_regions = st.session_state.historical_data[st.session_state.historical_data['country'] == selected_country]
                    
                    # Group by region
                    region_data = country_regions.groupby('region').agg({
                        'was_paid': ['mean', 'count']
                    }).reset_index()
                    
                    region_data.columns = ['Region', 'Payment Rate', 'Count']
                    region_data['Payment Rate'] = region_data['Payment Rate'].round(4)
                    
                    # Create regional visualization
                    cols = st.columns(2)
                    
                    with cols[0]:
                        # Bar chart for payment rate by region
                        fig = px.bar(
                            region_data,
                            x='Region',
                            y='Payment Rate',
                            text_auto='.1%',
                            title=f"Payment Rate by Region in {selected_country}",
                            color='Payment Rate',
                            color_continuous_scale='viridis'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Region",
                            yaxis_title="Payment Rate",
                            yaxis=dict(tickformat=".0%"),
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with cols[1]:
                        # Pie chart for record count by region
                        fig = px.pie(
                            region_data,
                            names='Region',
                            values='Count',
                            title=f"Record Distribution by Region in {selected_country}"
                        )
                        
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical comparison
                    st.subheader("Regional Comparison")
                    st.dataframe(region_data, use_container_width=True)
                    
                    # Calculate country average for comparison
                    country_avg = country_regions['was_paid'].mean()
                    
                    # Identify above/below average regions
                    above_avg = region_data[region_data['Payment Rate'] > country_avg]['Region'].tolist()
                    below_avg = region_data[region_data['Payment Rate'] < country_avg]['Region'].tolist()
                    
                    if above_avg:
                        st.markdown(f"**Above Country Average**: {', '.join(above_avg)}")
                    if below_avg:
                        st.markdown(f"**Below Country Average**: {', '.join(below_avg)}")
            else:
                st.info("For geographic insights on historical data, target variable 'was_paid' is required.")
                
                # Show sample visualization with dummy data
                st.subheader("Sample Geographic Analysis (with dummy data)")
                
                dummy_data = pd.DataFrame({
                    'Country': ['France', 'Germany', 'Spain', 'Italy', 'UK'],
                    'Payment Rate': [0.82, 0.78, 0.75, 0.71, 0.85],
                    'Count': [120, 95, 80, 65, 110]
                })
                
                fig = px.bar(
                    dummy_data,
                    x='Country',
                    y='Payment Rate',
                    color='Payment Rate',
                    text_auto='.1%',
                    title="Sample: Payment Rate by Country",
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Payment Rate",
                    yaxis=dict(tickformat=".0%"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geographic data not available in the current dataset.")
    
    with insight_tabs[3]:  # Payment Patterns
        st.subheader("Payment Pattern Analysis")
        
        st.markdown("""This analysis reveals patterns in payment behavior based on multiple factors. 
        Understand how combinations of features affect payment likelihood.""")
        
        if not st.session_state.historical_data.empty:
            # Allow selection of multiple factors to analyze together
            if st.session_state.model.trained:
                available_features = st.session_state.model.features
                
                analysis_col1, analysis_col2 = st.columns(2)
                with analysis_col1:
                    primary_factor = st.selectbox(
                        "Primary Factor:",
                        options=available_features,
                        key="payment_primary_factor"
                    )
                
                with analysis_col2:
                    secondary_factor = st.selectbox(
                        "Secondary Factor (optional):",
                        options=["None"] + [f for f in available_features if f != primary_factor],
                        key="payment_secondary_factor"
                    )
                
                # Check if factors are categorical or numerical
                primary_is_cat = st.session_state.historical_data[primary_factor].dtype == 'object' or \
                                st.session_state.historical_data[primary_factor].nunique() < 10
                                
                if secondary_factor != "None":
                    secondary_is_cat = st.session_state.historical_data[secondary_factor].dtype == 'object' or \
                                    st.session_state.historical_data[secondary_factor].nunique() < 10
                
                # Determine analysis type based on factor types
                if 'was_paid' in st.session_state.historical_data.columns:
                    if primary_is_cat:
                        if secondary_factor != "None" and secondary_is_cat:
                            # Both categorical - group by both
                            st.subheader(f"Payment Patterns by {primary_factor} and {secondary_factor}")
                            
                            grouped = st.session_state.historical_data.groupby([primary_factor, secondary_factor])['was_paid'].agg(['mean', 'count']).reset_index()
                            grouped.columns = [primary_factor, secondary_factor, 'Payment Rate', 'Count']
                            
                            # Create heatmap
                            pivot_table = grouped.pivot(index=primary_factor, columns=secondary_factor, values='Payment Rate')
                            
                            fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
                            sns.heatmap(
                                pivot_table, 
                                annot=True, 
                                fmt=".2f",
                                cmap='viridis', 
                                linewidths=.5, 
                                ax=ax
                            )
                            plt.title(f'Payment Rate by {primary_factor} and {secondary_factor}', fontsize=15)
                            st.pyplot(fig)
                            
                            # Table view
                            st.dataframe(grouped, use_container_width=True)
                            
                        elif secondary_factor != "None":
                            # Primary categorical, secondary numerical
                            st.subheader(f"Payment Patterns by {primary_factor} across {secondary_factor} values")
                            
                            # Create box plot
                            fig = px.box(
                                st.session_state.historical_data,
                                x=primary_factor,
                                y=secondary_factor,
                                color='was_paid',
                                title=f"{secondary_factor} Distribution by {primary_factor} and Payment Status",
                                color_discrete_map={0: 'red', 1: 'green'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            # Only primary (categorical)
                            st.subheader(f"Payment Patterns by {primary_factor}")
                            
                            grouped = st.session_state.historical_data.groupby(primary_factor)['was_paid'].agg(['mean', 'count']).reset_index()
                            grouped.columns = [primary_factor, 'Payment Rate', 'Count']
                            
                            # Create bar chart
                            fig = px.bar(
                                grouped,
                                x=primary_factor,
                                y='Payment Rate',
                                text_auto='.1%',
                                title=f"Payment Rate by {primary_factor}",
                                color='Payment Rate',
                                color_continuous_scale='viridis'
                            )
                            
                            fig.update_layout(
                                yaxis=dict(tickformat=".0%"),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Table view
                            st.dataframe(grouped, use_container_width=True)
                    else:
                        # Primary numerical
                        if secondary_factor != "None" and secondary_is_cat:
                            # Primary numerical, secondary categorical
                            st.subheader(f"Payment Patterns by {secondary_factor} across {primary_factor} values")
                            
                            # Create violin plot
                            fig = px.violin(
                                st.session_state.historical_data,
                                x=secondary_factor,
                                y=primary_factor,
                                color='was_paid',
                                box=True,
                                title=f"{primary_factor} Distribution by {secondary_factor} and Payment Status",
                                color_discrete_map={0: 'red', 1: 'green'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif secondary_factor != "None":
                            # Both numerical
                            st.subheader(f"Payment Patterns by {primary_factor} and {secondary_factor}")
                            
                            # Create scatter plot
                            fig = px.scatter(
                                st.session_state.historical_data,
                                x=primary_factor,
                                y=secondary_factor,
                                color='was_paid',
                                title=f"Relationship between {primary_factor} and {secondary_factor} by Payment Status",
                                color_discrete_map={0: 'red', 1: 'green'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            # Only primary (numerical)
                            st.subheader(f"Payment Patterns by {primary_factor}")
                            
                            # Create histogram
                            fig = px.histogram(
                                st.session_state.historical_data,
                                x=primary_factor,
                                color='was_paid',
                                barmode='overlay',
                                title=f"{primary_factor} Distribution by Payment Status",
                                color_discrete_map={0: 'red', 1: 'green'},
                                opacity=0.7,
                                histnorm='probability density'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate mean/median by payment status
                            stats = st.session_state.historical_data.groupby('was_paid')[primary_factor].agg(['mean', 'median']).reset_index()
                            stats.columns = ['Payment Status', 'Mean', 'Median']
                            stats['Payment Status'] = stats['Payment Status'].map({0: 'Not Paid', 1: 'Paid'})
                            
                            st.dataframe(stats, use_container_width=True)
                else:
                    st.info("For payment pattern analysis on historical data, target variable 'was_paid' is required.")
                    
                    # Show sample visualization with dummy data
                    st.subheader("Sample Payment Pattern Analysis (with dummy data)")
                    
                    if primary_is_cat:
                        dummy_data = pd.DataFrame({
                            'Category': ['A', 'B', 'C', 'D', 'E'],
                            'Payment Rate': [0.82, 0.68, 0.75, 0.91, 0.55],
                            'Count': [120, 95, 80, 65, 110]
                        })
                        
                        fig = px.bar(
                            dummy_data,
                            x='Category',
                            y='Payment Rate',
                            text_auto='.1%',
                            title=f"Sample: Payment Rate by {primary_factor}",
                            color='Payment Rate',
                            color_continuous_scale='viridis'
                        )
                        
                        fig.update_layout(
                            xaxis_title=primary_factor,
                            yaxis_title="Payment Rate",
                            yaxis=dict(tickformat=".0%"),
                            height=400
                        )
                    else:
                        # Generate dummy data for numerical feature
                        x = np.linspace(0, 100, 100)
                        y_paid = np.exp(-(x-70)**2/1000) + 0.1
                        y_unpaid = np.exp(-(x-30)**2/1000) + 0.05
                        
                        dummy_data_paid = pd.DataFrame({
                            'Value': x,
                            'Density': y_paid,
                            'Payment': ['Paid'] * 100
                        })
                        
                        dummy_data_unpaid = pd.DataFrame({
                            'Value': x,
                            'Density': y_unpaid,
                            'Payment': ['Not Paid'] * 100
                        })
                        
                        dummy_data = pd.concat([dummy_data_paid, dummy_data_unpaid])
                        
                        fig = px.line(
                            dummy_data,
                            x='Value',
                            y='Density',
                            color='Payment',
                            title=f"Sample: {primary_factor} Distribution by Payment Status",
                            color_discrete_map={'Paid': 'green', 'Not Paid': 'red'}
                        )
                        
                        fig.update_layout(
                            xaxis_title=primary_factor,
                            height=400
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please train the model first to enable pattern analysis.")
        else:
            st.info("No data available for pattern analysis.")

# Add a footer
st.markdown("---")
st.markdown("© 2025 Nomos AI - All Rights Reserved")

# Add a "Stop App" button at the bottom of the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Stop Application", use_container_width=True):
    st.stop()
