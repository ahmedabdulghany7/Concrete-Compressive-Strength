import streamlit as st
from streamlit_option_menu import option_menu

import data_loader
import exploratory_analysis
import model_training
import prediction
import utilities

# Page configuration
st.set_page_config(
    page_title="Concrete Strength Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom CSS
utilities.load_css()

# App title and description
st.title("Concrete Strength Analysis")
st.markdown("""
    <div class="subtitle">
        Interactive Analysis and Prediction of Concrete Compressive Strength
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home", 
            "Data Exploration", 
            "Exploratory Analysis", 
            "Model Training",
            "Prediction"
        ],
        icons=["house", "table", "graph-up", "gear", "calculator"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")

    st.markdown("### Supervisor:")
    st.markdown("Dr. Rasha Saleh")

    st.markdown("### Author")
    st.markdown("Ahmed Abdulghany")

# Main content
if selected == "Home":
    st.markdown("""
    ## Welcome to the Concrete Strength Analysis Application
    
    This application provides tools to analyze concrete mixture components and predict
    compressive strength using various machine learning models.
    
    ### Dataset Information
    
    The dataset contains information about concrete mixtures and their compressive strength:
    
    - **Cement**: Amount of cement in kg/m³
    - **Blast Furnace Slag**: Amount of blast furnace slag in kg/m³
    - **Fly Ash**: Amount of fly ash in kg/m³
    - **Water**: Amount of water in kg/m³
    - **Superplasticizer**: Amount of superplasticizer in kg/m³
    - **Coarse Aggregate**: Amount of coarse aggregate in kg/m³
    - **Fine Aggregate**: Amount of fine aggregate in kg/m³
    - **Age**: Age of the concrete in days
    - **Concrete Compressive Strength**: Target variable in MPa
    
    ### Application Features
    
    - **Data Exploration**: View and explore the dataset
    - **Exploratory Analysis**: Perform statistical analysis and visualizations
    - **Model Training**: Train and evaluate different machine learning models
    - **Prediction**: Make predictions using trained models
    
    To get started, select an option from the sidebar.
    """)
    
    st.markdown("### Key Dataset Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Load data for metrics
    df = data_loader.load_data()

    metric_color = 'color: #FF6347;'
    label_color = 'color: #4682B4;'

    with col1:
        st.markdown(f'<p style="{label_color}">Number of Samples</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="{metric_color}">{len(df)}</p>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<p style="{label_color}">Average Strength</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="{metric_color}">{df["Concrete compressive strength(MPa, megapascals) "].mean():.2f} MPa</p>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<p style="{label_color}">Max Strength</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="{metric_color}">{df["Concrete compressive strength(MPa, megapascals) "].max():.2f} MPa</p>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'<p style="{label_color}">Min Strength</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="{metric_color}">{df["Concrete compressive strength(MPa, megapascals) "].min():.2f} MPa</p>', unsafe_allow_html=True)


elif selected == "Data Exploration":
    data_loader.show_data_exploration()

elif selected == "Exploratory Analysis":
    exploratory_analysis.show_exploratory_analysis()

elif selected == "Model Training":
    model_training.show_model_training()

elif selected == "Prediction":
    prediction.show_prediction()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© Dr. Rasha Saleh | Ahmed Abdulghany </p>
</div>
""", unsafe_allow_html=True)
