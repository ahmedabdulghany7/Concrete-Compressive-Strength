import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
        /* Page Layout */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Title and Headers */
        h1 {
            color: #FFFFFF !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: #FFFFFF !important;
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: #FFFFFF !important;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.8rem !important;
        }
        
        /* Subtitle style */
        .subtitle {
            color: #FFFFFF;
            font-size: 1.2rem;
            font-style: italic;
            margin-bottom: 2rem;
        }
        
        /* Card-like elements */
        .stMetric {
            background-color: #1E1E1E;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
            color: #FFFFFF !important;
        }
        
        .stMetric:hover {
            transform: translateY(-2px);
        }
        
        /* Labels and text */
        label, .stMarkdown, p, .stText {
            color: #FFFFFF !important;
        }
        
        /* Footer */
        .footer {
            color: #FFFFFF;
            text-align: center;
            padding-top: 1rem;
            font-size: 0.8rem;
        }
        
        /* Option menu styling */
        .nav-link {
            font-weight: 500 !important;
            color: #FFFFFF !important;
        }
        
        .nav-link.active {
            background-color: #2563EB !important;
            font-weight: 600 !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            padding-top: 0.5rem;
            font-weight: 500;
            color: #FFFFFF !important;
        }
        
        /* Highlight text */
        .highlight {
            background-color: #2563EB;
            padding: 0.2rem 0.5rem;
            border-radius: 0.2rem;
            font-weight: 500;
            color: #FFFFFF;
        }
        
        /* Animation for metrics */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .stMetric {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2563EB;
            color: white;
            font-weight: 500;
            border-radius: 0.3rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #1E40AF;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Progress bar */
        .stProgress .st-bo {
            background-color: #2563EB;
        }
        
        /* Metric labels */
        .css-1wivap2, .css-j5r0tf {
            color: #FFFFFF !important;
        }
        
        /* Sidebar text */
        .css-163ttbj, .css-10trblm {
            color: #FFFFFF !important;
        }
        
        /* Dataframe styling */
        .dataframe {
            color: #FFFFFF !important;
        }
        
        /* Select box labels */
        .css-81oif8 {
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

def get_column_name_mapping():
    """Return a mapping of original column names to simplified names"""
    return {
        'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Slag',
        'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly Ash',
        'Water  (component 4)(kg in a m^3 mixture)': 'Water',
        'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse Aggregate',
        'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine Aggregate',
        'Age (day)': 'Age',
        'Concrete compressive strength(MPa, megapascals) ': 'Concrete Strength'
    }

def set_plot_style():
    """Set the style for matplotlib plots"""
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set custom parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Set seaborn style as well
    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.2)
    
    # Create a custom color palette
    sns.set_palette(['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#8c5bd8', '#00A8E1'])

def plot_feature_distributions(df, figsize=(12, 8)):
    """Generate a grid of histograms for all features"""
    # Get column mapping
    column_mapping = get_column_name_mapping()
    
    # Get feature columns (excluding target)
    feature_cols = [col for col in df.columns if col != 'Concrete compressive strength(MPa, megapascals) ']
    
    # Calculate grid dimensions
    n_features = len(feature_cols)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(feature_cols):
        if i < len(axes):
            # Use simplified name for display
            simple_name = column_mapping.get(feature, feature)
            
            # Plot histogram with KDE
            sns.histplot(df[feature], kde=True, ax=axes[i])
            axes[i].set_title(simple_name)
            axes[i].set_xlabel('')
    
    # Hide any unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_feature_importances(model, feature_names):
    """Plot feature importances for a tree-based model"""
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Get column mapping
    column_mapping = get_column_name_mapping()
    
    # Get simplified feature names
    simple_names = [column_mapping.get(name, name) for name in feature_names]
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': simple_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('')
    
    return fig

def create_download_link(df, title, filename):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'
    return href