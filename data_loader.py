import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utilities

@st.cache_data
def load_data():
    """Load and cache the concrete dataset"""
    try:
        df = pd.read_csv('Concrete_Data.csv')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def show_data_exploration():
    """Display data exploration section"""
    st.header("Data Exploration")
    
    # Load data
    df = load_data()
    
    # Display tabs for different data views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Preview", "Statistics", "Data Quality"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Missing Values:** {df.isna().sum().sum()}")
            st.write(f"**Duplicates:** {df.duplicated().sum()}")
        
        with col2:
            # Show column information
            column_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum()
            })
            st.write("**Column Information:**")
            st.dataframe(column_info)
        
        # Display column name mapping (original to simple)
        st.subheader("Column Name Mapping")
        name_mapping = utilities.get_column_name_mapping()
        mapping_df = pd.DataFrame({
            'Original Name': name_mapping.keys(),
            'Simple Name': name_mapping.values()
        })
        st.dataframe(mapping_df)
    
    with tab2:
        st.subheader("Data Preview")
        
        view_option = st.radio(
            "Select view option:",
            ["Simple Column Names", "Original Column Names"],
            horizontal=True
        )
        
        if view_option == "Simple Column Names":
            # Rename columns for display
            display_df = df.rename(columns=utilities.get_column_name_mapping())
        else:
            display_df = df
        
        # Add row selection slider
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        
        # Add display options
        display_type = st.radio(
            "Select display type:",
            ["Head", "Tail", "Sample", "All"],
            horizontal=True
        )
        
        if display_type == "Head":
            st.dataframe(display_df.head(num_rows), use_container_width=True)
        elif display_type == "Tail":
            st.dataframe(display_df.tail(num_rows), use_container_width=True)
        elif display_type == "Sample":
            st.dataframe(display_df.sample(num_rows), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Display descriptive statistics
        if st.checkbox("Show with simple column names", value=True):
            stats_df = df.rename(columns=utilities.get_column_name_mapping()).describe().T
        else:
            stats_df = df.describe().T
        
        # Add additional statistics
        stats_df['Skewness'] = df.skew()
        stats_df['Kurtosis'] = df.kurtosis()
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Data distribution
        st.subheader("Data Distribution")
        
        view_option = st.radio(
            "Select column name style:",
            ["Simple Names", "Original Names"],
            horizontal=True,
            key="dist_names"
        )
        
        if view_option == "Simple Names":
            column_options = list(utilities.get_column_name_mapping().values())
            column_mapping = {v: k for k, v in utilities.get_column_name_mapping().items()}
            selected_column = st.selectbox("Select column for distribution", column_options)
            original_column = column_mapping[selected_column]
        else:
            column_options = list(df.columns)
            selected_column = st.selectbox("Select column for distribution", column_options)
            original_column = selected_column
        
        # Plot distribution
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(df[original_column], kde=True, ax=ax[0])
        ax[0].set_title(f'Distribution of {selected_column}')
        
        sns.boxplot(y=df[original_column], ax=ax[1])
        ax[1].set_title(f'Boxplot of {selected_column}')
        
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Data Quality Analysis")
        
        # Missing values
        st.write("### Missing Values")
        missing_values = df.isna().sum().reset_index()
        missing_values.columns = ['Column', 'Missing Count']
        missing_values['Missing Percentage'] = (missing_values['Missing Count'] / len(df)) * 100
        
        if missing_values['Missing Count'].sum() == 0:
            st.success("No missing values found in the dataset!")
        else:
            st.warning(f"Found {missing_values['Missing Count'].sum()} missing values")
            st.dataframe(missing_values)
        
        # Duplicates
        st.write("### Duplicate Rows")
        if df.duplicated().sum() == 0:
            st.success("No duplicate rows found in the dataset!")
        else:
            st.warning(f"Found {df.duplicated().sum()} duplicate rows")
            st.dataframe(df[df.duplicated(keep='first')])
        
        # Outliers
        st.write("### Outliers Analysis")
        
        if st.checkbox("Show with simple column names", value=True, key="outlier_names"):
            outlier_df = df.rename(columns=utilities.get_column_name_mapping())
            column_names = list(utilities.get_column_name_mapping().values())
            column_names.remove('Concrete Strength')  # Remove target variable
        else:
            outlier_df = df
            column_names = list(df.columns)
            column_names.remove('Concrete compressive strength(MPa, megapascals) ')  # Remove target variable
        
        # Calculate IQR for each feature
        Q1 = outlier_df[column_names].quantile(0.25)
        Q3 = outlier_df[column_names].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers in each column
        outlier_counts = ((outlier_df[column_names] < lower_bound) | 
                          (outlier_df[column_names] > upper_bound)).sum()
        
        # Display outlier counts
        outlier_summary = pd.DataFrame({
            'Column': outlier_counts.index,
            'Number of Outliers': outlier_counts.values,
            'Percentage': (outlier_counts.values / len(outlier_df) * 100).round(2)
        })
        
        st.dataframe(outlier_summary)
        
        # Option to view boxplot of all features
        if st.checkbox("Show boxplot of all features", value=True):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=outlier_df[column_names])
            plt.xticks(rotation=90)
            plt.title('Box Plot of All Features')
            st.pyplot(fig)
        
        # Option to view specific outliers
        if st.checkbox("Show outlier data points", value=False):
            outliers = ((outlier_df[column_names] < lower_bound) | 
                       (outlier_df[column_names] > upper_bound)).any(axis=1)
            
            if outliers.sum() > 0:
                st.dataframe(outlier_df[outliers])
            else:
                st.info("No outliers found based on the IQR method.")
