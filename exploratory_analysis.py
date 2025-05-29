import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

import data_loader
import utilities

def show_exploratory_analysis():
    """Display exploratory analysis section"""
    st.header("Exploratory Data Analysis")
    
    # Load data
    df = data_loader.load_data()
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    
    with tab1:
        show_univariate_analysis(df)
    
    with tab2:
        show_bivariate_analysis(df)
    
    with tab3:
        show_multivariate_analysis(df)

def show_univariate_analysis(df):
    """Display univariate analysis"""
    st.subheader("Univariate Analysis")
    
    # Get column mapping
    column_mapping = utilities.get_column_name_mapping()
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    
    # Feature selection
    use_simple_names = st.checkbox("Use simple column names", value=True, key="uni_simple_names")
    
    if use_simple_names:
        column_options = list(column_mapping.values())
        selected_column = st.selectbox("Select feature for analysis", column_options)
        original_column = reverse_mapping[selected_column]
    else:
        column_options = list(df.columns)
        selected_column = st.selectbox("Select feature for analysis", column_options)
        original_column = selected_column
    
    # Display basic statistics
    st.write("### Basic Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats = df[original_column].describe()
        st.write(f"**Count:** {stats['count']}")
        st.write(f"**Mean:** {stats['mean']:.4f}")
        st.write(f"**Std Dev:** {stats['std']:.4f}")
        st.write(f"**Min:** {stats['min']:.4f}")
    
    with col2:
        st.write(f"**25%:** {stats['25%']:.4f}")
        st.write(f"**50% (Median):** {stats['50%']:.4f}")
        st.write(f"**75%:** {stats['75%']:.4f}")
        st.write(f"**Max:** {stats['max']:.4f}")
    
    # Additional statistics
    st.write("### Additional Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Skewness:** {skew(df[original_column].dropna()):.4f}")
        st.write(f"**Kurtosis:** {kurtosis(df[original_column].dropna(), fisher=True):.4f}")
    
    with col2:
        # Mode calculation
        mode_values = df[original_column].mode()
        if not mode_values.empty:
            mode_str = ', '.join([f"{val:.4f}" for val in mode_values])
            st.write(f"**Mode:** {mode_str}")
        
        # Range calculation
        data_range = df[original_column].max() - df[original_column].min()
        st.write(f"**Range:** {data_range:.4f}")
    
    # Display distribution visualizations
    st.write("### Distribution Visualizations")
    
    viz_type = st.radio(
        "Select visualization type:",
        ["All", "Histogram", "Boxplot", "Violin Plot", "KDE Plot"],
        horizontal=True
    )
    
    if viz_type == "All" or viz_type == "Histogram":
        st.write("#### Histogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[original_column], kde=True, bins=30, ax=ax)
        ax.set_title(f'Histogram of {selected_column}')
        ax.set_xlabel(selected_column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    if viz_type == "All" or viz_type == "Boxplot":
        st.write("#### Boxplot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=df[original_column], ax=ax)
        ax.set_title(f'Boxplot of {selected_column}')
        ax.set_ylabel(selected_column)
        st.pyplot(fig)
    
    if viz_type == "All" or viz_type == "Violin Plot":
        st.write("#### Violin Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(y=df[original_column], ax=ax)
        ax.set_title(f'Violin Plot of {selected_column}')
        ax.set_ylabel(selected_column)
        st.pyplot(fig)
    
    if viz_type == "All" or viz_type == "KDE Plot":
        st.write("#### KDE Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df[original_column], fill=True, ax=ax)
        ax.set_title(f'KDE Plot of {selected_column}')
        ax.set_xlabel(selected_column)
        ax.set_ylabel('Density')
        st.pyplot(fig)
    
    # Outlier detection
    st.write("### Outlier Detection")
    
    Q1 = df[original_column].quantile(0.25)
    Q3 = df[original_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[original_column] < lower_bound) | (df[original_column] > upper_bound)][original_column]
    
    st.write(f"**Number of outliers (IQR method):** {len(outliers)}")
    st.write(f"**Q1 (25%):** {Q1:.4f}")
    st.write(f"**Q3 (75%):** {Q3:.4f}")
    st.write(f"**IQR:** {IQR:.4f}")
    st.write(f"**Lower bound:** {lower_bound:.4f}")
    st.write(f"**Upper bound:** {upper_bound:.4f}")
    
    if len(outliers) > 0:
        st.write("**Outlier values:**")
        outlier_df = pd.DataFrame(outliers).reset_index()
        outlier_df.columns = ['Index', selected_column]
        st.dataframe(outlier_df)

def show_bivariate_analysis(df):
    """Display bivariate analysis"""
    st.subheader("Bivariate Analysis")
    
    # Get column mapping
    column_mapping = utilities.get_column_name_mapping()
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    
    # Use simple column names option
    use_simple_names = st.checkbox("Use simple column names", value=True, key="bi_simple_names")
    
    # Create column layout
    col1, col2 = st.columns(2)
    
    with col1:
        if use_simple_names:
            x_options = list(column_mapping.values())
            selected_x = st.selectbox("Select X-axis feature", x_options)
            original_x = reverse_mapping[selected_x]
        else:
            x_options = list(df.columns)
            selected_x = st.selectbox("Select X-axis feature", x_options)
            original_x = selected_x
    
    with col2:
        if use_simple_names:
            y_options = list(column_mapping.values())
            selected_y = st.selectbox("Select Y-axis feature", y_options, index=len(y_options)-1)
            original_y = reverse_mapping[selected_y]
        else:
            y_options = list(df.columns)
            selected_y = st.selectbox("Select Y-axis feature", y_options, index=len(y_options)-1)
            original_y = selected_y
    
    # Display relationship visualizations
    st.write("### Relationship Visualizations")
    
    viz_type = st.radio(
        "Select visualization type:",
        ["Scatter Plot", "Regression Plot", "Hexbin Plot", "Joint Plot"],
        horizontal=True
    )
    
    if viz_type == "Scatter Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df[original_x], y=df[original_y], ax=ax)
        ax.set_title(f'{selected_x} vs {selected_y}')
        ax.set_xlabel(selected_x)
        ax.set_ylabel(selected_y)
        st.pyplot(fig)
    
    elif viz_type == "Regression Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=df[original_x], y=df[original_y], scatter_kws={'alpha':0.5}, ax=ax)
        ax.set_title(f'{selected_x} vs {selected_y} (with Regression Line)')
        ax.set_xlabel(selected_x)
        ax.set_ylabel(selected_y)
        st.pyplot(fig)
    
    elif viz_type == "Hexbin Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = plt.hexbin(df[original_x], df[original_y], gridsize=20, cmap='Blues')
        plt.colorbar(hb)
        ax.set_title(f'Hexbin Plot: {selected_x} vs {selected_y}')
        ax.set_xlabel(selected_x)
        ax.set_ylabel(selected_y)
        st.pyplot(fig)
    
    elif viz_type == "Joint Plot":
        fig = plt.figure(figsize=(10, 8))
        g = sns.jointplot(
            x=df[original_x], 
            y=df[original_y],
            kind="reg",
            truncate=False,
            height=8,
            joint_kws={"scatter_kws": {"alpha": 0.5}}
        )
        g.fig.suptitle(f'Joint Plot: {selected_x} vs {selected_y}', y=1.02)
        st.pyplot(g.fig)
    
    # Display correlation analysis
    st.write("### Correlation Analysis")
    
    corr = df[[original_x, original_y]].corr().iloc[0, 1]
    
    # Display correlation coefficient with styled color
    if abs(corr) < 0.3:
        corr_strength = "weak"
        corr_color = "gray"
    elif abs(corr) < 0.7:
        corr_strength = "moderate"
        corr_color = "blue"
    else:
        corr_strength = "strong"
        corr_color = "green" if corr > 0 else "red"
    
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
        <p>Pearson correlation coefficient: 
        <span style="font-weight: bold; color: {corr_color};">{corr:.4f}</span> 
        ({corr_strength} {'positive' if corr > 0 else 'negative'} correlation)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show correlation significance
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**R²:** {corr**2:.4f}")
    
    with col2:
        # Simple interpretation of correlation
        if abs(corr) < 0.3:
            st.write("**Interpretation:** Little to no linear relationship")
        elif abs(corr) < 0.5:
            st.write("**Interpretation:** Weak linear relationship")
        elif abs(corr) < 0.7:
            st.write("**Interpretation:** Moderate linear relationship")
        elif abs(corr) < 0.9:
            st.write("**Interpretation:** Strong linear relationship")
        else:
            st.write("**Interpretation:** Very strong linear relationship")

def show_multivariate_analysis(df):
    """Display multivariate analysis"""
    st.subheader("Multivariate Analysis")
    
    # Analysis type selection
    analysis_type = st.radio(
        "Select analysis type:",
        ["Correlation Matrix", "Pairplot", "3D Scatter Plot", "Feature Importance"],
        horizontal=True
    )
    
    # Get column mapping
    column_mapping = utilities.get_column_name_mapping()
    
    if analysis_type == "Correlation Matrix":
        show_correlation_matrix(df, column_mapping)
    
    elif analysis_type == "Pairplot":
        show_pairplot(df, column_mapping)
    
    elif analysis_type == "3D Scatter Plot":
        show_3d_scatter(df, column_mapping)
    
    elif analysis_type == "Feature Importance":
        show_feature_importance(df, column_mapping)

def show_correlation_matrix(df, column_mapping):
    """Display correlation matrix"""
    use_simple_names = st.checkbox("Use simple column names", value=True, key="corr_simple_names")
    
    if use_simple_names:
        corr_df = df.rename(columns=column_mapping)
    else:
        corr_df = df
    
    corr_matrix = corr_df.corr()
    
    st.write("### Correlation Matrix")
    
    # Visualiztion options
    viz_type = st.radio(
        "Select visualization style:",
        ["Heatmap", "Clustermap"],
        horizontal=True
    )
    
    if viz_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(corr_matrix)
        cmap = st.selectbox(
            "Select color palette:",
            ["coolwarm", "viridis", "plasma", "Blues", "Greens", "Reds"]
        )
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            mask=mask,
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5, 
            fmt=".2f",
            ax=ax
        )
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    
    else:  # Clustermap
        fig = plt.figure(figsize=(10, 8))
        cmap = st.selectbox(
            "Select color palette:",
            ["coolwarm", "viridis", "plasma", "Blues", "Greens", "Reds"]
        )
        
        g = sns.clustermap(
            corr_matrix,
            annot=True,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=.5,
            fmt=".2f",
            figsize=(10, 10)
        )
        plt.title('Clustered Correlation Matrix', fontsize=16)
        st.pyplot(g)
    
    # Table of strongest correlations
    st.write("### Strongest Correlations")
    
    # Convert the correlation matrix to a long format
    corr_long = corr_matrix.unstack().reset_index()
    corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    # Filter out self-correlations and duplicates
    corr_long = corr_long[corr_long['Feature 1'] != corr_long['Feature 2']]
    corr_long['Pair'] = corr_long.apply(lambda row: tuple(sorted([row['Feature 1'], row['Feature 2']])), axis=1)
    corr_long = corr_long.drop_duplicates(subset=['Pair'])
    corr_long = corr_long.drop(columns=['Pair'])
    
    # Sort by absolute correlation value
    corr_long['Abs Correlation'] = corr_long['Correlation'].abs()
    corr_long = corr_long.sort_values('Abs Correlation', ascending=False).drop(columns=['Abs Correlation'])
    
    # Display the top N correlations
    top_n = st.slider("Number of top correlations to show", 5, 20, 10)
    st.dataframe(corr_long.head(top_n))

def show_pairplot(df, column_mapping):
    """Display pairplot for selected features"""
    st.write("### Feature Pairplot")
    
    # Use simple column names option
    use_simple_names = st.checkbox("Use simple column names", value=True, key="pair_simple_names")
    
    # Feature selection
    if use_simple_names:
        feature_options = list(column_mapping.values())
        reverse_mapping = {v: k for k, v in column_mapping.items()}
    else:
        feature_options = list(df.columns)
    
    # Allow selecting a subset of features
    with st.expander("Select features to include", expanded=True):
        default_features = [feature_options[-1]]  # Default to target variable
        selected_features = st.multiselect(
            "Features to include in the pairplot", 
            feature_options,
            default=default_features
        )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for the pairplot")
        return
    
    # Map back to original names if using simple names
    if use_simple_names:
        original_features = [reverse_mapping[feat] for feat in selected_features]
        plot_df = df[original_features].rename(columns=column_mapping)
    else:
        plot_df = df[selected_features]
    
    # Progress indicator
    with st.spinner("Generating pairplot... This may take a moment for large feature sets."):
        # Set hue to the target variable if it's among the selected features
        target_var = "Concrete Strength" if use_simple_names else "Concrete compressive strength(MPa, megapascals) "
        
        # Check if target variable is in selected features
        if target_var in selected_features:
            hue_var = target_var
            # Create discrete categories for the target variable
            plot_df[f"{hue_var}_cat"] = pd.qcut(plot_df[hue_var], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"])
            hue = f"{hue_var}_cat"
        else:
            hue = None
        
        # Generate and display the pairplot
        fig = sns.pairplot(
            plot_df,
            hue=hue,
            diag_kind="kde",
            height=2.5,
            corner=True,
            plot_kws={"alpha": 0.6}
        )
        
        if hue:
            fig.fig.suptitle('Pairplot with Strength Categories', y=1.02, fontsize=16)
        else:
            fig.fig.suptitle('Feature Pairplot', y=1.02, fontsize=16)
        
        st.pyplot(fig.fig)

def show_3d_scatter(df, column_mapping):
    """Display 3D scatter plot for selected features"""
    st.write("### 3D Scatter Plot")
    
    # Use simple column names option
    use_simple_names = st.checkbox("Use simple column names", value=True, key="3d_simple_names")
    
    # Feature selection
    if use_simple_names:
        feature_options = list(column_mapping.values())
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # Default to Cement, Water, and Strength
        default_x = "Cement"
        default_y = "Water"
        default_z = "Concrete Strength"
        
        selected_x = st.selectbox("X-axis feature", feature_options, index=feature_options.index(default_x))
        selected_y = st.selectbox("Y-axis feature", feature_options, index=feature_options.index(default_y))
        selected_z = st.selectbox("Z-axis feature", feature_options, index=feature_options.index(default_z))
        
        # Map back to original names
        original_x = reverse_mapping[selected_x]
        original_y = reverse_mapping[selected_y]
        original_z = reverse_mapping[selected_z]
    else:
        feature_options = list(df.columns)
        
        # Try to find sensible defaults
        try:
            default_x = "Cement (component 1)(kg in a m^3 mixture)"
            default_y = "Water  (component 4)(kg in a m^3 mixture)"
            default_z = "Concrete compressive strength(MPa, megapascals) "
            
            selected_x = st.selectbox("X-axis feature", feature_options, index=feature_options.index(default_x))
            selected_y = st.selectbox("Y-axis feature", feature_options, index=feature_options.index(default_y))
            selected_z = st.selectbox("Z-axis feature", feature_options, index=feature_options.index(default_z))
        except:
            selected_x = st.selectbox("X-axis feature", feature_options, index=0)
            selected_y = st.selectbox("Y-axis feature", feature_options, index=min(1, len(feature_options)-1))
            selected_z = st.selectbox("Z-axis feature", feature_options, index=min(2, len(feature_options)-1))
        
        original_x = selected_x
        original_y = selected_y
        original_z = selected_z
    
    # Check if all selected features are different
    if len(set([selected_x, selected_y, selected_z])) < 3:
        st.warning("Please select three different features for the 3D scatter plot")
        return
    
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add a color dimension based on the z variable (usually the target)
    scatter = ax.scatter(
        df[original_x],
        df[original_y],
        df[original_z],
        c=df[original_z],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add labels and title
    ax.set_xlabel(selected_x)
    ax.set_ylabel(selected_y)
    ax.set_zlabel(selected_z)
    ax.set_title(f'3D Scatter Plot: {selected_x} vs {selected_y} vs {selected_z}')
    
    # Add a color bar
    fig.colorbar(scatter, ax=ax, label=selected_z)
    
    # Improve the view angle
    ax.view_init(elev=30, azim=45)
    
    # Display the plot
    st.pyplot(fig)

def show_feature_importance(df, column_mapping):
    """Display feature importance using a random forest model"""
    from sklearn.ensemble import RandomForestRegressor
    
    st.write("### Feature Importance Analysis")
    
    # Set up feature and target variables
    X = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    y = df['Concrete compressive strength(MPa, megapascals) ']
    
    # Use simple column names option
    use_simple_names = st.checkbox("Use simple column names", value=True, key="fi_simple_names")
    
    # Train a Random Forest model
    with st.spinner("Training model for feature importance..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    if use_simple_names:
        # Remove target from column_mapping
        feature_mapping = {k: v for k, v in column_mapping.items() 
                         if k != 'Concrete compressive strength(MPa, megapascals) '}
        features = list(feature_mapping.values())
    else:
        features = X.columns.tolist()
    
    # Create a DataFrame for feature importances
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    ax.set_title('Feature Importance from Random Forest')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
    
    # Display the importance values
    st.write("### Feature Importance Values")
    st.dataframe(feature_importance)
    
    st.info("""
        Note: This feature importance is derived from a Random Forest model. 
        The higher the value, the more important the feature is for predicting concrete strength.
    """)