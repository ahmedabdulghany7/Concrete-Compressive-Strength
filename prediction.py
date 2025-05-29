import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler

import data_loader
import utilities

def show_prediction():
    """Display prediction interface"""
    st.header("Concrete Strength Prediction")
    
    # Check for saved models
    models = load_saved_models()
    
    if not models:
        st.warning("""
            No saved models found. Please go to the 'Model Training' tab to train and save models.
            
            Alternatively, you can use the 'Quick Prediction' tab which uses pre-trained models.
        """)
        
        show_quick_prediction()
        return
    
    # Create tabs for different prediction types
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "What-If Analysis"])
    
    with tab1:
        show_single_prediction(models)
    
    with tab2:
        show_batch_prediction(models)
    
    with tab3:
        show_what_if_analysis(models)

def load_saved_models():
    """Load saved models from the models directory"""
    models = {}
    
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            # Extract model name from filename
            model_name = model_file.replace('_model.joblib', '').replace('_', ' ').title()
            
            try:
                # Load the model
                model = joblib.load(f"models/{model_file}")
                models[model_name] = model
            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")
    
    return models

def show_quick_prediction():
    """Show quick prediction interface using on-the-fly trained models"""
    st.subheader("Quick Prediction")
    
    # Load data
    df = data_loader.load_data()
    X = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    y = df['Concrete compressive strength(MPa, megapascals) ']
    
    # Train a simple random forest model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("Training a simple Random Forest model..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    st.success("Model trained! You can now make predictions.")
    
    # Create prediction form
    st.write("### Input Concrete Mixture Components")
    
    # Get column mapping for display
    column_mapping = utilities.get_column_name_mapping()
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    # Dictionary to store input values
    inputs = {}
    
    # Populate the form with input fields
    with col1:
        inputs['Cement'] = st.number_input("Cement (kg/m³)", min_value=0.0, max_value=1000.0, value=300.0, step=10.0)
        inputs['Slag'] = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=500.0, value=0.0, step=10.0)
        inputs['Fly Ash'] = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=500.0, value=0.0, step=10.0)
        inputs['Water'] = st.number_input("Water (kg/m³)", min_value=100.0, max_value=300.0, value=180.0, step=5.0)
    
    with col2:
        inputs['Superplasticizer'] = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
        inputs['Coarse Aggregate'] = st.number_input("Coarse Aggregate (kg/m³)", min_value=500.0, max_value=1500.0, value=1000.0, step=20.0)
        inputs['Fine Aggregate'] = st.number_input("Fine Aggregate (kg/m³)", min_value=500.0, max_value=1500.0, value=800.0, step=20.0)
        inputs['Age'] = st.number_input("Age (days)", min_value=1, max_value=365, value=28, step=1)
    
    # Make prediction button
    if st.button("Predict Strength"):
        # Map inputs to original column names
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        input_data = {}
        
        for simple_name, value in inputs.items():
            original_name = reverse_mapping[simple_name]
            input_data[original_name] = value
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[X.columns]  # Ensure same column order as training data
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display prediction with gauge chart
        st.write("### Prediction Result")
        
        # Create a gauge-like display
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define a color gradient for the gauge
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Set up the gauge
        strength_range = np.linspace(0, 100, 1000)
        gauge_colors = cm.RdYlGn(np.linspace(0, 1, len(strength_range)))
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_gauge', gauge_colors)
        
        # Plot the gauge background
        plt.barh(0, 100, height=0.6, color='lightgrey', alpha=0.3)
        
        # Plot the prediction value
        plt.barh(0, min(prediction, 100), height=0.6, color=cmap(prediction/100))
        
        # Add the prediction value text
        plt.text(min(prediction + 5, 95), 0, f"{prediction:.2f} MPa", 
                 va='center', ha='left', fontsize=16, fontweight='bold')
        
        # Customize the plot
        plt.xlim(0, 100)
        plt.ylim(-0.5, 0.5)
        plt.title("Predicted Concrete Strength", fontsize=16)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.yticks([])
        
        # Add custom x-ticks
        plt.xticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100+'])
        plt.xlabel("Strength (MPa)")
        
        st.pyplot(fig)
        
        # Interpret the prediction
        interpret_strength_prediction(prediction)

def show_single_prediction(models):
    """Display interface for single prediction"""
    st.subheader("Single Concrete Sample Prediction")
    
    # Load data for column information
    df = data_loader.load_data()
    X_columns = df.drop(columns=['Concrete compressive strength(MPa, megapascals) ']).columns
    
    # Model selection
    model_name = st.selectbox("Select model for prediction:", list(models.keys()))
    model = models[model_name]
    
    # Get column mapping for display
    column_mapping = utilities.get_column_name_mapping()
    
    # Create user input form
    st.write("### Enter Concrete Mixture Components")
    
    # Provide example data option
    show_example = st.checkbox("Use example data", value=False)
    
    if show_example:
        # Get a sample from the dataset
        sample = df.sample(1)
        sample_inputs = sample.drop(columns=['Concrete compressive strength(MPa, megapascals) ']).iloc[0].to_dict()
        actual_strength = sample['Concrete compressive strength(MPa, megapascals) '].iloc[0]
        
        st.info(f"Using example data with actual strength: {actual_strength:.2f} MPa")
    else:
        sample_inputs = None
        actual_strength = None
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    # Dictionary to store input values
    inputs = {}
    
    # Determine ranges for each input
    input_ranges = {}
    for col in X_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        input_ranges[col] = (min_val, max_val, mean_val)
    
    # Populate the form with input fields
    with col1:
        for i, col in enumerate(X_columns[:4]):
            simple_name = column_mapping.get(col, col)
            min_val, max_val, mean_val = input_ranges[col]
            
            # Set default value from example or mean
            default_val = sample_inputs[col] if sample_inputs else mean_val
            
            # For age, use integer input
            if 'Age' in col:
                inputs[col] = st.number_input(
                    f"{simple_name}", 
                    min_value=int(min_val), 
                    max_value=int(max_val),
                    value=int(default_val),
                    step=1
                )
            else:
                inputs[col] = st.number_input(
                    f"{simple_name}", 
                    min_value=float(min_val), 
                    max_value=float(max_val),
                    value=float(default_val),
                    step=float((max_val - min_val) / 100)
                )
    
    with col2:
        for i, col in enumerate(X_columns[4:]):
            simple_name = column_mapping.get(col, col)
            min_val, max_val, mean_val = input_ranges[col]
            
            # Set default value from example or mean
            default_val = sample_inputs[col] if sample_inputs else mean_val
            
            # For age, use integer input
            if 'Age' in col:
                inputs[col] = st.number_input(
                    f"{simple_name}", 
                    min_value=int(min_val), 
                    max_value=int(max_val),
                    value=int(default_val),
                    step=1
                )
            else:
                inputs[col] = st.number_input(
                    f"{simple_name}", 
                    min_value=float(min_val), 
                    max_value=float(max_val),
                    value=float(default_val),
                    step=float((max_val - min_val) / 100)
                )
    
    # Make prediction button
    if st.button("Predict Strength"):
        # Create input DataFrame with proper column order
        input_array = np.array([list(inputs.values())])
        input_df = pd.DataFrame(input_array, columns=list(inputs.keys()))
        
        # Standardize if needed (for now, assuming model expects raw inputs)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display prediction
        st.write("### Prediction Result")
        
        # Create a gauge chart for the prediction
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define a color gradient for the gauge
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Set up the gauge
        strength_range = np.linspace(0, 100, 1000)
        gauge_colors = cm.RdYlGn(np.linspace(0, 1, len(strength_range)))
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_gauge', gauge_colors)
        
        # Plot the gauge background
        plt.barh(0, 100, height=0.6, color='lightgrey', alpha=0.3)
        
        # Plot the prediction value
        plt.barh(0, min(prediction, 100), height=0.6, color=cmap(prediction/100))
        
        # Add the prediction value text
        plt.text(min(prediction + 5, 95), 0, f"{prediction:.2f} MPa", 
                 va='center', ha='left', fontsize=16, fontweight='bold')
        
        # If we have actual strength, show it as a vertical line
        if actual_strength is not None:
            plt.axvline(x=actual_strength, color='black', linestyle='--', linewidth=2)
            plt.text(actual_strength, 0.3, f"Actual: {actual_strength:.2f}", 
                     ha='center', va='bottom', rotation=90, fontsize=12)
        
        # Customize the plot
        plt.xlim(0, 100)
        plt.ylim(-0.5, 0.5)
        plt.title("Predicted Concrete Strength", fontsize=16)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.yticks([])
        
        # Add custom x-ticks
        plt.xticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100+'])
        plt.xlabel("Strength (MPa)")
        
        st.pyplot(fig)
        
        # Show error if actual strength is available
        if actual_strength is not None:
            error = prediction - actual_strength
            error_pct = (error / actual_strength) * 100
            
            st.write(f"**Absolute Error:** {abs(error):.2f} MPa")
            st.write(f"**Relative Error:** {abs(error_pct):.2f}%")
        
        # Interpret the prediction
        interpret_strength_prediction(prediction)

def show_batch_prediction(models):
    """Display interface for batch prediction"""
    st.subheader("Batch Prediction")
    
    # Model selection
    model_name = st.selectbox("Select model for batch prediction:", list(models.keys()))
    model = models[model_name]
    
    # File upload
    st.write("### Upload a CSV or Excel file with concrete mixture data")
    
    file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
    
    if file is not None:
        try:
            # Read the file
            if file.name.endswith('.csv'):
                input_df = pd.read_csv(file)
            else:
                input_df = pd.read_excel(file)
            
            # Display the uploaded data
            st.write("### Uploaded Data")
            st.dataframe(input_df)
            
            # Validate columns
            df = data_loader.load_data()
            required_columns = df.drop(columns=['Concrete compressive strength(MPa, megapascals) ']).columns.tolist()
            
            # Check if all required columns are present (ignoring case)
            input_cols_lower = [col.lower() for col in input_df.columns]
            required_cols_lower = [col.lower() for col in required_columns]
            
            missing_columns = []
            for req_col, req_col_lower in zip(required_columns, required_cols_lower):
                if req_col_lower not in input_cols_lower and req_col not in input_df.columns:
                    missing_columns.append(req_col)
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your file contains all the necessary components for concrete strength prediction.")
                return
            
            # Map column names if needed
            column_mapping = {}
            for req_col, req_col_lower in zip(required_columns, required_cols_lower):
                if req_col in input_df.columns:
                    column_mapping[req_col] = req_col
                else:
                    # Find the column with matching lowercase name
                    matching_cols = [col for col, col_lower in zip(input_df.columns, input_cols_lower) 
                                    if col_lower == req_col_lower]
                    if matching_cols:
                        column_mapping[matching_cols[0]] = req_col
            
            # Rename columns if needed
            if column_mapping:
                input_df_renamed = input_df.rename(columns=column_mapping)
                input_df_for_pred = input_df_renamed[required_columns]
            else:
                input_df_for_pred = input_df[required_columns]
            
            # Make predictions
            if st.button("Make Batch Predictions"):
                with st.spinner("Making predictions..."):
                    predictions = model.predict(input_df_for_pred)
                
                # Add predictions to the input data
                result_df = input_df.copy()
                result_df['Predicted Strength (MPa)'] = predictions
                
                # Display results
                st.write("### Prediction Results")
                st.dataframe(result_df)
                
                # Visualize predictions
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.hist(predictions, bins=20, alpha=0.7, color='#2196F3')
                plt.axvline(predictions.mean(), color='red', linestyle='--', linewidth=2)
                plt.title("Distribution of Predicted Strengths")
                plt.xlabel("Predicted Strength (MPa)")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Summary statistics
                st.write("### Prediction Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Strength", f"{predictions.mean():.2f} MPa")
                
                with col2:
                    st.metric("Minimum Strength", f"{predictions.min():.2f} MPa")
                
                with col3:
                    st.metric("Maximum Strength", f"{predictions.max():.2f} MPa")
                
                # Download option for predictions
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="concrete_strength_predictions.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        # Show example template
        st.info("No file uploaded. Please upload a CSV or Excel file with the required concrete mixture components.")
        
        # Show example format
        st.write("### Example Format")
        
        # Get column mapping for display
        column_mapping = utilities.get_column_name_mapping()
        
        # Create example DataFrame with simplified column names
        simple_columns = [col for col in column_mapping.values() if col != "Concrete Strength"]
        example_df = pd.DataFrame(columns=simple_columns)
        
        # Add a single row with example values
        example_df.loc[0] = [300, 0, 0, 180, 5, 1000, 800, 28]
        
        st.dataframe(example_df)
        
        # Provide downloadable template
        csv = example_df.to_csv(index=False)
        st.download_button(
            label="Download Template CSV",
            data=csv,
            file_name="concrete_mixture_template.csv",
            mime="text/csv"
        )

def show_what_if_analysis(models):
    """Display interface for what-if analysis"""
    st.subheader("What-If Analysis")
    
    # Load data for column information
    df = data_loader.load_data()
    X_columns = df.drop(columns=['Concrete compressive strength(MPa, megapascals) ']).columns
    
    # Model selection
    model_name = st.selectbox("Select model for analysis:", list(models.keys()))
    model = models[model_name]
    
    # Get column mapping for display
    column_mapping = utilities.get_column_name_mapping()
    
    # Base values for concrete mixture
    st.write("### Base Concrete Mixture")
    
    # Dictionary to store base input values
    base_inputs = {}
    
    # Use example data option
    show_example = st.checkbox("Use example data as base", value=True)
    
    if show_example:
        # Get average values from the dataset
        for col in X_columns:
            base_inputs[col] = df[col].mean()
        
        # Display the base mixture
        base_df = pd.DataFrame([base_inputs])
        base_df_display = base_df.rename(columns=column_mapping)
        st.dataframe(base_df_display)
    else:
        # Manual input for base mixture
        col1, col2 = st.columns(2)
        
        # Determine ranges for each input
        input_ranges = {}
        for col in X_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            input_ranges[col] = (min_val, max_val, mean_val)
        
        # Populate the form with input fields
        with col1:
            for i, col in enumerate(X_columns[:4]):
                simple_name = column_mapping.get(col, col)
                min_val, max_val, mean_val = input_ranges[col]
                
                # For age, use integer input
                if 'Age' in col:
                    base_inputs[col] = st.number_input(
                        f"{simple_name}", 
                        min_value=int(min_val), 
                        max_value=int(max_val),
                        value=int(mean_val),
                        step=1,
                        key=f"base_{col}"
                    )
                else:
                    base_inputs[col] = st.number_input(
                        f"{simple_name}", 
                        min_value=float(min_val), 
                        max_value=float(max_val),
                        value=float(mean_val),
                        step=float((max_val - min_val) / 100),
                        key=f"base_{col}"
                    )
        
        with col2:
            for i, col in enumerate(X_columns[4:]):
                simple_name = column_mapping.get(col, col)
                min_val, max_val, mean_val = input_ranges[col]
                
                # For age, use integer input
                if 'Age' in col:
                    base_inputs[col] = st.number_input(
                        f"{simple_name}", 
                        min_value=int(min_val), 
                        max_value=int(max_val),
                        value=int(mean_val),
                        step=1,
                        key=f"base_{col}"
                    )
                else:
                    base_inputs[col] = st.number_input(
                        f"{simple_name}", 
                        min_value=float(min_val), 
                        max_value=float(max_val),
                        value=float(mean_val),
                        step=float((max_val - min_val) / 100),
                        key=f"base_{col}"
                    )
    
    # Get base prediction
    base_array = np.array([list(base_inputs.values())])
    base_df = pd.DataFrame(base_array, columns=list(base_inputs.keys()))
    base_prediction = model.predict(base_df)[0]
    
    st.write(f"**Base Mixture Predicted Strength:** {base_prediction:.2f} MPa")
    
    # Variable to vary
    st.write("### Select Variable to Analyze")
    
    # Select variable and range
    var_to_vary = st.selectbox(
        "Select component to vary:",
        [(col, column_mapping.get(col, col)) for col in X_columns],
        format_func=lambda x: x[1]
    )[0]
    
    # Get min and max values for the selected variable
    var_min = df[var_to_vary].min()
    var_max = df[var_to_vary].max()
    
    # Set range to vary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_value = st.number_input(
            "Minimum value", 
            min_value=float(var_min), 
            max_value=float(var_max),
            value=float(max(var_min, base_inputs[var_to_vary] * 0.5)),
            step=float((var_max - var_min) / 100)
        )
    
    with col2:
        max_value = st.number_input(
            "Maximum value", 
            min_value=float(var_min), 
            max_value=float(var_max),
            value=float(min(var_max, base_inputs[var_to_vary] * 1.5)),
            step=float((var_max - var_min) / 100)
        )
    
    with col3:
        steps = st.number_input("Number of steps", min_value=5, max_value=100, value=20, step=1)
    
    # Perform analysis
    if st.button("Run What-If Analysis"):
        with st.spinner("Running analysis..."):
            # Generate values to vary
            values = np.linspace(min_value, max_value, steps)
            
            # Make predictions for each value
            predictions = []
            
            for val in values:
                # Create a copy of base inputs and update the varying variable
                inputs_copy = base_inputs.copy()
                inputs_copy[var_to_vary] = val
                
                # Create input DataFrame
                input_array = np.array([list(inputs_copy.values())])
                input_df = pd.DataFrame(input_array, columns=list(inputs_copy.keys()))
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                predictions.append(prediction)
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the relationship
            ax.plot(values, predictions, 'o-', color='#2196F3', linewidth=2, markersize=6)
            
            # Highlight the base value
            base_value = base_inputs[var_to_vary]
            ax.axvline(x=base_value, color='red', linestyle='--', linewidth=1)
            ax.text(base_value, min(predictions), "Base", 
                    color='red', ha='center', va='bottom', rotation=90)
            
            # Customize the plot
            simple_var_name = column_mapping.get(var_to_vary, var_to_vary)
            ax.set_xlabel(simple_var_name)
            ax.set_ylabel("Predicted Strength (MPa)")
            ax.set_title(f"Effect of {simple_var_name} on Concrete Strength")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Create a table with the results
            result_df = pd.DataFrame({
                simple_var_name: values,
                "Predicted Strength (MPa)": predictions,
            #    "Change from Base (MPa)": [p - base_prediction for p in predictions],
            #    "Change from Base (%)": [(p - base_prediction) / base_prediction * 100 for p in predictions]
            })
            
            st.write("### Detailed Results")
            st.dataframe(result_df.style.highlight_max(axis=0, subset=["Predicted Strength (MPa)"]))
            
            # Download option for results
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"what_if_analysis_{simple_var_name}.csv",
                mime="text/csv"
            )
            
            # Key insights
            st.write("### Key Insights")
            
            max_idx = np.argmax(predictions)
            max_value = values[max_idx]
            max_prediction = predictions[max_idx]
            
            min_idx = np.argmin(predictions)
            min_value = values[min_idx]
            min_prediction = predictions[min_idx]
            
            st.write(f"**Optimal value of {simple_var_name}:** {max_value:.2f} (Strength: {max_prediction:.2f} MPa)")
            st.write(f"**Worst value of {simple_var_name}:** {min_value:.2f} (Strength: {min_prediction:.2f} MPa)")
            
            # Calculate sensitivity (average change in strength per unit change in variable)
            total_var_change = values[-1] - values[0]
            total_strength_change = predictions[-1] - predictions[0]
            sensitivity = total_strength_change / total_var_change
            
            st.write(f"**Sensitivity:** {sensitivity:.4f} MPa per unit change in {simple_var_name}")
            
            # Recommend next steps
            st.write("### Recommendations")
            
            if sensitivity > 0:
                st.success(f"Increasing {simple_var_name} generally improves concrete strength.")
                if base_value < max_value:
                    st.info(f"Consider increasing {simple_var_name} from {base_value:.2f} to {max_value:.2f} for optimal strength.")
            elif sensitivity < 0:
                st.success(f"Decreasing {simple_var_name} generally improves concrete strength.")
                if base_value > max_value:
                    st.info(f"Consider decreasing {simple_var_name} from {base_value:.2f} to {max_value:.2f} for optimal strength.")
            else:
                st.info(f"{simple_var_name} has minimal impact on concrete strength.")

def interpret_strength_prediction(strength):
    """Provide interpretation of predicted strength value"""
    st.write("### Strength Interpretation")
    
    if strength < 20:
        st.warning("""
            **Low Strength Concrete (< 20 MPa)**
            
            Typical uses:
            - Non-structural applications
            - Pathways and garden walls
            - Leveling or blinding concrete
            
            This strength may not be suitable for structural applications or load-bearing elements.
        """)
    elif strength < 30:
        st.info("""
            **Medium Strength Concrete (20-30 MPa)**
            
            Typical uses:
            - Residential foundations
            - Driveways and patios
            - Low-rise building structures
            - Interior floor slabs
            
            This is common for general construction applications.
        """)
    elif strength < 40:
        st.success("""
            **Standard Structural Concrete (30-40 MPa)**
            
            Typical uses:
            - Commercial building structures
            - Beams and columns
            - Bridge components
            - Reinforced concrete elements
            
            This strength is suitable for most structural applications.
        """)
    elif strength < 60:
        st.success("""
            **High Strength Concrete (40-60 MPa)**
            
            Typical uses:
            - High-rise buildings
            - Heavy-duty industrial floors
            - Bridges and infrastructure
            - Precast concrete elements
            
            This provides excellent structural performance for demanding applications.
        """)
    else:
        st.success("""
            **Very High Strength Concrete (> 60 MPa)**
            
            Typical uses:
            - Skyscrapers
            - Critical infrastructure
            - Heavy-load industrial structures
            - Special applications requiring exceptional durability
            
            This provides exceptional strength for the most demanding applications.
        """)