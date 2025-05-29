import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

import data_loader
import utilities

def show_model_training():
    """Display model training section"""
    st.header("Model Training and Evaluation")
    
    # Load data
    df = data_loader.load_data()
    
    # Setup feature selection
    X = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    y = df['Concrete compressive strength(MPa, megapascals) ']
    
    # Create tabs for different stages
    tab1, tab2, tab3 = st.tabs(["Data Preparation", "Model Training", "Model Evaluation"])
    
    with tab1:
        X_train, X_test, y_train, y_test, scaler = data_preparation(X, y)
    
    with tab2:
        trained_models = model_training(X_train, X_test, y_train, y_test)
    
    with tab3:
        model_evaluation(trained_models, X_train, X_test, y_train, y_test, scaler)

def data_preparation(X, y):
    """Prepare data for modeling"""
    st.subheader("Data Preparation")
    
    # Test size selection
    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    
    # Random state selection
    random_state = st.number_input("Random state (seed)", 0, 100, 42)
    
    # Train-test split
    st.write("### Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"Training set: {X_train.shape[0]} samples")
    st.write(f"Testing set: {X_test.shape[0]} samples")
    
    # Visualization of the split
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].pie([len(X_train), len(X_test)], labels=['Train', 'Test'], autopct='%1.1f%%', colors=['#4CAF50', '#2196F3'])
    ax[0].set_title('Train-Test Split')
    
    # Visualization of target distribution in train vs test
    sns.histplot(y_train, kde=True, color='#4CAF50', ax=ax[1], label='Train', alpha=0.6)
    sns.histplot(y_test, kde=True, color='#2196F3', ax=ax[1], label='Test', alpha=0.6)
    ax[1].set_title('Target Distribution: Train vs Test')
    ax[1].set_xlabel('Concrete Strength (MPa)')
    ax[1].legend()
    
    st.pyplot(fig)
    
    # Feature scaling
    st.write("### Feature Scaling")
    
    scaling_method = st.radio(
        "Feature scaling method:",
        ["StandardScaler", "None"],
        horizontal=True
    )
    
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.success("Features have been standardized (mean=0, std=1)")
        
        # Show scaling effect
        if st.checkbox("Show effect of scaling on features", value=False):
            feature_idx = st.selectbox(
                "Select feature to visualize scaling effect:",
                [(i, col) for i, col in enumerate(X.columns)],
                format_func=lambda x: x[1]
            )[0]
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(X_train.iloc[:, feature_idx], kde=True, ax=ax[0])
            ax[0].set_title('Before Scaling')
            
            sns.histplot(X_train_scaled[:, feature_idx], kde=True, ax=ax[1])
            ax[1].set_title('After Scaling')
            
            st.pyplot(fig)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    else:
        st.info("No scaling applied")
        return X_train, X_test, y_train, y_test, None

def model_training(X_train, X_test, y_train, y_test):
    """Train machine learning models"""
    st.subheader("Model Training")
    
    # Model selection
    model_options = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel='rbf')
    }
    
    selected_models = st.multiselect(
        "Select models to train:",
        list(model_options.keys()),
        default=["Linear Regression", "Random Forest", "XGBoost"]
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train")
        return {}
    
    # Advanced model configuration
    st.write("### Model Configuration")
    
    show_advanced = st.checkbox("Show advanced model configuration", value=False)
    
    if show_advanced:
        model_options = configure_models(model_options, selected_models)
    
    # Train selected models
    st.write("### Training Progress")
    
    trained_models = {}
    metrics = []
    
    for model_name in selected_models:
        model = model_options[model_name]
        
        with st.spinner(f"Training {model_name}..."):
            # Training
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics calculation
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            metrics.append({
                'Model': model_name,
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Test-Train MSE Ratio': test_mse / train_mse if train_mse > 0 else float('inf')
            })
            
            trained_models[model_name] = {
                'model': model,
                'train_pred': y_train_pred,
                'test_pred': y_test_pred
            }
            
            st.success(f"{model_name} training completed")
    
    # Display metrics table
    metrics_df = pd.DataFrame(metrics)
    
    st.write("### Model Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Train R²', 'Test R²'])
                          .highlight_min(axis=0, subset=['Train MSE', 'Test MSE', 'Train MAE', 'Test MAE']), 
                use_container_width=True)
    
    # Save trained models
    if st.button("Save Trained Models"):
        save_models(trained_models)
    
    return trained_models

def configure_models(model_options, selected_models):
    """Configure model hyperparameters"""
    new_model_options = model_options.copy()
    
    # Configuration for each model type
    for model_name in selected_models:
        st.write(f"#### {model_name} Configuration")
        
        if model_name == "Decision Tree":
            max_depth = st.slider("Max depth", 1, 30, 10, key=f"{model_name}_depth")
            min_samples_split = st.slider("Min samples split", 2, 20, 2, key=f"{model_name}_split")
            min_samples_leaf = st.slider("Min samples leaf", 1, 20, 1, key=f"{model_name}_leaf")
            
            new_model_options[model_name] = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        
        elif model_name == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 300, 100, key=f"{model_name}_trees")
            max_depth = st.slider("Max depth", 1, 30, 10, key=f"{model_name}_depth")
            min_samples_split = st.slider("Min samples split", 2, 20, 2, key=f"{model_name}_split")
            
            new_model_options[model_name] = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        
        elif model_name == "XGBoost":
            n_estimators = st.slider("Number of trees", 10, 300, 100, key=f"{model_name}_trees")
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, step=0.01, key=f"{model_name}_lr")
            max_depth = st.slider("Max depth", 1, 15, 6, key=f"{model_name}_depth")
            
            new_model_options[model_name] = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        elif model_name == "KNN":
            n_neighbors = st.slider("Number of neighbors", 1, 20, 5, key=f"{model_name}_neighbors")
            weights = st.selectbox("Weight function", ["uniform", "distance"], key=f"{model_name}_weights")
            
            new_model_options[model_name] = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights
            )
        
        elif model_name == "SVR":
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2, key=f"{model_name}_kernel")
            C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, key=f"{model_name}_C")
            epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, key=f"{model_name}_epsilon")
            
            new_model_options[model_name] = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon
            )
    
    return new_model_options

def model_evaluation(trained_models, X_train, X_test, y_train, y_test, scaler):
    """Evaluate trained models"""
    if not trained_models:
        st.warning("Please train models in the 'Model Training' tab first")
        return
    
    st.subheader("Model Evaluation")
    
    # Model selection for evaluation
    model_name = st.selectbox(
        "Select model to evaluate:",
        list(trained_models.keys())
    )
    
    model_info = trained_models[model_name]
    model = model_info['model']
    y_train_pred = model_info['train_pred']
    y_test_pred = model_info['test_pred']
    
    # Evaluation type
    eval_type = st.radio(
        "Select evaluation type:",
        ["Prediction vs Actual", "Residual Analysis", "Learning Curve", "Feature Importance"],
        horizontal=True
    )
    
    if eval_type == "Prediction vs Actual":
        show_prediction_vs_actual(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    elif eval_type == "Residual Analysis":
        show_residual_analysis(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    elif eval_type == "Learning Curve":
        show_learning_curve(model, X_train, y_train, model_name)
    
    elif eval_type == "Feature Importance":
        show_feature_importance(model, X_train, model_name)

def show_prediction_vs_actual(y_train, y_train_pred, y_test, y_test_pred, model_name):
    """Display prediction vs actual comparison"""
    st.write("### Prediction vs Actual")
    
    # Dataset selection
    dataset = st.radio(
        "Select dataset:",
        ["Test Set", "Training Set", "Both"],
        horizontal=True
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if dataset in ["Training Set", "Both"]:
        ax.scatter(y_train, y_train_pred, alpha=0.5, label="Training Data", color="#4CAF50")
    
    if dataset in ["Test Set", "Both"]:
        ax.scatter(y_test, y_test_pred, alpha=0.5, label="Test Data", color="#2196F3")
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(y_train), min(y_test))
    max_val = max(max(y_train), max(y_test))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    ax.set_xlabel("Actual Strength (MPa)")
    ax.set_ylabel("Predicted Strength (MPa)")
    ax.set_title(f"{model_name}: Prediction vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Metrics calculation
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Training Set Metrics")
        st.write(f"MSE: {train_mse:.4f}")
        st.write(f"RMSE: {np.sqrt(train_mse):.4f}")
        st.write(f"MAE: {train_mae:.4f}")
        st.write(f"R²: {train_r2:.4f}")
    
    with col2:
        st.write("#### Test Set Metrics")
        st.write(f"MSE: {test_mse:.4f}")
        st.write(f"RMSE: {np.sqrt(test_mse):.4f}")
        st.write(f"MAE: {test_mae:.4f}")
        st.write(f"R²: {test_r2:.4f}")

def show_residual_analysis(y_train, y_train_pred, y_test, y_test_pred, model_name):
    """Display residual analysis"""
    st.write("### Residual Analysis")
    
    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    # Dataset selection
    dataset = st.radio(
        "Select dataset:",
        ["Test Set", "Training Set"],
        horizontal=True
    )
    
    residuals = train_residuals if dataset == "Training Set" else test_residuals
    predictions = y_train_pred if dataset == "Training Set" else y_test_pred
    
    # Plot types
    plot_type = st.radio(
        "Select plot type:",
        ["Residuals vs Predicted", "Residual Distribution", "Both"],
        horizontal=True
    )
    
    if plot_type in ["Residuals vs Predicted", "Both"]:
        st.write("#### Residuals vs Predicted Values")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predictions, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{model_name}: Residuals vs Predicted ({dataset})")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    if plot_type in ["Residual Distribution", "Both"]:
        st.write("#### Residual Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model_name}: Residual Distribution ({dataset})")
        
        st.pyplot(fig)
        
        # Additional residual statistics
        st.write("#### Residual Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Mean: {np.mean(residuals):.4f}")
            st.write(f"Median: {np.median(residuals):.4f}")
        
        with col2:
            st.write(f"Standard Deviation: {np.std(residuals):.4f}")
            st.write(f"Skewness: {skew(residuals):.4f}")
        
        # Check for normality of residuals
        from scipy.stats import shapiro
        
        stat, p = shapiro(residuals)
        st.write(f"Shapiro-Wilk Test (normality): p-value = {p:.4f}")
        
        if p < 0.05:
            st.info("Residuals may not be normally distributed (p < 0.05)")
        else:
            st.success("Residuals appear to be normally distributed (p >= 0.05)")

def show_learning_curve(model, X_train, y_train, model_name):
    """Display model learning curve"""
    st.write("### Learning Curve Analysis")
    
    # Parameters
    with st.spinner("Generating learning curve... This may take a moment."):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='r2', random_state=42
        )
    
    # Calculate statistics
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training scores
    ax.plot(train_sizes, train_scores_mean, 'o-', color='#4CAF50', label="Training Score")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='#4CAF50')
    
    # Plot cross-validation scores
    ax.plot(train_sizes, test_scores_mean, 'o-', color='#2196F3', label="Cross-validation Score")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='#2196F3')
    
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("R² Score")
    ax.set_title(f"Learning Curve: {model_name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Interpretation
    st.write("### Learning Curve Interpretation")
    
    final_gap = train_scores_mean[-1] - test_scores_mean[-1]
    
    st.write(f"**Final Training Score:** {train_scores_mean[-1]:.4f}")
    st.write(f"**Final Validation Score:** {test_scores_mean[-1]:.4f}")
    st.write(f"**Final Gap:** {final_gap:.4f}")
    
    # Interpretation of the learning curve
    if final_gap > 0.2:
        st.warning("""
            **High Variance (Overfitting):** The model performs significantly better on the training data than on validation data.
            Consider using regularization, gathering more training data, or simplifying the model.
        """)
    elif test_scores_mean[-1] < 0.6:
        st.warning("""
            **High Bias (Underfitting):** The model has low performance on both training and validation data.
            Consider using a more complex model, adding features, or reducing regularization.
        """)
    else:
        st.success("""
            **Good Fit:** The model shows good performance with a reasonable gap between training and validation scores.
        """)

def show_feature_importance(model, X_train, model_name):
    """Display feature importance analysis"""
    st.write("### Feature Importance Analysis")
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        features = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importances
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title(f"Feature Importance: {model_name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        
        st.pyplot(fig)
        
        # Display importance values
        st.write("#### Feature Importance Values")
        st.dataframe(importance_df)
        
    elif model_name == "Linear Regression" and hasattr(model, 'coef_'):
        # For linear regression, show coefficients
        coef = model.coef_
        features = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coef
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        # Plot coefficients
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        bar_plot = sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='RdBu_r', ax=ax)
        
        # Color negative and positive coefficients differently
        for i, (_, row) in enumerate(coef_df.iterrows()):
            if row['Coefficient'] < 0:
                bar_plot.patches[i].set_facecolor('#EF5350')
            else:
                bar_plot.patches[i].set_facecolor('#42A5F5')
        
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_title(f"Feature Coefficients: {model_name}")
        ax.set_xlabel("Coefficient Value")
        ax.set_ylabel("Feature")
        
        st.pyplot(fig)
        
        # Display coefficient values
        st.write("#### Feature Coefficient Values")
        st.dataframe(coef_df)
        
        # Add note about standardization
        st.info("""
            Note: For a fair comparison of feature importance in linear regression, features should be standardized.
            Larger coefficient magnitudes indicate stronger influence on the prediction.
        """)
    else:
        st.warning(f"Feature importance visualization is not available for {model_name}")

def save_models(trained_models):
    """Save trained models to disk"""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    saved_models = []
    
    for model_name, model_info in trained_models.items():
        try:
            # Save the model
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(model_info['model'], model_path)
            saved_models.append(model_name)
        except Exception as e:
            st.error(f"Error saving {model_name}: {e}")
    
    if saved_models:
        st.success(f"Successfully saved models: {', '.join(saved_models)}")
    else:
        st.error("No models were saved")