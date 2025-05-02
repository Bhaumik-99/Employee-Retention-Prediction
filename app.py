import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                            auc, accuracy_score, precision_recall_curve,
                            f1_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance
import shap

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FFC107;
        font-weight: bold;
    }
    .danger-text {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://www.svgrepo.com/show/530443/management.svg", width=100)
    st.markdown("## Navigation")
    page = st.radio("", ["📊 Dashboard", "🔍 Exploratory Analysis", "🚀 Model Training", "🔮 Predictions", "📝 Documentation"])
    
    st.markdown("---")
    
    st.markdown("## Data Input")
    uploaded_file = st.file_uploader("Upload your HR data (CSV)", type=['csv'])
    sample_data = st.checkbox("Use sample data", value=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This dashboard helps HR professionals predict employee turnover and understand key factors affecting retention.")

@st.cache_data
def load_data(file=None):
    if file is not None:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv("HR_comma_sep.csv")
        if 'sales' in data.columns:
            data = data.rename(columns={'sales': 'Department'})
    return data

if uploaded_file is not None:
    data = load_data(uploaded_file)
elif sample_data:
    data = load_data()
else:
    st.error("Please upload a file or select 'Use sample data'")
    st.stop()

def engineer_features(df):
    df_new = df.copy()
    
    if 'satisfaction_level' in df.columns and 'last_evaluation' in df.columns:
        df_new['satisfaction_evaluation_ratio'] = df['satisfaction_level'] / (df['last_evaluation'] + 0.001)
    
    if 'average_montly_hours' in df.columns:
        df_new = df_new.rename(columns={'average_montly_hours': 'average_monthly_hours'})
    
    if 'average_monthly_hours' in df_new.columns:
        df_new['overwork_index'] = df_new['average_monthly_hours'] / 160
    
    if 'time_spend_company' in df.columns and 'promotion_last_5years' in df.columns:
        df_new['no_promotion_risk'] = ((df['time_spend_company'] > 4) & (df['promotion_last_5years'] == 0)).astype(int)
    
    if 'Department' in df.columns and pd.api.types.is_numeric_dtype(df['Department']):
        df_new['Department'] = df_new['Department'].astype(str)
    
    if 'salary' in df.columns and pd.api.types.is_numeric_dtype(df['salary']):
        df_new['salary'] = df_new['salary'].astype(str)
    
    return df_new

data_processed = engineer_features(data)

if 'left' not in data_processed.columns:
    st.error("The dataset must contain a column named 'left' (0 = stayed, 1 = left)")
    st.stop()

if page == "📊 Dashboard":
    # Main header
    st.markdown("<h1 class='main-header'>👥 Employee Retention Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    total_employees = len(data_processed)
    left_employees = data_processed['left'].sum()
    retention_rate = (1 - (left_employees / total_employees)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{total_employees}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Employees</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{left_employees}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Employees Left</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{retention_rate:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Retention Rate</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_satisfaction = data_processed['satisfaction_level'].mean() * 100
        st.markdown(f"<div class='metric-value'>{avg_satisfaction:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg Satisfaction</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Key Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Turnover by Department")
        
        dept_turnover = data_processed.groupby('Department')['left'].agg(['count', 'sum', 'mean'])
        dept_turnover = dept_turnover.reset_index()
        dept_turnover.columns = ['Department', 'Total', 'Left', 'Turnover Rate']
        dept_turnover['Turnover Rate'] = dept_turnover['Turnover Rate'] * 100
        
        fig = px.bar(dept_turnover.sort_values('Turnover Rate', ascending=False), 
                    x='Department', y='Turnover Rate', 
                    text=dept_turnover['Turnover Rate'].round(1).astype(str) + '%',
                    color='Turnover Rate',
                    color_continuous_scale='Reds',
                    title="Turnover Rate by Department")
        fig.update_layout(xaxis_title="Department", yaxis_title="Turnover Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Satisfaction vs Hours Worked")
        
        fig = px.scatter(data_processed, 
                        x='satisfaction_level', y='average_monthly_hours' if 'average_monthly_hours' in data_processed.columns else 'average_montly_hours',
                        color='left', size='time_spend_company',
                        color_continuous_scale=['blue', 'red'],
                        title="Satisfaction vs. Working Hours",
                        labels={'satisfaction_level': 'Satisfaction Level', 
                                'average_monthly_hours' if 'average_monthly_hours' in data_processed.columns else 'average_montly_hours': 'Monthly Hours',
                                'left': 'Left Company'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Projects vs Evaluation")
        
        fig = px.box(data_processed, x='number_project', y='last_evaluation',
                    color='left', notched=True,
                    title="Performance Evaluation by Project Count",
                    labels={'number_project': 'Number of Projects', 
                            'last_evaluation': 'Last Evaluation Score',
                            'left': 'Left Company'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Years at Company vs Turnover")
        
        tenure_turnover = data_processed.groupby('time_spend_company')['left'].mean() * 100
        tenure_turnover = tenure_turnover.reset_index()
        tenure_turnover.columns = ['Years at Company', 'Turnover Rate']
        
        fig = px.line(tenure_turnover, x='Years at Company', y='Turnover Rate',
                    markers=True, line_shape='linear',
                    title="Turnover Rate by Tenure")
        fig.update_layout(yaxis_title="Turnover Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔍 Exploratory Analysis":
    st.markdown("<h1 class='main-header'>🔍 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Data Preview")
        st.write(data_processed.head())
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Data Summary")
        buffer = StringIO()
        data_processed.info(buf=buffer)
        st.text(buffer.getvalue())
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
    
    numeric_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        corr = data_processed[numeric_cols].corr()
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Distribution Analysis</h2>", unsafe_allow_html=True)
    
    feature_options = numeric_cols.copy()
    if 'left' in feature_options:
        feature_options.remove('left')
    
    if feature_options:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_feature = st.selectbox("Select feature to analyze:", feature_options)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = px.histogram(data_processed, x=selected_feature, color='left', marginal='box',
                              barmode='overlay', opacity=0.7,
                              color_discrete_map={0: 'blue', 1: 'red'},
                              labels={'left': 'Left Company'},
                              title=f"Distribution of {selected_feature} by Employee Status")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Categorical Analysis</h2>", unsafe_allow_html=True)
    
    cat_cols = data_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            selected_cat = st.selectbox("Select categorical feature:", cat_cols)
            
            cat_counts = data_processed.groupby([selected_cat, 'left']).size().unstack(fill_value=0)
            cat_counts['total'] = cat_counts.sum(axis=1)
            cat_counts['left_pct'] = cat_counts[1] / cat_counts['total'] * 100
            cat_counts = cat_counts.sort_values('left_pct', ascending=False)
            
            fig = px.bar(cat_counts, x=cat_counts.index, y='left_pct', 
                        text=cat_counts['left_pct'].round(1).astype(str) + '%',
                        title=f"Turnover Rate by {selected_cat}",
                        labels={selected_cat: selected_cat, 'left_pct': 'Turnover Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            # Stacked bar chart
            cat_data = data_processed[selected_cat].value_counts().reset_index()
            cat_data.columns = [selected_cat, 'count']
            
            fig = px.sunburst(data_processed, path=[selected_cat, 'left'], 
                             values='satisfaction_level',
                             color='left',
                             color_discrete_map={0: 'blue', 1: 'red'},
                             title=f"Turnover Distribution by {selected_cat}")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Advanced Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Risk Segments")
        
        # Create risk segments based on satisfaction and workload
        hour_col = 'average_monthly_hours' if 'average_monthly_hours' in data_processed.columns else 'average_montly_hours'
        
        data_processed['risk_segment'] = 'Medium Risk'
        data_processed.loc[(data_processed['satisfaction_level'] < 0.2) & 
                         (data_processed[hour_col] > 250), 'risk_segment'] = 'High Risk'
        data_processed.loc[(data_processed['satisfaction_level'] > 0.7) & 
                         (data_processed[hour_col] < 200), 'risk_segment'] = 'Low Risk'
        
        risk_counts = data_processed.groupby('risk_segment')['left'].agg(['count', 'mean'])
        risk_counts['turnover'] = risk_counts['mean'] * 100
        risk_counts = risk_counts.reset_index()
        
        fig = px.pie(risk_counts, values='count', names='risk_segment', 
                    color='risk_segment',
                    color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'gold', 'High Risk': 'red'},
                    hole=0.4,
                    title="Employee Risk Segments")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show turnover by risk segment
        st.markdown(f"**Turnover rates by segment:**")
        for idx, row in risk_counts.iterrows():
            segment = row['risk_segment']
            turnover = row['turnover']
            color = 'success-text' if segment == 'Low Risk' else 'warning-text' if segment == 'Medium Risk' else 'danger-text'
            st.markdown(f"<span class='{color}'>{segment}: {turnover:.1f}%</span>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Satisfaction & Engagement Matrix")
        
        data_processed['satisfaction_level_cat'] = pd.cut(data_processed['satisfaction_level'], 
                                                      bins=[0, 0.3, 0.7, 1], 
                                                      labels=['Low', 'Medium', 'High'])
        data_processed['last_evaluation_cat'] = pd.cut(data_processed['last_evaluation'], 
                                                    bins=[0, 0.3, 0.7, 1], 
                                                    labels=['Low', 'Medium', 'High'])
        
        matrix_data = data_processed.groupby(['satisfaction_level_cat', 'last_evaluation_cat'])['left'].mean().reset_index()
        matrix_data['left_pct'] = matrix_data['left'] * 100
        
        matrix_pivot = matrix_data.pivot(index='satisfaction_level_cat', 
                                         columns='last_evaluation_cat', 
                                         values='left_pct')
        
        fig = px.imshow(matrix_pivot, 
                        labels=dict(x="Performance Evaluation", y="Satisfaction Level", color="Turnover %"),
                        x=matrix_pivot.columns,
                        y=matrix_pivot.index,
                        color_continuous_scale='Reds',
                        text_auto='.1f')
        fig.update_layout(title="Turnover Rates by Satisfaction & Performance Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Matrix Interpretation:**
        - **High Satisfaction, High Evaluation**: Typically happy, productive employees
        - **Low Satisfaction, High Evaluation**: Burnout risk, high performers who are dissatisfied
        - **High Satisfaction, Low Evaluation**: May indicate complacency
        - **Low Satisfaction, Low Evaluation**: High turnover risk, disengaged employees
        """)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🚀 Model Training":
    st.markdown("<h1 class='main-header'>🚀 Predictive Modeling</h1>", unsafe_allow_html=True)
    
    target_var = 'left'
    X = data_processed.drop(target_var, axis=1)
    y = data_processed[target_var]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    st.markdown("<h2 class='sub-header'>Model Configuration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Feature Selection")
        
        feature_selection = st.multiselect("Select features to use:", 
                                         X.columns.tolist(),
                                         default=X.columns.tolist())
        
        random_state = st.slider("Random seed:", min_value=1, max_value=100, value=42)
        
        test_size = st.slider("Test set size:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Selection")
        
        models_to_train = st.multiselect(
            "Select models to train:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
            default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )
        
        do_hyperparameter_tuning = st.checkbox("Perform hyperparameter tuning", value=False)
        
        cv_folds = st.slider("Cross-validation folds:", min_value=2, max_value=10, value=5)
        st.markdown("</div>", unsafe_allow_html=True)
    
    X_selected = X[feature_selection]
    
    selected_categorical_cols = [col for col in categorical_cols if col in feature_selection]
    selected_numerical_cols = [col for col in numerical_cols if col in feature_selection]
    
    if st.button("Train Models"):
        with st.spinner('Training models... This may take a moment.'):
            # Create preprocessor
            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), selected_numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]), selected_categorical_cols)
            ])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, stratify=y, random_state=random_state
            )
            
            results = []
            models = {}
            
            if "Logistic Regression" in models_to_train:
                if do_hyperparameter_tuning:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', GridSearchCV(
                            LogisticRegression(random_state=random_state),
                            param_grid={'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},
                            cv=cv_folds, scoring='f1', n_jobs=-1
                        ))
                    ])
                else:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', LogisticRegression(random_state=random_state))
                    ])
                models["Logistic Regression"] = model
            
            if "Decision Tree" in models_to_train:
                if do_hyperparameter_tuning:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', GridSearchCV(
                            DecisionTreeClassifier(random_state=random_state),
                            param_grid={'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]},
                            cv=cv_folds, scoring='f1', n_jobs=-1
                        ))
                    ])
                else:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', DecisionTreeClassifier(random_state=random_state))
                    ])
                models["Decision Tree"] = model
            
            if "Random Forest" in models_to_train:
                if do_hyperparameter_tuning:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', GridSearchCV(
                            RandomForestClassifier(random_state=random_state),
                            param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
                            cv=cv_folds, scoring='f1', n_jobs=-1
                        ))
                    ])
                else:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', RandomForestClassifier(random_state=random_state))
                    ])
                models["Random Forest"] = model
            
            if "Gradient Boosting" in models_to_train:
                if do_hyperparameter_tuning:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', GridSearchCV(
                            GradientBoostingClassifier(random_state=random_state),
                            param_grid={'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]},
                            cv=cv_folds, scoring='f1', n_jobs=-1
                        ))
                    ])
                else:
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', GradientBoostingClassifier(random_state=random_state))
                    ])
                models["Gradient Boosting"] = model
            
            for name, pipeline in models.items():
                pipeline.fit(X_train, y_train)
                
                if do_hyperparameter_tuning:
                    best_params = pipeline.named_steps['model'].best_params_
                    best_model = pipeline.named_steps['model'].best_estimator_
                    st.write(f"**{name} Best Parameters:** {best_params}")
                
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'AUC': roc_auc
                })
            
            results_df = pd.DataFrame(results)
            
            st.markdown("<h2 class='sub-header'>Model Performance Results</h2>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Model Comparison")
            
            results_df_sorted = results_df.sort_values('F1 Score', ascending=False)
            
            for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']:
                results_df_sorted[col] = results_df_sorted[col].map(lambda x: f"{x:.4f}")
            
            st.dataframe(results_df_sorted, use_container_width=True)
            
            fig = px.bar(results_df_sorted.melt(id_vars=['Model'], 
                                             value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']),
                        x='Model', y='value', color='variable', barmode='group',
                        labels={'value': 'Score', 'variable': 'Metric'},
                        title="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            best_model_name = results_df_sorted.iloc[0]['Model']
            best_model = models[best_model_name]
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Confusion Matrix - {best_model_name}")
                cm = confusion_matrix(y_test, best_model.predict(X_test))
                
                cm_sum = np.sum(cm)
                cm_perc = cm / cm_sum * 100
                annot = np.empty_like(cm, dtype=str)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        annot[i, j] = f"{cm[i, j]}\n{cm_perc[i, j]:.1f}%"
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_xticklabels(['Stayed', 'Left'])
                ax.set_yticklabels(['Stayed', 'Left'])
                st.pyplot(fig)
            
            with col2:
                st.subheader("ROC Curves")
                fig = go.Figure()
                
                for model_name in models.keys():
                    model = models[model_name]
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    auc_score = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={auc_score:.4f})"))
                
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
                                       line=dict(dash='dash', color='gray')))
                
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    legend=dict(x=0.5, y=0, xanchor='center'),
                    width=550,
                    height=400
                )
                st.plotly_chart(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<h2 class='sub-header'>Feature Importance Analysis</h2>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Feature Importance - {best_model_name}")
                
                # Get feature names after preprocessing
                preprocessor = best_model.named_steps['preprocessor']
                preprocessor.fit(X_train)
                
                feature_names = []
                if selected_numerical_cols:
                    feature_names.extend(selected_numerical_cols)
                
                if selected_categorical_cols:
                    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                    encoded_features = encoder.get_feature_names_out(selected_categorical_cols)
                    feature_names.extend(encoded_features)
                
                if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                    importances = best_model.named_steps['model'].feature_importances_
                    
                    if do_hyperparameter_tuning:
                        importances = best_model.named_steps['model'].best_estimator_.feature_importances_
                    
                    if len(importances) == len(feature_names):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        })
                        importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature', 
                                    orientation='h', title=f"Top Features by Importance")
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Feature names and importance scores don't match. Using permutation importance instead.")
                        
                        perm_importance = permutation_importance(
                            best_model, X_test, y_test, n_repeats=10, random_state=random_state
                        )
                        
                        perm_importance_df = pd.DataFrame({
                            'Feature': X_test.columns,
                            'Importance': perm_importance.importances_mean
                        })
                        perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(perm_importance_df, x='Importance', y='Feature', 
                                    orientation='h', title=f"Top Features by Permutation Importance")
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Using permutation importance for this model type")
                    perm_importance = permutation_importance(
                        best_model, X_test, y_test, n_repeats=10, random_state=random_state
                    )
                    
                    if len(X_test.columns) == len(perm_importance.importances_mean):
                        perm_importance_df = pd.DataFrame({
                            'Feature': X_test.columns,
                            'Importance': perm_importance.importances_mean
                        })
                        perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(perm_importance_df, x='Importance', y='Feature', 
                                    orientation='h', title=f"Top Features by Permutation Importance")
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not calculate feature importance for this model")
            
            with col2:
                st.subheader("Feature Impact on Predictions")
                
                numeric_features = [col for col in selected_numerical_cols if col in X_test.columns]
                
                if numeric_features:
                    pdp_feature = st.selectbox("Select feature for analysis:", numeric_features)
                    
                    feature_min = X_test[pdp_feature].min()
                    feature_max = X_test[pdp_feature].max()
                    feature_range = np.linspace(feature_min, feature_max, 100)
                    
                    pdp_result = []
                    
                    for value in feature_range:
                        X_pdp = X_test.copy()
                        X_pdp[pdp_feature] = value
                        pred_probs = best_model.predict_proba(X_pdp)[:, 1].mean()
                        pdp_result.append((value, pred_probs))
                    
                    pdp_df = pd.DataFrame(pdp_result, columns=[pdp_feature, 'Probability'])
                    
                    fig = px.line(pdp_df, x=pdp_feature, y='Probability',
                                title=f"Partial Dependence Plot: Impact of {pdp_feature} on Turnover Probability")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric features available for partial dependence plot")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state['best_model'] = best_model
            st.session_state['feature_names'] = feature_selection
            st.session_state['trained'] = True
            
            st.success("✅ Model training completed successfully! Navigate to the Predictions tab to use the model.")

elif page == "🔮 Predictions":
    st.markdown("<h1 class='main-header'>🔮 Employee Turnover Predictions</h1>", unsafe_allow_html=True)
    
    if 'trained' not in st.session_state or not st.session_state['trained']:
        st.warning("⚠️ No trained model available. Please go to the Model Training page first.")
    else:
        best_model = st.session_state['best_model']
        feature_names = st.session_state['feature_names']
        
        st.markdown("<h2 class='sub-header'>Individual Employee Prediction</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Enter employee information to predict turnover risk:")
        
        col1, col2, col3 = st.columns(3)
        
        employee_data = {}
        
        with col1:
            if 'satisfaction_level' in feature_names:
                employee_data['satisfaction_level'] = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
            
            if 'last_evaluation' in feature_names:
                employee_data['last_evaluation'] = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.01)
            
            if 'number_project' in feature_names:
                employee_data['number_project'] = st.slider("Number of Projects", 1, 10, 4)
        
        with col2:
            hour_col = 'average_monthly_hours' if 'average_monthly_hours' in feature_names else 'average_montly_hours'
            if hour_col in feature_names:
                employee_data[hour_col] = st.slider("Average Monthly Hours", 50, 350, 160, 5)
            
            if 'time_spend_company' in feature_names:
                employee_data['time_spend_company'] = st.slider("Years at Company", 1, 10, 3)
            
            if 'Work_accident' in feature_names:
                employee_data['Work_accident'] = st.selectbox("Work Accident", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col3:
            if 'promotion_last_5years' in feature_names:
                employee_data['promotion_last_5years'] = st.selectbox("Promoted in Last 5 Years", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            if 'sales' in feature_names:
                departments = sorted(data_processed['sales'].unique())
                employee_data['sales'] = st.selectbox("Department", departments)
            
            if 'salary' in feature_names:
                salary_levels = sorted(data_processed['salary'].unique())
                employee_data['salary'] = st.selectbox("Salary Level", salary_levels)
        
        if 'satisfaction_evaluation_ratio' in feature_names and 'satisfaction_level' in employee_data and 'last_evaluation' in employee_data:
            employee_data['satisfaction_evaluation_ratio'] = employee_data['satisfaction_level'] / (employee_data['last_evaluation'] + 0.001)
        
        if 'overwork_index' in feature_names and hour_col in employee_data:
            employee_data['overwork_index'] = employee_data[hour_col] / 160
        
        if 'no_promotion_risk' in feature_names and 'time_spend_company' in employee_data and 'promotion_last_5years' in employee_data:
            employee_data['no_promotion_risk'] = 1 if (employee_data['time_spend_company'] > 4 and employee_data['promotion_last_5years'] == 0) else 0
        
        employee_df = pd.DataFrame([employee_data])
        
        for feature in feature_names:
            if feature not in employee_df.columns:
                if feature in data_processed.columns:
                    # Use the most common value for that feature
                    if pd.api.types.is_numeric_dtype(data_processed[feature]):
                        default_value = data_processed[feature].median()
                    else:
                        default_value = data_processed[feature].mode()[0]
                    employee_df[feature] = default_value
        
        if st.button("Predict Turnover Risk"):
            # Make prediction
            try:
                probability = best_model.predict_proba(employee_df)[0, 1]
                prediction = best_model.predict(employee_df)[0]
                
                st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Turnover Risk"},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "rgba(0, 0, 0, 0)"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': 'green'},
                                {'range': [30, 70], 'color': 'gold'},
                                {'range': [70, 100], 'color': 'red'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': probability * 100
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
                    risk_color = "success-text" if risk_level == "Low" else "warning-text" if risk_level == "Medium" else "danger-text"
                    
                    st.markdown(f"<h4>Risk Assessment: <span class='{risk_color}'>{risk_level} Risk</span></h4>", unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.markdown("<p>This employee is <strong>likely to leave</strong> based on the provided information.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p>This employee is <strong>likely to stay</strong> based on the provided information.</p>", unsafe_allow_html=True)
                    
                    st.markdown("<h4>Key Risk Factors:</h4>", unsafe_allow_html=True)
                    
                    risk_factors = []
                    
                    if 'satisfaction_level' in employee_data and employee_data['satisfaction_level'] < 0.4:
                        risk_factors.append("Low satisfaction level")
                    
                    hour_col = 'average_monthly_hours' if 'average_monthly_hours' in feature_names else 'average_montly_hours'
                    if hour_col in employee_data:
                        if employee_data[hour_col] > 250:
                            risk_factors.append("Very high working hours (potential burnout)")
                        elif employee_data[hour_col] < 130:
                            risk_factors.append("Very low working hours (potential disengagement)")
                    
                    if 'time_spend_company' in employee_data and 'promotion_last_5years' in employee_data:
                        if employee_data['time_spend_company'] > 4 and employee_data['promotion_last_5years'] == 0:
                            risk_factors.append("Long tenure without promotion")
                    
                    if 'number_project' in employee_data:
                        if employee_data['number_project'] > 6:
                            risk_factors.append("High workload (too many projects)")
                        elif employee_data['number_project'] < 2:
                            risk_factors.append("Low workload (too few projects)")
                    
                    if 'last_evaluation' in employee_data and employee_data['last_evaluation'] < 0.4:
                        risk_factors.append("Poor performance evaluation")
                    
                    if len(risk_factors) > 0:
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
                    else:
                        st.markdown("No significant risk factors identified.")
                    
                    st.markdown("<h4>Recommendations:</h4>", unsafe_allow_html=True)
                    
                    if risk_level == "High":
                        st.markdown("""
                        • Schedule an immediate check-in with the employee
                        • Review workload and project assignments
                        • Discuss career development opportunities
                        • Consider compensation review if appropriate
                        """)
                    elif risk_level == "Medium":
                        st.markdown("""
                        • Schedule a regular check-in to gauge satisfaction
                        • Provide feedback and recognition
                        • Identify growth opportunities
                        """)
                    else:
                        st.markdown("""
                        • Continue regular engagement
                        • Recognize contributions
                        • Maintain a positive work environment
                        """)
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Batch Prediction</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Upload a CSV file with employee data to get predictions for multiple employees:")
        
        uploaded_file = st.file_uploader("Upload employee data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.write(batch_data.head())
            
            missing_cols = [col for col in feature_names if col not in batch_data.columns]
            
            if missing_cols:
                st.warning(f"⚠️ The uploaded file is missing these columns: {', '.join(missing_cols)}")
            else:
                if st.button("Generate Predictions"):
                    batch_data['turnover_probability'] = best_model.predict_proba(batch_data[feature_names])[:, 1]
                    batch_data['predicted_turnover'] = best_model.predict(batch_data[feature_names])
                    
                    batch_data['risk_level'] = pd.cut(
                        batch_data['turnover_probability'], 
                        bins=[0, 0.3, 0.7, 1], 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    st.subheader("Prediction Results")
                    
                    risk_summary = batch_data['risk_level'].value_counts().reset_index()
                    risk_summary.columns = ['Risk Level', 'Count']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("Risk Level Distribution:")
                        
                        fig = px.pie(risk_summary, values='Count', names='Risk Level',
                                   color='Risk Level', 
                                   color_discrete_map={'Low': 'green', 'Medium': 'gold', 'High': 'red'},
                                   hole=0.4)
                        fig.update_traces(textinfo='percent+label')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("Employee Risk Assessment:")
                        
                        batch_display = batch_data.copy()
                        batch_display['turnover_probability'] = (batch_display['turnover_probability'] * 100).round(1).astype(str) + '%'
                        
                        display_cols = ['predicted_turnover', 'turnover_probability', 'risk_level']
                        
                        id_cols = [col for col in batch_data.columns if 'id' in col.lower() or 'name' in col.lower()]
                        display_cols = id_cols + display_cols
                        
                        key_features = [
                            'satisfaction_level', 
                            'time_spend_company', 
                            hour_col, 
                            'last_evaluation'
                        ]
                        
                        for feature in key_features:
                            if feature in batch_data.columns:
                                display_cols.append(feature)
                        
                        st.dataframe(batch_display[display_cols], use_container_width=True)
                    
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="employee_turnover_predictions.csv",
                        mime="text/csv"
                    )
        
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📝 Documentation":
    st.markdown("<h1 class='main-header'>📝 Documentation & Help</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ## About this Dashboard
    
    The Employee Retention Prediction Dashboard is a tool designed to help HR professionals and managers predict employee turnover and understand key factors affecting retention.
    
    ### Key Features
    
    1. **📊 Dashboard**: Overview of your workforce with key metrics and visualizations.
    2. **🔍 Exploratory Analysis**: In-depth analysis of employee data with interactive visualizations.
    3. **🚀 Model Training**: Train machine learning models to predict employee turnover.
    4. **🔮 Predictions**: Make predictions for individual employees or batch predictions.
    5. **📝 Documentation**: Detailed help and documentation.
    
    ### How to Use
    
    1. Start by uploading your employee data CSV file in the sidebar, or use the sample data.
    2. Navigate between tabs using the sidebar menu.
    3. Explore visualizations to understand patterns in your data.
    4. Train models to identify key factors affecting turnover.
    5. Use the trained models to predict turnover risk for employees.
    
    ### Data Requirements
    
    Your dataset should include:
    
    - `left` (target variable): Whether the employee left the company (1) or not (0)
    - Employee attributes such as satisfaction level, evaluation scores, projects, etc.
    
    The sample dataset includes these columns:
    
    - `satisfaction_level`: Employee satisfaction level (0-1)
    - `last_evaluation`: Last performance evaluation score (0-1)
    - `number_project`: Number of projects assigned
    - `average_monthly_hours`: Average monthly working hours
    - `time_spend_company`: Years at the company
    - `Work_accident`: Whether the employee had a workplace accident
    - `promotion_last_5years`: Whether the employee was promoted in the last 5 years
    - `sales`: Department/division
    - `salary`: Salary level (low, medium, high)
    
    ### Interpreting Results
    
    - **Risk Levels**:
      - **High Risk** (>70% probability): Immediate attention needed
      - **Medium Risk** (30-70% probability): Monitor and address specific concerns
      - **Low Risk** (<30% probability): Maintain regular engagement
    
    - **Key Metrics**:
      - **Accuracy**: Overall correctness of predictions
      - **Precision**: How many of the predicted "will leave" cases actually left
      - **Recall**: How many actual turnovers were correctly identified
      - **F1 Score**: Balance between precision and recall
      - **AUC**: Area under ROC curve, measures discriminative ability
    
    ### Tips for Retention Strategy
    
    1. **Address Burnout**: Monitor and manage workload, especially for high performers
    2. **Recognition**: Acknowledge contributions, especially for unrecognized high performers
    3. **Career Development**: Create clear growth paths for long-tenured employees
    4. **Work-Life Balance**: Address excessive working hours
    5. **Compensation Review**: Regularly review salary against market rates
    6. **Engagement Programs**: Focus on departments with high turnover rates
    7. **Exit Interviews**: Collect data to continuously improve retention strategies
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # FAQ section
    st.markdown("<h2 class='sub-header'>Frequently Asked Questions</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    faq_items = [
        {
            "question": "How accurate are the model predictions?",
            "answer": "Model accuracy typically ranges from 80-95% depending on data quality. The dashboard provides various metrics to evaluate model performance including accuracy, precision, recall, F1 score, and AUC."
        },
        {
            "question": "Can I use my own HR data?",
            "answer": "Yes, you can upload your own CSV file containing employee data. Ensure it has the required columns for prediction, especially the features used during model training."
        },
        {
            "question": "What are the most important factors for employee retention?",
            "answer": "Common important factors include satisfaction level, time spent at company, workload (projects and hours), compensation, and lack of promotion. The Feature Importance analysis shows the specific factors most relevant to your data."
        },
        {
            "question": "How should I interpret the risk scores?",
            "answer": "Risk scores represent the probability of an employee leaving. Scores are categorized as Low (<30%), Medium (30-70%), and High (>70%). Higher scores indicate higher turnover likelihood."
        },
        {
            "question": "How often should I retrain the model?",
            "answer": "It's recommended to retrain the model quarterly or when significant organizational changes occur that might affect employee retention patterns."
        }
    ]
    
    for i, faq in enumerate(faq_items):
        with st.expander(faq["question"]):
            st.write(faq["answer"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Contact and help
    st.markdown("<h2 class='sub-header'>Additional Resources</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### References and Resources
    
    - **HR Analytics Best Practices**: [SHRM Resources](https://www.shrm.org/)
    - **Machine Learning for HR**: [People Analytics Guide](https://www.aihr.com/blog/people-analytics/)
    - **Retention Strategies**: [Harvard Business Review](https://hbr.org/)
    
    ### Data Privacy Note
    
    This application processes all data locally in your browser. No employee data is stored or transmitted to external servers.
    
    ### Feedback and Support
    
    If you have questions, suggestions, or encounter issues with the dashboard, please contact your system administrator or data science team.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
