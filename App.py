# ============================================================================
# STREAMLIT INTERACTIVE DATA ANALYSIS DASHBOARD
# ============================================================================
# This application provides comprehensive exploratory data analysis (EDA)
# features for CSV datasets including visualizations, statistics, and
# interactive filtering options.
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
    <style>
    .header-text { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .subheader-text { font-size: 1.5rem; font-weight: bold; color: #ff7f0e; }
    .info-box { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
# Store data in session state to prevent reloading on every interaction
if 'df' not in st.session_state:
    st.session_state.df = None

# ============================================================================
# MAIN TITLE
# ============================================================================
st.markdown('<p class="header-text">📊 Interactive Data Analysis Dashboard</p>', 
            unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# DATA LOADING
# ============================================================================
# Load the Economic Freedom dataset
try:
    st.session_state.df = pd.read_csv("Economic Freedom.csv")
except FileNotFoundError:
    st.error("❌ Error: 'Economic Freedom.csv' not found in the current directory.")
    st.stop()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("## 🎯 Dashboard Configuration")
st.sidebar.markdown("📊 Economic Freedom Index Analysis")

df = st.session_state.df

# ============================================================================
# DATASET OVERVIEW SECTION
# ============================================================================
st.markdown('<p class="subheader-text">📈 Dataset Overview</p>', 
            unsafe_allow_html=True)

# Display dataset metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Columns", f"{df.shape[1]}")
with col3:
    st.metric("Missing Values", f"{df.isnull().sum().sum()}")
with col4:
    st.metric("Data Size (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

# Display the full dataset
st.markdown("#### 🗂️ Dataset Preview")
st.dataframe(df, use_container_width=True, height=300)

# ============================================================================
# DATASET INFORMATION SECTION
# ============================================================================
st.markdown("#### 📋 Column Information")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("**Column Names & Data Types:**")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(info_df, use_container_width=True)

with col_info2:
    st.markdown("**Summary Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

# ============================================================================
# MISSING VALUES ANALYSIS
# ============================================================================
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    st.markdown("#### ⚠️ Missing Values Analysis")
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Percentage': (missing_data.values / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    st.dataframe(missing_df, use_container_width=True)
else:
    st.info("✅ No missing values detected in the dataset!")

# ============================================================================
# SIDEBAR: COLUMN SELECTION
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Column Selection for EDA")

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Column selection options
selected_numeric = st.sidebar.multiselect(
    "Select Numeric Columns",
    numeric_cols,
    default=numeric_cols[:3] if numeric_cols else []
)

selected_categorical = st.sidebar.multiselect(
    "Select Categorical Columns",
    categorical_cols,
    default=categorical_cols[:2] if categorical_cols else []
)

# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) SECTION
# ============================================================================
st.markdown("---")
st.markdown('<p class="subheader-text">🔬 Exploratory Data Analysis (EDA)</p>', 
            unsafe_allow_html=True)

# ============================================================================
# NUMERIC COLUMNS ANALYSIS
# ============================================================================
if selected_numeric:
    st.markdown("#### 📊 Numeric Columns Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Histograms", "Box Plots", "Statistical Summary"])
    
    # Tab 1: Histograms
    with tab1:
        st.markdown("**Histograms for Numeric Columns**")
        num_cols_display = len(selected_numeric)
        cols_per_row = 2
        
        for i in range(0, num_cols_display, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < num_cols_display:
                    col_name = selected_numeric[i + j]
                    with col:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(df[col_name].dropna(), bins=30, color='steelblue', 
                               edgecolor='black', alpha=0.7)
                        ax.set_title(f"Distribution of {col_name}", fontsize=12, fontweight='bold')
                        ax.set_xlabel(col_name)
                        ax.set_ylabel("Frequency")
                        ax.grid(axis='y', alpha=0.3)
                        st.pyplot(fig, use_container_width=True)
    
    # Tab 2: Box Plots (Outlier Detection)
    with tab2:
        st.markdown("**Box Plots for Outlier Detection**")
        num_cols_display = len(selected_numeric)
        cols_per_row = 2
        
        for i in range(0, num_cols_display, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < num_cols_display:
                    col_name = selected_numeric[i + j]
                    with col:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.boxplot(df[col_name].dropna(), vert=True, patch_artist=True,
                                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2))
                        ax.set_title(f"Box Plot of {col_name}", fontsize=12, fontweight='bold')
                        ax.set_ylabel(col_name)
                        ax.grid(axis='y', alpha=0.3)
                        st.pyplot(fig, use_container_width=True)
    
    # Tab 3: Statistical Summary
    with tab3:
        st.markdown("**Detailed Statistics for Selected Numeric Columns**")
        stats_df = df[selected_numeric].describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['q3-q1'] = stats_df['75%'] - stats_df['25%']
        st.dataframe(stats_df, use_container_width=True)
else:
    st.info("👈 Select numeric columns from the sidebar to see analysis.")

# ============================================================================
# CATEGORICAL COLUMNS ANALYSIS
# ============================================================================
if selected_categorical:
    st.markdown("#### 📊 Categorical Columns Analysis")
    
    cat_cols_display = len(selected_categorical)
    cols_per_row = 2
    
    for i in range(0, cat_cols_display, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < cat_cols_display:
                col_name = selected_categorical[i + j]
                with col:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Get top 10 categories to avoid overcrowding
                    value_counts = df[col_name].value_counts().head(10)
                    colors = sns.color_palette("husl", len(value_counts))
                    
                    value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
                    ax.set_title(f"Distribution of {col_name}", fontsize=12, fontweight='bold')
                    ax.set_xlabel(col_name)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha='right')
                    ax.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
else:
    st.info("👈 Select categorical columns from the sidebar to see analysis.")

# ============================================================================
# CORRELATION HEATMAP
# ============================================================================
if len(selected_numeric) > 1:
    st.markdown("#### 🔗 Correlation Analysis")
    
    # Calculate correlation matrix
    corr_matrix = df[selected_numeric].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
else:
    st.info("📌 Select at least 2 numeric columns to see correlation analysis.")

# ============================================================================
# DATA EXPORT SECTION
# ============================================================================
st.markdown("---")
st.markdown("#### 💾 Export Options")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Export filtered dataset
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv_data,
        file_name="analyzed_data.csv",
        mime="text/csv"
    )

with col_exp2:
    # Export summary statistics
    summary_stats = df[selected_numeric].describe() if selected_numeric else pd.DataFrame()
    if not summary_stats.empty:
        csv_stats = summary_stats.to_csv().encode('utf-8')
        st.download_button(
            label="📊 Download Summary Statistics",
            data=csv_stats,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.85rem;'>
    <p>📊 Interactive Data Analysis Dashboard | Built with Streamlit</p>
    <p>✨ Tip: Use the sidebar to customize your analysis!</p>
    </div>
    """, unsafe_allow_html=True)
