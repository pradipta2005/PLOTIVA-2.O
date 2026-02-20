"""
User Guide Module for Plotiva
Comprehensive help and documentation system
"""

import streamlit as st


def render_user_guide():
    """Render comprehensive user guide"""
    
    st.header("📚 Plotiva User Guide")
    st.markdown("*Your complete guide to mastering data analysis with Plotiva*")
    
    # Create guide tabs
    guide_tabs = st.tabs([
        "🚀 Getting Started", 
        "📊 Data Upload", 
        "📈 Visualizations", 
        "🤖 Machine Learning",
        "🔧 Data Processing",
        "💡 Tips & Tricks",
        "❓ FAQ"
    ])
    
    with guide_tabs[0]:  # Getting Started
        st.subheader("🚀 Getting Started with Plotiva")
        
        st.markdown("""
        Welcome to **Plotiva** - your professional data analysis platform! This guide will help you get the most out of all features.
        
        ### What is Plotiva?
        Plotiva is a comprehensive data analysis platform that combines:
        - **Interactive Visualizations** - Create stunning charts and graphs
        - **Machine Learning** - Build predictive models with ease
        - **Data Processing** - Clean and transform your data
        - **AI Insights** - Get automated recommendations
        
        ### Quick Start (3 Steps)
        1. **Upload Data** - Use the sidebar to upload CSV, Excel, or other formats
        2. **Explore** - Navigate through tabs to analyze your data
        3. **Visualize** - Create charts and build dashboards
        """)
        
        # Feature overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 Key Features
            - ✅ **Universal File Support** - CSV, Excel, JSON, Parquet
            - ✅ **Real-time Filtering** - Dynamic data exploration
            - ✅ **Professional Charts** - 15+ chart types
            - ✅ **Machine Learning** - AutoML capabilities
            - ✅ **Interactive Dashboards** - Custom layouts
            - ✅ **AI-Powered Insights** - Smart recommendations
            """)
        
        with col2:
            st.markdown("""
            #### 🛠️ System Requirements
            - **Python 3.8+** - Modern Python version
            - **4GB RAM** - For medium datasets
            - **Modern Browser** - Chrome, Firefox, Safari
            - **Internet Connection** - For initial setup
            
            #### 📱 Supported Formats
            - **CSV** - Comma-separated values
            - **Excel** - .xlsx, .xls files
            - **JSON** - JavaScript Object Notation
            - **Parquet** - Columnar storage format
            """)
    
    with guide_tabs[1]:  # Data Upload
        st.subheader("📊 Data Upload Guide")
        
        st.markdown("""
        ### Uploading Your Data
        
        Plotiva supports multiple data formats and provides flexible upload options.
        """)
        
        # Upload methods
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📁 File Upload Methods
            
            **1. Drag & Drop**
            - Simply drag your file to the upload area
            - Supports all major formats
            - Instant validation and preview
            
            **2. Browse & Select**
            - Click "Browse files" button
            - Navigate to your data file
            - Select and upload
            
            **3. Sample Data**
            - Try built-in sample datasets
            - Perfect for learning and testing
            - Includes sales, customer, and financial data
            """)
        
        with col2:
            st.markdown("""
            #### ✅ File Requirements
            
            **Size Limits**
            - Maximum file size: 200MB
            - Recommended: Under 50MB for best performance
            
            **Format Guidelines**
            - **CSV**: Use comma separators, UTF-8 encoding
            - **Excel**: Modern .xlsx format preferred
            - **JSON**: Flat structure works best
            - **Parquet**: Optimized for large datasets
            
            **Data Quality Tips**
            - Include column headers
            - Avoid special characters in column names
            - Use consistent date formats
            """)
        
        st.markdown("""
        ### 🔍 Data Validation
        
        After upload, Plotiva automatically:
        - **Validates** file format and structure
        - **Analyzes** data types and quality
        - **Detects** missing values and duplicates
        - **Suggests** data cleaning operations
        
        ### 🎛️ Data Preview
        
        The Data Overview tab shows:
        - First 20 rows of your data
        - Column types and statistics
        - Data quality metrics
        - Missing value analysis
        """)
    
    with guide_tabs[2]:  # Visualizations
        st.subheader("📈 Visualization Guide")
        
        st.markdown("""
        ### Creating Stunning Visualizations
        
        Plotiva offers 15+ chart types with professional styling and interactive features.
        """)
        
        # Chart types overview
        chart_info = {
            "📊 Histogram": "Distribution of numeric variables",
            "🔍 Scatter Plot": "Relationships between two variables",
            "📈 Line Chart": "Trends over time or sequence",
            "📊 Bar Chart": "Comparisons across categories",
            "📦 Box Plot": "Distribution summaries by group",
            "🎻 Violin Plot": "Detailed distribution shapes",
            "🍰 Pie Chart": "Proportions and percentages",
            "🔥 Heatmap": "Correlation matrices",
            "🌐 3D Scatter": "Three-dimensional relationships",
            "🎬 Animated Plots": "Time-based animations"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Basic Charts")
            for chart, desc in list(chart_info.items())[:5]:
                st.markdown(f"**{chart}**  \n{desc}")
        
        with col2:
            st.markdown("#### 🎨 Advanced Charts")
            for chart, desc in list(chart_info.items())[5:]:
                st.markdown(f"**{chart}**  \n{desc}")
        
        st.markdown("""
        ### 🎨 Customization Options
        
        **Color Themes**
        - Aurora, Sunset, Ocean, Galaxy themes
        - Professional color palettes
        - Consistent styling across charts
        
        **Interactive Features**
        - Zoom and pan capabilities
        - Hover tooltips with details
        - Click-to-filter functionality
        - Export options (PNG, SVG, PDF)
        
        ### 📋 Dashboard Creation
        
        1. **Create Charts** - Build individual visualizations
        2. **Add to Dashboard** - Select charts for dashboard
        3. **Customize Layout** - Arrange in grid format
        4. **Export & Share** - Download or share dashboards
        """)
    
    with guide_tabs[3]:  # Machine Learning
        st.subheader("🤖 Machine Learning Guide")
        
        st.markdown("""
        ### Automated Machine Learning
        
        Plotiva makes machine learning accessible with automated model selection and tuning.
        """)
        
        ml_types = {
            "📈 Regression": {
                "description": "Predict continuous values",
                "algorithms": ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
                "use_cases": ["Sales forecasting", "Price prediction", "Performance metrics"]
            },
            "🎯 Classification": {
                "description": "Predict categories or classes",
                "algorithms": ["Logistic Regression", "Random Forest", "SVM", "Neural Networks"],
                "use_cases": ["Customer segmentation", "Fraud detection", "Quality control"]
            },
            "🔍 Clustering": {
                "description": "Find hidden patterns in data",
                "algorithms": ["K-Means", "DBSCAN", "Hierarchical"],
                "use_cases": ["Market segmentation", "Anomaly detection", "Data exploration"]
            }
        }
        
        for ml_type, info in ml_types.items():
            with st.expander(f"{ml_type} - {info['description']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Available Algorithms:**")
                    for algo in info['algorithms']:
                        st.markdown(f"• {algo}")
                
                with col2:
                    st.markdown("**Common Use Cases:**")
                    for use_case in info['use_cases']:
                        st.markdown(f"• {use_case}")
        
        st.markdown("""
        ### ⚡ AutoML Features
        
        **Automated Model Selection**
        - Tests multiple algorithms automatically
        - Compares performance metrics
        - Selects best performing model
        
        **Hyperparameter Tuning**
        - Optimizes model parameters
        - Uses cross-validation
        - Prevents overfitting
        
        **Model Evaluation**
        - Comprehensive performance metrics
        - Visualization of results
        - Feature importance analysis
        """)
    
    with guide_tabs[4]:  # Data Processing
        st.subheader("🔧 Data Processing Guide")
        
        st.markdown("""
        ### Data Cleaning & Transformation
        
        Prepare your data for analysis with powerful processing tools.
        """)
        
        processing_features = {
            "🔍 Data Quality Assessment": [
                "Completeness analysis",
                "Consistency checking", 
                "Validity verification",
                "Overall quality scoring"
            ],
            "🧹 Data Cleaning": [
                "Missing value imputation",
                "Duplicate removal",
                "Outlier detection and removal",
                "Data type corrections"
            ],
            "🔧 Feature Engineering": [
                "Create new calculated columns",
                "Mathematical transformations",
                "Date/time feature extraction",
                "Categorical encoding"
            ],
            "🎛️ Data Filtering": [
                "Real-time filtering",
                "Multiple column filters",
                "Range and category selection",
                "Dynamic chart updates"
            ]
        }
        
        for feature, capabilities in processing_features.items():
            with st.expander(feature):
                for capability in capabilities:
                    st.markdown(f"• {capability}")
        
        st.markdown("""
        ### 🎯 Best Practices
        
        **Data Quality**
        1. Always check data quality metrics first
        2. Address missing values before analysis
        3. Remove or investigate outliers
        4. Ensure consistent data types
        
        **Feature Engineering**
        1. Create meaningful derived features
        2. Normalize or scale numeric features
        3. Encode categorical variables properly
        4. Consider interaction terms
        """)
    
    with guide_tabs[5]:  # Tips & Tricks
        st.subheader("💡 Tips & Tricks")
        
        st.markdown("""
        ### Pro Tips for Better Analysis
        
        Master these techniques to become a Plotiva power user!
        """)
        
        tips_categories = {
            "🚀 Performance Tips": [
                "Use data sampling for large datasets (>100k rows)",
                "Apply filters to reduce data size before complex operations",
                "Close unused tabs to free up memory",
                "Use Parquet format for faster loading of large files"
            ],
            "📊 Visualization Tips": [
                "Choose appropriate chart types for your data",
                "Use color strategically to highlight insights",
                "Add meaningful titles and labels",
                "Consider your audience when designing charts"
            ],
            "🤖 ML Tips": [
                "Start with simple models before trying complex ones",
                "Always validate your models on unseen data",
                "Check feature importance to understand your model",
                "Use cross-validation for reliable performance estimates"
            ],
            "🔧 Data Tips": [
                "Clean your data before analysis",
                "Understand your data types and distributions",
                "Look for patterns in missing data",
                "Document your data processing steps"
            ]
        }
        
        for category, tips in tips_categories.items():
            st.markdown(f"#### {category}")
            for tip in tips:
                st.markdown(f"💡 {tip}")
            st.markdown("")
        
        st.markdown("""
        ### 🎯 Workflow Recommendations
        
        **Typical Analysis Workflow:**
        1. **Upload & Explore** - Load data and understand structure
        2. **Clean & Process** - Handle missing values and outliers
        3. **Visualize** - Create exploratory charts
        4. **Filter & Focus** - Narrow down to interesting subsets
        5. **Model & Predict** - Apply machine learning if needed
        6. **Dashboard** - Create summary dashboard
        7. **Export & Share** - Save results and insights
        """)
    
    with guide_tabs[6]:  # FAQ
        st.subheader("❓ Frequently Asked Questions")
        
        faqs = {
            "🔧 Technical Questions": {
                "What file formats are supported?": "CSV, Excel (.xlsx, .xls), JSON, and Parquet files are fully supported.",
                "What's the maximum file size?": "200MB is the limit, but we recommend files under 50MB for optimal performance.",
                "Can I use Plotiva offline?": "Yes! Once installed, Plotiva runs entirely on your local machine.",
                "Is my data secure?": "Absolutely! All data processing happens locally on your computer - nothing is sent to external servers."
            },
            "📊 Data & Analysis": {
                "How do I handle missing data?": "Use the Data Processing tab to impute missing values with mean, median, mode, or remove rows with missing data.",
                "Can I create custom calculations?": "Yes! Use the Feature Engineering section to create new columns with mathematical operations.",
                "How accurate are the ML models?": "Model accuracy depends on your data quality and size. Plotiva provides detailed metrics to help you evaluate performance.",
                "Can I export my charts?": "Yes! Charts can be exported as PNG, SVG, or PDF files directly from the interface."
            },
            "🎨 Visualization": {
                "How do I customize chart colors?": "Use the theme selector in the sidebar to choose from professional color palettes.",
                "Can I create animated charts?": "Yes! The Advanced Visualizations tab includes animated scatter plots and time-series animations.",
                "How do I add charts to a dashboard?": "Create charts in the Visualizations tab, then use the 'Add to Dashboard' button to include them in your dashboard.",
                "Can I share my dashboards?": "You can export dashboard data and generate reports that can be shared with others."
            }
        }
        
        for category, questions in faqs.items():
            st.markdown(f"### {category}")
            for question, answer in questions.items():
                with st.expander(f"❓ {question}"):
                    st.markdown(answer)
        
        st.markdown("""
        ### 🆘 Need More Help?
        
        **Troubleshooting Steps:**
        1. Check the deployment report for any setup issues
        2. Ensure all required packages are installed
        3. Try with a smaller sample of your data
        4. Restart the application if you encounter errors
        
        **Getting Support:**
        - Check the console output for error messages
        - Try the sample datasets to isolate issues
        - Review the user guide sections above
        - Ensure your data meets the format requirements
        """)

if __name__ == "__main__":
    render_user_guide()