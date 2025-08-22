import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings
import hashlib
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="NPS Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Password Configuration
PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # "password" in SHA256
# To generate a new password hash, use: hashlib.sha256("your_password".encode()).hexdigest()

def verify_password(password):
    """Verify the entered password"""
    return hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH

def login_page():
    """Display login page"""
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 70vh;">
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center; max-width: 400px;">
            <h2>🔐 Secure Access Required</h2>
            <p>Please enter your password to access the NPS Dashboard</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("### Enter Password")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("🔓 Login", use_container_width=True)
        
        if login_button:
            if verify_password(password):
                st.session_state.authenticated = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid password. Access denied.")
                st.session_state.authenticated = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #ffc107; }
    .critical { color: #dc3545; font-weight: bold; }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def parse_csv_file(uploaded_file):
    """Parse CSV file with proper error handling"""
    try:
        # Try different encoding options
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Try to read as regular CSV first
                df = pd.read_csv(uploaded_file, encoding=encoding)
                
                # Check if it has the expected columns
                expected_cols = ['Timestamp', 'Rating', 'Reason']
                if all(col in df.columns for col in expected_cols):
                    return df
                
                # If not, try to parse as the malformed format
                uploaded_file.seek(0)
                content = uploaded_file.read().decode(encoding)
                lines = content.split('\n')
                
                data = []
                for line in lines[1:]:  # Skip header
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    
                    match = re.match(r'^([^,]+),(\d+),"?(.+?)"?$', line)
                    
                    if match:
                        timestamp, rating, reason = match.groups()
                        reason = reason.strip('"').strip()
                        
                        data.append({
                            'Timestamp': timestamp,
                            'Rating': int(rating),
                            'Reason': reason
                        })
                
                if data:
                    return pd.DataFrame(data)
                    
            except Exception as e:
                continue
        
        # If all fails, return empty DataFrame
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def ultra_enhanced_categorize_feedback(reason):
    """Ultra-enhanced categorization function"""
    if pd.isna(reason) or not reason.strip():
        return '⚫ Generic: No Feedback Provided'
    
    reason_lower = str(reason).lower()
    reason_clean = re.sub(r'[^\w\s]', ' ', reason_lower)
    
    # Generic responses
    if re.match(r'^(10|ok|good|x|\.|\-|n|NIL|Ban|yes|100|Super|asfaf|hjgvkutdjtydch|dsf|ptroto|Na|\.\.\.|9|8|7|6|5|4|3|2|1|top|best|--|okey|God)$', reason_lower.strip()):
        return '⚫ Generic: Non-Descriptive Response'
    
    # Critical system issues
    if any(term in reason_lower for term in [
        'doesnt work', 'does not work', 'not working', 'it doesn\'t work', 'dont work',
        'site is offline', 'offline', 'crash', 'error', 'bug', 'frozen', 'freeze',
        'white page', 'blank screen', 'no response', 'not open', 'does not open'
    ]):
        return '🔴 Critical: System Not Functioning'
    
    if any(term in reason_lower for term in [
        'login', 'cant login', 'wrong mail', 'email', 'password', 'register',
        'authentication', 'auth issues', 'disappeared', 'cant get', 'sign up',
        'account', 'access'
    ]):
        return '🔴 Critical: Authentication/Access Issues'
    
    # Performance issues
    if any(term in reason_lower for term in [
        'slow', 'slowness', 'super slow', 'too slow', 'loading', 'too long',
        'lenteur', 'traag', 'performance'
    ]):
        return '🟡 Performance: Speed/Loading Issues'
    
    # UX Issues
    if any(term in reason_lower for term in [
        'not user friendly', 'not intuitive', 'bloody impossible', 'difficult',
        'hard', 'complicated', 'obnoxious', 'rubbish', 'ux from the 90s',
        'not easy', 'impossible', 'confusing', 'unclear', 'messy'
    ]):
        return '🟠 UX: Poor Usability/Interface'
    
    if any(term in reason_lower for term in [
        'zero help', 'no help', 'too much info', 'help', 'guidance',
        'documentation', 'tutorial', 'instructions', 'support'
    ]):
        return '🟠 UX: Lack of Help/Documentation'
    
    # Tool specific
    if any(term in reason_lower for term in [
        'calculator', 'calc', 'blubase', 'esdec', 'old calculator', 'new website',
        'changed', 'transition', 'migration', 'old site', 'previous'
    ]):
        return '🟡 Tool: Calculator/Platform Transition'
    
    if any(term in reason_lower for term in [
        'design', 'drawing', 'draw', 'roof', 'can\'t make', 'manual',
        'panel', 'layout', 'configurator', 'calepinage', 'building',
        'project', 'installation'
    ]):
        return '🟡 Tool: Design/Drawing Functionality'
    
    # Features
    if any(term in reason_lower for term in [
        'flatfix', 'wave plus', 'can\'t select', 'cant select', 'select', 'choose',
        'product selection', 'item selection', 'picking', 'selecting'
    ]):
        return '🟣 Product: Selection/Availability Issues'
    
    if any(term in reason_lower for term in [
        'mounting', 'installation', 'screw', 'threaded', 'canal tiles',
        'explicit', 'specification', 'technical details', 'mounting system'
    ]):
        return '🟣 Product: Technical Specifications'
    
    # Localization
    if any(term in reason_lower for term in [
        'english', 'translate', 'language', 'dutch', 'spanish', 'français',
        'deutsch', 'nederlands', 'idioma', 'translation'
    ]):
        return '🔵 Localization: Language/Translation'
    
    # Business/Industry
    if any(term in reason_lower for term in [
        'supplier', 'presentation', 'demo', 'showcase', 'exhibition',
        'trade show', 'meeting', 'conference'
    ]):
        return '🏭 Industry: Business/Professional Interaction'
    
    # Competition
    if any(term in reason_lower for term in [
        'k2', 'competitor', 'alternative', 'other', 'different', 'better',
        'worse', 'compare', 'comparison', 'versus', 'pleasant', 'prefer'
    ]):
        return '🔄 Comparison: Competitive Reference'
    
    # Positive feedback
    if any(term in reason_lower for term in [
        'everything top', 'alles top', 'gladly again', 'gerne wieder',
        'excellent', 'outstanding', 'exceptional', 'amazing', 'fantastic',
        'brilliant', 'superb', 'wonderful', 'magnificent', 'perfect'
    ]):
        return '🟢 Positive: Exceptional Satisfaction'
    
    if any(term in reason_lower for term in [
        'good', 'great', 'nice', 'fine', 'solid', 'decent', 'satisfactory',
        'pleased', 'happy', 'satisfied'
    ]):
        return '🟢 Positive: General Satisfaction'
    
    if any(term in reason_lower for term in [
        'easy', 'simple', 'intuitive', 'user-friendly', 'straightforward',
        'smooth', 'seamless', 'effortless', 'convenient'
    ]):
        return '🟢 Positive: Ease of Use'
    
    if any(term in reason_lower for term in [
        'fast', 'quick', 'rapid', 'speedy', 'efficient', 'prompt'
    ]):
        return '🟢 Positive: Speed/Efficiency'
    
    if any(term in reason_lower for term in [
        'quality', 'accurate', 'reliable', 'professional', 'complete',
        'comprehensive', 'detailed', 'thorough', 'precise'
    ]):
        return '🟢 Positive: Quality/Accuracy'
    
    # Status
    if any(term in reason_lower for term in [
        'haven\'t seen', 'havent seen', 'haven\'t used', 'havent used',
        'haven\'t tried', 'havent tried', 'no experience', 'dont know',
        'unfamiliar', 'beginning', 'starting', 'new', 'first time'
    ]):
        return '⚪ Status: New User/Exploring'
    
    if any(term in reason_lower for term in [
        'testing', 'trial', 'trying', 'evaluating', 'checking',
        'exploring', 'reviewing'
    ]):
        return '⚪ Status: Testing/Evaluation'
    
    # Negative
    if any(term in reason_lower for term in [
        'waste', 'useless', 'pointless', 'worthless', 'terrible',
        'awful', 'horrible', 'disaster', 'nightmare'
    ]):
        return '🔺 Negative: Strong Dissatisfaction'
    
    return '🔘 Other: Truly Uncategorized'

@st.cache_data
def process_dataframe(df):
    """Process the uploaded dataframe"""
    if df.empty:
        return df
    
    # Ensure proper column names
    if 'Timestamp' not in df.columns and len(df.columns) >= 3:
        df.columns = ['Timestamp', 'Rating', 'Reason']
    
    # Process timestamps
    try:
        if df['Timestamp'].dtype == 'object':
            # Try different timestamp formats
            timestamp_formats = [
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%d-%m-%Y %H:%M:%S',
                '%Y/%m/%d %H:%M:%S'
            ]
            
            for fmt in timestamp_formats:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=fmt)
                    break
                except:
                    continue
            
            # If no format works, try pandas default parser
            if df['Timestamp'].dtype == 'object':
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                
    except Exception as e:
        st.error(f"Error parsing timestamps: {str(e)}")
        return pd.DataFrame()
    
    # Ensure Rating is numeric
    try:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df.dropna(subset=['Rating'])
        df = df[(df['Rating'] >= 0) & (df['Rating'] <= 10)]
    except:
        st.error("Error processing ratings. Please ensure ratings are numeric values between 0-10.")
        return pd.DataFrame()
    
    # Create NPS categories
    def get_nps_category(rating):
        if rating >= 9: return 'Promoter'
        elif rating >= 7: return 'Passive'
        else: return 'Detractor'
    
    df['NPS_Category'] = df['Rating'].apply(get_nps_category)
    df['Category'] = df['Reason'].apply(ultra_enhanced_categorize_feedback)
    
    # Add time dimensions
    df['Month'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['Month_Year'] = df['Timestamp'].dt.to_period('M')
    
    # Sentiment analysis
    def get_sentiment(text):
        if pd.isna(text): return 0, 'Neutral'
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0.3: return polarity, 'Very Positive'
            elif polarity > 0.1: return polarity, 'Positive'
            elif polarity < -0.3: return polarity, 'Very Negative'
            elif polarity < -0.1: return polarity, 'Negative'
            else: return polarity, 'Neutral'
        except: return 0, 'Neutral'
    
    sentiment_results = df['Reason'].apply(get_sentiment)
    df['Sentiment_Score'] = [result[0] for result in sentiment_results]
    df['Sentiment_Label'] = [result[1] for result in sentiment_results]
    
    return df

def calculate_nps_metrics(df):
    """Calculate NPS metrics"""
    total = len(df)
    promoters = len(df[df['NPS_Category'] == 'Promoter'])
    detractors = len(df[df['NPS_Category'] == 'Detractor'])
    passives = len(df[df['NPS_Category'] == 'Passive'])
    
    nps_score = ((promoters - detractors) / total) * 100 if total > 0 else 0
    
    return {
        'nps_score': nps_score,
        'total': total,
        'promoters': promoters,
        'passives': passives,
        'detractors': detractors,
        'promoter_rate': (promoters / total) * 100 if total > 0 else 0,
        'passive_rate': (passives / total) * 100 if total > 0 else 0,
        'detractor_rate': (detractors / total) * 100 if total > 0 else 0
    }

def create_nps_gauge(nps_score):
    """Create NPS gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = nps_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "NPS Score"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, 0], 'color': "lightgray"},
                {'range': [0, 50], 'color': "gray"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def file_upload_section():
    """Display file upload section"""
    st.markdown('<h1 class="main-header">📊 NPS Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-box">
        <h3>📁 Upload Your NPS Data</h3>
        <p>Please upload your CSV file containing NPS feedback data to begin analysis.</p>
        <p><strong>Expected format:</strong> Timestamp, Rating (0-10), Reason</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your NPS feedback CSV file. Expected columns: Timestamp, Rating, Reason"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner('📈 Processing your data...'):
                df = parse_csv_file(uploaded_file)
                
                if df.empty:
                    st.error("❌ Could not process the file. Please check the format and try again.")
                    st.info("""
                    **Expected CSV format:**
                    - Column 1: Timestamp (various date formats supported)
                    - Column 2: Rating (0-10)
                    - Column 3: Reason/Feedback text
                    """)
                    return None
                
                processed_df = process_dataframe(df)
                
                if processed_df.empty:
                    st.error("❌ No valid data found after processing. Please check your data format.")
                    return None
                
                # Show data preview
                st.success(f"✅ Successfully loaded {len(processed_df)} responses!")
                
                with st.expander("📋 Data Preview", expanded=True):
                    st.write("**First 5 rows of your data:**")
                    preview_df = processed_df[['Timestamp', 'Rating', 'Reason', 'NPS_Category']].head()
                    st.dataframe(preview_df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Responses", len(processed_df))
                    with col2:
                        st.metric("Date Range", f"{processed_df['Timestamp'].min().strftime('%Y-%m-%d')} to {processed_df['Timestamp'].max().strftime('%Y-%m-%d')}")
                    with col3:
                        avg_rating = processed_df['Rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.1f}")
                
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    st.session_state.data_loaded = True
                    st.session_state.df = processed_df
                    st.rerun()
                    
                return processed_df
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and try again.")
            return None
    
    return None

# Main Application Logic
def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Authentication check
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Add logout button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🔐 Logout", type="secondary"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Data loading check
    if not st.session_state.data_loaded or st.session_state.df is None:
        file_upload_section()
        return
    
    # Main dashboard
    df = st.session_state.df
    
    st.markdown('<h1 class="main-header">🎯 NPS Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("📋 Dashboard Controls")
    
    # Data info
    st.sidebar.success(f"✅ {len(df)} responses loaded")
    
    # Option to load new data
    if st.sidebar.button("📁 Load New Data"):
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.rerun()
    
    # Calculate metrics
    metrics = calculate_nps_metrics(df)
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    try:
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(df['Timestamp'].min().date(), df['Timestamp'].max().date()),
            min_value=df['Timestamp'].min().date(),
            max_value=df['Timestamp'].max().date()
        )
    except:
        date_range = []
    
    # Filter data by date
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['Timestamp'].dt.date >= start_date) & 
            (df['Timestamp'].dt.date <= end_date)
        ]
        metrics_filtered = calculate_nps_metrics(df_filtered)
    else:
        df_filtered = df
        metrics_filtered = metrics
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="NPS Score",
            value=f"{metrics_filtered['nps_score']:.1f}",
            delta=f"{metrics_filtered['nps_score']:.1f}" if metrics_filtered['nps_score'] >= 0 else None
        )
    
    with col2:
        st.metric(
            label="Total Responses",
            value=metrics_filtered['total']
        )
    
    with col3:
        st.metric(
            label="Promoters",
            value=f"{metrics_filtered['promoters']} ({metrics_filtered['promoter_rate']:.1f}%)"
        )
    
    with col4:
        st.metric(
            label="Detractors",
            value=f"{metrics_filtered['detractors']} ({metrics_filtered['detractor_rate']:.1f}%)"
        )
    
    # NPS Status
    if metrics_filtered['nps_score'] > 50:
        status = "🟢 EXCELLENT - World-class performance!"
    elif metrics_filtered['nps_score'] > 30:
        status = "🟡 GOOD - Above industry average"
    elif metrics_filtered['nps_score'] > 0:
        status = "🟠 FAIR - Room for improvement"
    else:
        status = "🔴 CRITICAL - Immediate attention needed"
    
    st.markdown(f"### Status: {status}")
    
    # Charts section
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Trends", 
        "🔍 Categories", 
        "💭 Sentiment", 
        "📝 Feedback"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # NPS Gauge
            st.plotly_chart(create_nps_gauge(metrics_filtered['nps_score']), use_container_width=True)
        
        with col2:
            # NPS Distribution
            nps_counts = df_filtered['NPS_Category'].value_counts()
            fig_pie = px.pie(
                values=nps_counts.values,
                names=nps_counts.index,
                title="NPS Distribution",
                color_discrete_map={
                    'Promoter': '#2ecc71',
                    'Passive': '#f39c12',
                    'Detractor': '#e74c3c'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Rating Distribution
        fig_rating = px.histogram(
            df_filtered, 
            x='Rating',
            title="Rating Distribution",
            nbins=10,
            color_discrete_sequence=['#3498db']
        )
        fig_rating.update_layout(
            xaxis=dict(dtick=1, range=[0.5, 10.5]),
            bargap=0.1
        )
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with tab2:
        # Monthly trends
        if len(df_filtered) > 0:
            monthly_data = df_filtered.groupby('Month_Year').agg({
                'Rating': ['mean', 'count'],
                'NPS_Category': lambda x: ((x == 'Promoter').sum() - (x == 'Detractor').sum()) / len(x) * 100
            }).round(2)
            
            monthly_data.columns = ['Avg_Rating', 'Response_Count', 'NPS_Score']
            monthly_data = monthly_data.reset_index()
            monthly_data['Month_Year_Str'] = monthly_data['Month_Year'].astype(str)
            
            if len(monthly_data) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_trend = px.line(
                        monthly_data,
                        x='Month_Year_Str',
                        y='Avg_Rating',
                        title="Average Rating Trend",
                        markers=True
                    )
                    fig_trend.update_layout(yaxis=dict(range=[0, 10]))
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with col2:
                    fig_nps_trend = px.line(
                        monthly_data,
                        x='Month_Year_Str',
                        y='NPS_Score',
                        title="NPS Score Trend",
                        markers=True,
                        color_discrete_sequence=['purple']
                    )
                    fig_nps_trend.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
                    st.plotly_chart(fig_nps_trend, use_container_width=True)
                
                fig_volume = px.bar(
                    monthly_data,
                    x='Month_Year_Str',
                    y='Response_Count',
                    title="Monthly Response Volume"
                )
                st.plotly_chart(fig_volume, use_container_width=True)
            else:
                st.info("Need data from multiple months to show trends")
    
    with tab3:
        # Category analysis
        category_counts = df_filtered['Category'].value_counts().head(15)
        
        fig_categories = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 15 Feedback Categories"
        )
        fig_categories.update_layout(height=600)
        st.plotly_chart(fig_categories, use_container_width=True)
        
        # Category breakdown by NPS
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Top Detractor Issues")
            detractor_issues = df_filtered[df_filtered['NPS_Category'] == 'Detractor']['Category'].value_counts().head(5)
            for category, count in detractor_issues.items():
                percentage = (count / metrics_filtered['detractors']) * 100 if metrics_filtered['detractors'] > 0 else 0
                st.write(f"**{category}**: {count} ({percentage:.1f}%)")
        
        with col2:
            st.subheader("🟢 Top Promoter Reasons")
            promoter_reasons = df_filtered[df_filtered['NPS_Category'] == 'Promoter']['Category'].value_counts().head(5)
            for category, count in promoter_reasons.items():
                percentage = (count / metrics_filtered['promoters']) * 100 if metrics_filtered['promoters'] > 0 else 0
                st.write(f"**{category}**: {count} ({percentage:.1f}%)")
    
    with tab4:
        # Sentiment analysis
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df_filtered['Sentiment_Label'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            try:
                sentiment_nps = df_filtered.groupby(['NPS_Category', 'Sentiment_Label']).size().unstack(fill_value=0)
                fig_sentiment_nps = px.bar(
                    sentiment_nps,
                    title="Sentiment by NPS Category",
                    barmode='group'
                )
                st.plotly_chart(fig_sentiment_nps, use_container_width=True)
            except:
                st.write("Not enough data for sentiment by NPS breakdown")
        
        # Sentiment scores distribution
        fig_sentiment_dist = px.histogram(
            df_filtered,
            x='Sentiment_Score',
            title="Sentiment Score Distribution",
            nbins=30
        )
        fig_sentiment_dist.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.7)
        st.plotly_chart(fig_sentiment_dist, use_container_width=True)
    
    with tab5:
        # Raw feedback exploration
        st.subheader("🔍 Explore Feedback")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nps_filter = st.selectbox(
                "Filter by NPS Category",
                options=['All'] + list(df_filtered['NPS_Category'].unique())
            )
        
        with col2:
            category_filter = st.selectbox(
                "Filter by Category",
                options=['All'] + list(df_filtered['Category'].unique())
            )
        
        with col3:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=['All'] + list(df_filtered['Sentiment_Label'].unique())
            )
        
        # Apply filters
        feedback_df = df_filtered.copy()
        
        if nps_filter != 'All':
            feedback_df = feedback_df[feedback_df['NPS_Category'] == nps_filter]
        
        if category_filter != 'All':
            feedback_df = feedback_df[feedback_df['Category'] == category_filter]
        
        if sentiment_filter != 'All':
            feedback_df = feedback_df[feedback_df['Sentiment_Label'] == sentiment_filter]
        
        # Display feedback
        st.write(f"Showing {len(feedback_df)} feedback entries:")
        
        for idx, row in feedback_df.head(20).iterrows():
            with st.expander(f"Rating {row['Rating']} - {row['NPS_Category']} - {row['Timestamp'].strftime('%Y-%m-%d')}"):
                st.write(f"**Feedback**: {row['Reason']}")
                st.write(f"**Category**: {row['Category']}")
                st.write(f"**Sentiment**: {row['Sentiment_Label']} ({row['Sentiment_Score']:.2f})")
    
    # Action items sidebar
    st.sidebar.subheader("💡 Action Items")
    
    if metrics_filtered['nps_score'] < 0:
        st.sidebar.error("🚨 CRISIS MODE")
        st.sidebar.write("1. Fix critical system issues immediately")
        st.sidebar.write("2. Contact recent detractors")
        st.sidebar.write("3. Emergency customer recovery program")
    elif metrics_filtered['nps_score'] < 30:
        st.sidebar.warning("🔧 IMPROVEMENT NEEDED")
        st.sidebar.write("1. Address top technical issues")
        st.sidebar.write("2. Improve user documentation")
        st.sidebar.write("3. Focus on usability improvements")
    else:
        st.sidebar.success("📈 GOOD PERFORMANCE")
        st.sidebar.write("1. Scale what's working well")
        st.sidebar.write("2. Address remaining pain points")
        st.sidebar.write("3. Maintain consistency")
    
    # Export functionality
    if st.sidebar.button("📥 Export Analysis"):
        export_df = df_filtered[['Timestamp', 'Rating', 'Reason', 'NPS_Category', 'Category', 'Sentiment_Label', 'Sentiment_Score']]
        csv = export_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"nps_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
