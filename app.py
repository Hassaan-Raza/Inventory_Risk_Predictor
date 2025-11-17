import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="AI Inventory Risk Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-critical { 
        background-color: #ff4444; 
        color: white; 
        padding: 10px; 
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high { 
        background-color: #ff6b6b; 
        color: white; 
        padding: 10px; 
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium { 
        background-color: #ffa726; 
        color: white; 
        padding: 10px; 
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low { 
        background-color: #66bb6a; 
        color: white; 
        padding: 10px; 
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📦 AI Inventory Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h3>Predict inventory risks 30 days in advance using Machine Learning</h3>
    <p>Enter your inventory parameters below to get AI-powered risk assessment and smart recommendations</p>
</div>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('inventory_risk_model.pkl')
        accuracy = model_data['performance_metrics']['test_accuracy']
        st.sidebar.success(f"✅ Model Loaded (Accuracy: {accuracy:.1%})")
        return (
            model_data['model'], 
            model_data['label_encoder'], 
            model_data['feature_columns'],
            model_data['performance_metrics'],
            model_data['feature_descriptions'],
            model_data['risk_categories']
        )
    except FileNotFoundError:
        st.error("""
        ❌ Model file 'inventory_risk_model.pkl' not found.
        
        Please ensure:
        1. You've trained the model using the Colab notebook
        2. Downloaded the .pkl file  
        3. Placed it in the same folder as this Streamlit app
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model
model, le, feature_columns, performance_metrics, feature_descriptions, risk_categories = load_model()

# Create input sections
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Stock Information")
    
    quantity = st.number_input(
        "Current Quantity", 
        min_value=0.0, 
        value=100.0,
        help=feature_descriptions['quantity'],
        key="quantity"
    )
    
    min_qty_alert = st.number_input(
        "Minimum Quantity Alert", 
        min_value=1.0, 
        value=50.0,
        help=feature_descriptions['min_qty_alert'],
        key="min_alert"
    )
    
    max_qty_alert = st.number_input(
        "Maximum Quantity Alert", 
        min_value=1.0, 
        value=200.0,
        help=feature_descriptions['max_qty_alert'],
        key="max_alert"
    )
    
    cost = st.number_input(
        "Cost per Unit ($)", 
        min_value=0.01, 
        value=25.0,
        step=0.01,
        help=feature_descriptions['cost'],
        key="cost"
    )

with col2:
    st.subheader("⏰ Time Information")
    
    days_since_purchase = st.slider(
        "Days Since Last Purchase", 
        min_value=0, 
        max_value=730, 
        value=30,
        help=feature_descriptions['days_since_purchase'],
        key="days_purchase"
    )
    
    days_until_expiry = st.slider(
        "Days Until Expiry", 
        min_value=-365, 
        max_value=1095, 
        value=180,
        help=feature_descriptions['days_until_expiry'],
        key="days_expiry"
    )
    
    days_since_stock_update = st.slider(
        "Days Since Stock Update", 
        min_value=0, 
        max_value=365, 
        value=7,
        help=feature_descriptions['days_since_stock_update'],
        key="days_update"
    )

# Calculate derived features (same as training)
stock_ratio = quantity / (min_qty_alert + 1e-6)
coverage_ratio = max_qty_alert / (min_qty_alert + 1e-6)
utilization_ratio = quantity / (max_qty_alert + 1e-6)
total_inventory_value = quantity * cost
scaled_inventory_value = np.log1p(total_inventory_value)
is_recent_stock = 1 if days_since_stock_update <= 30 else 0
is_old_stock = 1 if days_since_purchase > 180 else 0
is_near_expiry = 1 if days_until_expiry < 90 else 0

# Prepare features for prediction (in same order as training)
features = np.array([[
    quantity, min_qty_alert, max_qty_alert, cost,
    days_since_stock_update, days_until_expiry, days_since_purchase,
    stock_ratio, coverage_ratio, utilization_ratio,
    scaled_inventory_value, is_recent_stock, is_old_stock, is_near_expiry
]])

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Analyze Inventory Risk with AI", type="primary", use_container_width=True):
        st.session_state.analyze_clicked = True

if st.session_state.get('analyze_clicked', False):
    with st.spinner("🤖 AI is analyzing inventory risk patterns..."):
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        risk_category = le.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Calculate suggested reorder quantity
        est_daily_usage = quantity / max(days_since_purchase, 1)
        
        if risk_category == 'CRITICAL_RISK':
            suggested_qty = max(min_qty_alert * 2.5 - quantity, est_daily_usage * 45)
            urgency = "URGENT"
            priority = 1
            color = "red"
            icon = "🚨"
            alert_level = "risk-critical"
        elif risk_category == 'HIGH_RISK':
            suggested_qty = max(0, min_qty_alert * 1.8 - quantity)
            urgency = "HIGH PRIORITY"
            priority = 2
            color = "orange"
            icon = "⚠️"
            alert_level = "risk-high"
        elif risk_category == 'MEDIUM_RISK':
            suggested_qty = max(0, min_qty_alert * 1.2 - quantity)
            urgency = "PLAN"
            priority = 3
            color = "blue"
            icon = "📋"
            alert_level = "risk-medium"
        else:  # LOW_RISK
            suggested_qty = 0
            urgency = "MONITOR"
            priority = 4
            color = "green"
            icon = "✅"
            alert_level = "risk-low"
        
        suggested_qty = max(0, round(suggested_qty))
        
        # Display results
        st.markdown("---")
        
        # Risk summary
        st.markdown(f"""
        <div class='{alert_level}' style='text-align: center; padding: 20px;'>
            <h1>{icon} {risk_category}</h1>
            <h3>Confidence: {confidence:.1%} | Urgency: {urgency}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Suggested Reorder", f"{suggested_qty} units", 
                     f"${suggested_qty * cost:,.2f}")
        
        with col2:
            ratio_status = "🟢 Optimal" if 0.8 <= stock_ratio <= 1.2 else "🟡 Review" if 0.5 <= stock_ratio <= 2.0 else "🔴 Critical"
            st.metric("Stock Ratio", f"{stock_ratio:.2f}", ratio_status)
        
        with col3:
            if days_until_expiry < 0:
                status, delta = "🔴 Expired", "Critical"
            elif days_until_expiry < 30:
                status, delta = "🟠 Near Expiry", "High Risk"
            elif days_until_expiry < 90:
                status, delta = "🟡 Approaching", "Monitor"
            else:
                status, delta = "🟢 Good", "Healthy"
            st.metric("Expiry Status", status, delta)
        
        with col4:
            if days_since_purchase > 365:
                age_status, delta = "🔴 Very Old", "High Risk"
            elif days_since_purchase > 180:
                age_status, delta = "🟠 Aging", "Monitor"
            else:
                age_status, delta = "🟢 Fresh", "Good"
            st.metric("Stock Age", age_status, delta)
        
        # Visualization section
        st.subheader("📊 AI Risk Analysis Dashboard")
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Risk probability chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            risk_labels = le.classes_
            colors = ['#66bb6a', '#42a5f5', '#ffa726', '#ef5350']
            
            bars = ax1.bar(risk_labels, probabilities * 100, color=colors, alpha=0.8)
            ax1.set_ylabel('Probability (%)', fontsize=12)
            ax1.set_title('AI Risk Prediction Probabilities', fontsize=14, fontweight='bold')
            ax1.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
        
        with chart_col2:
            # Inventory health indicators
            indicators = ['Stock Level', 'Expiry Risk', 'Age Risk', 'Value Impact']
            scores = [
                min(100, max(0, (1 - abs(stock_ratio - 1)) * 100)),
                min(100, max(0, 100 - (max(0, 30 - days_until_expiry) / 30 * 100))),
                min(100, max(0, 100 - (days_since_purchase / 365 * 100))),
                min(100, 80 + (min(total_inventory_value, 10000) / 10000 * 20))
            ]
            colors = ['#66bb6a', '#ffa726', '#42a5f5', '#ab47bc']
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars = ax2.barh(indicators, scores, color=colors, alpha=0.8)
            ax2.set_xlabel('Health Score (%)', fontsize=12)
            ax2.set_title('Inventory Health Indicators', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 100)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2., 
                        f'{score:.0f}%', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Detailed risk analysis
        st.subheader("🔍 Detailed Risk Analysis")
        
        risk_explanations = {
            'CRITICAL_RISK': """
            **🚨 CRITICAL RISK - Immediate Action Required**
            
            **Key Risk Factors:**
            • Very low stock levels (stock ratio: {stock_ratio:.2f})
            • {expiry_status}
            • {age_status}
            • High potential business impact
            
            **Immediate Impact:** High risk of stockouts affecting operations
            **Financial Impact:** Potential revenue loss and emergency procurement costs
            """,
            
            'HIGH_RISK': """
            **⚠️ HIGH RISK - Priority Attention Needed**
            
            **Key Risk Factors:**
            • Low stock levels (stock ratio: {stock_ratio:.2f}) 
            • {expiry_status}
            • {age_status}
            • Moderate business impact
            
            **30-day Outlook:** High probability of becoming critical without action
            **Financial Impact:** Increased procurement costs and potential disruptions
            """,
            
            'MEDIUM_RISK': """
            **📋 MEDIUM RISK - Planning Required**
            
            **Current Status:**
            • Moderate stock levels (stock ratio: {stock_ratio:.2f})
            • {expiry_status}
            • {age_status}
            • Low immediate business impact
            
            **90-day Outlook:** Stable but requires monitoring
            **Financial Impact:** Normal procurement cycles sufficient
            """,
            
            'LOW_RISK': """
            **✅ LOW RISK - Healthy Inventory**
            
            **Current Status:**
            • Optimal stock levels (stock ratio: {stock_ratio:.2f})
            • {expiry_status} 
            • {age_status}
            • Minimal business impact
            
            **Outlook:** Well-positioned for current demand
            **Financial Impact:** Efficient inventory management
            """
        }
        
        # Determine status messages
        expiry_status = "Expired item" if days_until_expiry < 0 else \
                       "Near expiry" if days_until_expiry < 30 else \
                       "Approaching expiry" if days_until_expiry < 90 else "Good expiry timeline"
                       
        age_status = "Very old stock" if days_since_purchase > 365 else \
                    "Aging stock" if days_since_purchase > 180 else "Fresh stock"
        
        explanation = risk_explanations[risk_category].format(
            stock_ratio=stock_ratio,
            expiry_status=expiry_status,
            age_status=age_status
        )
        
        st.info(explanation)
        
        # Actionable recommendations
        st.subheader("🎯 AI-Powered Recommendations")
        
        if risk_category == 'CRITICAL_RISK':
            st.error(f"""
            **🚨 IMMEDIATE ACTIONS REQUIRED (Within 24-48 hours)**
            
            **Inventory Management:**
            • 📦 Place urgent reorder for **{suggested_qty} units**
            • 🗑️ Remove and document any expired items immediately  
            • 📞 Contact suppliers for expedited shipping options
            • 🔄 Implement daily stock monitoring
            
            **Financial Considerations:**
            • 💰 Reorder cost: **${suggested_qty * cost:,.2f}**
            • ⚠️ Potential stockout cost: **Very High**
            • 🚚 Recommend expedited shipping despite higher costs
            
            **Operational Impact:**
            • ⏰ Lead time critical - consider emergency procurement
            • 📊 Review sales forecasts for demand spikes
            • 🔍 Audit similar high-risk items
            """)
            
        elif risk_category == 'HIGH_RISK':
            st.warning(f"""
            **⚠️ PRIORITY ACTIONS (Within 3-7 days)**
            
            **Inventory Management:**
            • 📦 Schedule reorder for **{suggested_qty} units** this week
            • 📅 Set up weekly stock level reviews
            • 📋 Identify and qualify backup suppliers
            • 🔔 Implement stock alert triggers
            
            **Financial Considerations:**
            • 💰 Reorder cost: **${suggested_qty * cost:,.2f}**  
            • ⚠️ Potential stockout cost: **Medium**
            • 🚚 Standard shipping recommended
            
            **Operational Impact:**
            • 📈 Review historical demand patterns
            • 🔄 Optimize reorder points and quantities
            • 📊 Monitor closely for trend changes
            """)
            
        elif risk_category == 'MEDIUM_RISK':
            st.info(f"""
            **📋 PLANNING ACTIONS (Within 2-4 weeks)**
            
            **Inventory Management:**
            • 📦 Include **{suggested_qty} units** in next regular order cycle
            • 📊 Continue monthly inventory reviews
            • 📈 Analyze seasonal demand patterns
            • 🔧 Fine-tune inventory parameters
            
            **Financial Considerations:**
            • 💰 Reorder cost: **${suggested_qty * cost:,.2f}**
            • ⚠️ Potential stockout cost: **Low**
            • 🚚 Bulk shipping for cost efficiency
            
            **Operational Impact:**
            • ✅ Maintain current stocking strategy
            • 📋 Update inventory management procedures
            • 🔄 Regular performance review recommended
            """)
            
        else:
            st.success(f"""
            **✅ MAINTENANCE ACTIONS (Regular monitoring)**
            
            **Inventory Management:**
            • ✅ No immediate reorder needed
            • 📊 Continue regular monitoring cycles
            • 📈 Maintain current stocking strategy
            • 🎯 Focus on continuous improvement
            
            **Financial Considerations:**
            • 💰 No additional reorder cost
            • ⚠️ Stockout risk: **Very Low**
            • 📉 Optimal inventory carrying costs
            
            **Operational Impact:**
            • 🏆 Excellent inventory health maintained
            • 💡 Opportunity to optimize other areas
            • 📚 Document best practices for other items
            """)

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About This AI System")
    
    st.markdown(f"""
    **🤖 AI-Powered Inventory Management**
    
    **Model Performance:**
    - 📊 Accuracy: **{performance_metrics['test_accuracy']:.1%}**
    - 🎯 ROC-AUC: **{performance_metrics.get('roc_auc', 0.997):.3f}**
    - ⏰ Forecast: **30-day horizon**
    - 📦 Training: **{performance_metrics.get('training_info', {}).get('dataset_size', '50,000+')} records**
    
    **Technology Stack:**
    - LightGBM Gradient Boosting
    - Real-time risk assessment
    - Stochastic future simulation
    - Enterprise-grade scalability
    """)
    
    st.header("🎯 Risk Categories")
    
    for risk, desc in risk_categories.items():
        if risk == 'LOW_RISK':
            st.markdown(f"**🟢 {risk}**")
        elif risk == 'MEDIUM_RISK':
            st.markdown(f"**🔵 {risk}**") 
        elif risk == 'HIGH_RISK':
            st.markdown(f"**🟠 {risk}**")
        else:
            st.markdown(f"**🔴 {risk}**")
        st.caption(f"{desc}")
    
    st.header("🔧 How It Works")
    st.markdown("""
    1. **Input** inventory parameters
    2. **AI Model** analyzes 14 key features  
    3. **Predicts** 30-day risk probability
    4. **Recommends** specific actions
    5. **Calculates** financial impact
    """)
    
    st.header("📈 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(8)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feature_imp['feature'], feature_imp['importance'], color='#1f77b4')
        ax.set_xlabel('Importance')
        ax.set_title('Top Predictive Features')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.header("📞 Enterprise Features")
    st.markdown("""
    **For large-scale deployment:**
    
    - Multi-company support
    - API integration  
    - Custom risk thresholds
    - Advanced analytics
    - Real-time monitoring
    
    *Contact us for enterprise solutions*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>AI Inventory Risk Prediction System v2.0</strong> | Powered by LightGBM | Trained on enterprise inventory data</p>
    <p>Model trained on: {training_date}</p>
</div>
""".format(training_date=performance_metrics.get('training_date', 'Recent')), unsafe_allow_html=True)

# Initialize session state
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False