import streamlit as st
import requests
import json
import calendar
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Hotel Reservation Prediction ❤️",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match your original styling with improvements
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #333;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .prediction-result {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    .cancel-prediction {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    
    .no-cancel-prediction {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    
    .confidence-bar {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
    }
    
    .confidence-fill {
        height: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #5a6fd8;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_days_in_month(month, year=2024):
    """Get the number of days in a given month"""
    return calendar.monthrange(year, month)[1]

def validate_inputs(data):
    """Validate input data"""
    errors = []
    
    if data['lead_time'] < 0:
        errors.append("Lead time cannot be negative")
    
    if data['avg_price_per_room'] <= 0:
        errors.append("Price per room must be greater than 0")
    
    if data['arrival_date'] > get_days_in_month(data['arrival_month']):
        errors.append(f"Invalid date: Month {data['arrival_month']} doesn't have {data['arrival_date']} days")
    
    total_nights = data['no_of_week_nights'] + data['no_of_weekend_nights']
    if total_nights == 0:
        errors.append("Total nights cannot be 0")
    
    return errors

@st.cache_data(ttl=60)  # Cache for 1 minute
def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Header
st.markdown('<h1 class="main-header">🏨 Hotel Reservation Prediction</h1>', unsafe_allow_html=True)

# API Health Check with better UX
health_data = check_api_health()

if not health_data:
    st.error("❌ Cannot connect to prediction API. Please ensure FastAPI server is running on port 8000.")
    st.code("python main.py", language="bash")
    st.stop()
elif not health_data.get("model_loaded", False):
    st.error("⚠️ Model is not loaded on the backend!")
    if st.button("🔄 Try Reload Model"):
        try:
            reload_response = requests.post(f"{API_URL}/reload-model", timeout=10)
            if reload_response.status_code == 200:
                st.success("✅ Model reload attempted. Please refresh the page.")
            else:
                st.error("❌ Failed to reload model")
        except:
            st.error("❌ Cannot communicate with API")
    st.stop()
else:
    st.success("✅ API is healthy and model is loaded!")

# Information section
with st.expander("ℹ️ About this prediction system"):
    st.markdown("""
    <div class="info-card">
        <strong>What does this predict?</strong><br>
        This system predicts whether a hotel guest is likely to cancel their reservation based on booking details.
        
        <br><br><strong>How it works:</strong><br>
        • Uses machine learning trained on historical hotel booking data<br>
        • Analyzes patterns in guest behavior and booking characteristics<br>
        • Provides confidence scores for predictions
    </div>
    """, unsafe_allow_html=True)

# Create form
with st.form("prediction_form"):
    st.subheader("📋 Enter Reservation Details")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Booking Information**")
        
        lead_time = st.number_input(
            "Lead Time (days)", 
            min_value=0,
            max_value=365,
            value=50,
            help="Number of days between booking and arrival"
        )
        
        no_of_special_request = st.number_input(
            "Number of Special Requests", 
            min_value=0,
            max_value=10,
            value=0,
            help="Total number of special requests made"
        )
        
        avg_price_per_room = st.number_input(
            "Average Price per Room ($)", 
            min_value=0.01,
            max_value=1000.0,
            value=100.0, 
            format="%.2f",
            help="Average price per room per night"
        )
        
        st.markdown("**📍 Arrival Details**")
        
        arrival_month = st.selectbox(
            "Arrival Month",
            options=list(range(1, 13)),
            format_func=lambda x: calendar.month_name[x],
            index=0,
            help="Month of arrival"
        )
        
        # Dynamic date selection based on month
        max_days = get_days_in_month(arrival_month)
        arrival_date = st.selectbox(
            "Arrival Date",
            options=list(range(1, max_days + 1)),
            index=0,
            help=f"{calendar.month_name[arrival_month]} has {max_days} days"
        )
    
    with col2:
        st.markdown("**🏢 Booking Details**")
        
        market_segment_type = st.selectbox(
            "Market Segment Type",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: ["Aviation", "Complimentary", "Corporate", "Offline", "Online"][x],
            index=4,
            help="How the booking was made"
        )
        
        type_of_meal_plan = st.selectbox(
            "Type of Meal Plan",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"][x],
            index=3,
            help="Selected meal plan type"
        )
        
        room_type_reserved = st.selectbox(
            "Room Type Reserved",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [f"Room Type {i+1}" for i in range(7)][x],
            index=0,
            help="Type of room reserved"
        )
        
        st.markdown("**🛏️ Stay Duration**")
        
        no_of_week_nights = st.number_input(
            "Number of Week Nights", 
            min_value=0,
            max_value=30,
            value=2,
            help="Number of weekday nights (Mon-Thu)"
        )
        
        no_of_weekend_nights = st.number_input(
            "Number of Weekend Nights", 
            min_value=0,
            max_value=10,
            value=1,
            help="Number of weekend nights (Fri-Sun)"
        )
    
    # Display summary
    total_nights = no_of_week_nights + no_of_weekend_nights
    total_cost = total_nights * avg_price_per_room
    
    st.markdown("---")
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("Total Nights", total_nights)
    with col_summary2:
        st.metric("Total Cost", f"${total_cost:.2f}")
    with col_summary3:
        st.metric("Lead Time", f"{lead_time} days")
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Reservation Status", use_container_width=True)
    
    if submitted:
        # Prepare prediction request
        prediction_data = {
            "lead_time": lead_time,
            "no_of_special_request": no_of_special_request,
            "avg_price_per_room": avg_price_per_room,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "no_of_week_nights": no_of_week_nights,
            "no_of_weekend_nights": no_of_weekend_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "room_type_reserved": room_type_reserved
        }
        
        # Validate inputs
        validation_errors = validate_inputs(prediction_data)
        if validation_errors:
            st.error("❌ Input validation failed:")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            # Make prediction request
            with st.spinner("🤔 Analyzing reservation data..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=prediction_data,
                        timeout=30,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        prediction_text = result["prediction_text"]
                        confidence = result.get("confidence")
                        model_info = result.get("model_info", {})
                        
                        # Display result with styling
                        if prediction == 0:
                            st.markdown(f"""
                            <div class="prediction-result cancel-prediction">
                                ❌ {prediction_text}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-result no-cancel-prediction">
                                ✅ {prediction_text}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show confidence if available
                        if confidence:
                            st.markdown("### 🎯 Prediction Confidence")
                            confidence_percent = confidence * 100
                            
                            # Create columns for confidence display
                            conf_col1, conf_col2 = st.columns([3, 1])
                            with conf_col1:
                                st.progress(confidence)
                            with conf_col2:
                                st.metric("Confidence", f"{confidence_percent:.1f}%")
                            
                            # Interpretation
                            if confidence >= 0.8:
                                st.success("🎯 High confidence prediction")
                            elif confidence >= 0.6:
                                st.info("🤔 Moderate confidence prediction")
                            else:
                                st.warning("⚠️ Low confidence prediction - consider additional factors")
                        
                        # Show model info
                        if model_info:
                            with st.expander("🔍 Model Information"):
                                st.json(model_info)
                                
                    elif response.status_code == 400:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Invalid input: {error_detail}")
                    elif response.status_code == 503:
                        st.error("❌ Model service unavailable. Please try again later.")
                    else:
                        st.error(f"❌ Prediction failed (Status: {response.status_code})")
                        st.error(response.text)
                        
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Please check if the server is running.")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

# Footer with additional info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🏨 Hotel Reservation Prediction System</p>
    <p>Powered by Streamlit + FastAPI</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        💡 <strong>Tip:</strong> Higher lead times and special requests may increase cancellation probability
    </p>
</div>
""", unsafe_allow_html=True)

# Optional: Add debug info by adding ?debug=true to your URL
if "debug" in st.query_params and st.query_params["debug"] == "true":
    with st.expander("🐛 Debug Information"):
        st.write("API URL:", API_URL)
        st.write("Health Data:", health_data)