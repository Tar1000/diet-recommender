import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import base64
import io
from sklearn.neighbors import NearestNeighbors
from utils.preprocessing import classify_sugar_level, classify_bmi, encode_dataset, encode_user_input

# Setup
st.set_page_config(page_title="Diabetic Diet Recommender", layout="centered")
st.title("🍎 Diet Recommendation System for Diabetics")

#add app icon
with st.sidebar:
    st.image("assets/icon.png", width=120)
    st.markdown("### **GlucoMeal Advisor**")
    st.markdown("##### by TTSET GLOBAL")


# Form
st.header("🧾 Enter Your Health Information")
with st.form("user_input_form"):
    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=400, step=1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1)
    diet_type = st.selectbox("Diet Preference", ["Veg", "Non-Veg"])
    calorie_needs = st.slider("Daily Calorie Needs", min_value=1000, max_value=3000, step=100)
    use_ml = st.checkbox("🤖 Use ML-Based Suggestion Instead of Rule-Based", value=False)
    submit = st.form_submit_button("🔍 Get Recommendations")

@st.cache_data
def load_food_data():
    return pd.read_csv("food_dataset.csv")

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Diet Recommendation Report", ln=True, align="C")
    pdf.ln(10)

    for index, row in df.iterrows():
        pdf.cell(0, 10, f"{row['Food']} - {row['Calories']} cal", ln=True)

    # ✅ Output PDF as a byte string
    pdf_output = pdf.output(dest='S').encode('latin-1')

    # ✅ Convert to base64 for download link
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="recommendations.pdf">📥 Download PDF</a>'
    return href


if submit:
    sugar_level_category = classify_sugar_level(fbs)
    bmi_category = classify_bmi(bmi)
    st.write("### 🧠 Profile Analysis")
    st.write(f"- **Sugar Level Category:** {sugar_level_category}")
    st.write(f"- **BMI Category:** {bmi_category}")

    food_data = load_food_data()
    encoded_data = encode_dataset(food_data)
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(encoded_data[['Sugar_Level_Code', 'BMI_Code', 'Diet_Code']])

    if use_ml:
        st.subheader("🤖 ML-Based Recommendations")
        user_vector = encode_user_input(sugar_level_category, bmi_category, diet_type)
        distances, indices = knn.kneighbors([user_vector])
        recommended_foods = encoded_data.iloc[indices[0]]
        st.dataframe(recommended_foods[['Food', 'Calories']].reset_index(drop=True))
    else:
        st.subheader("📋 Rule-Based Recommendations")
        matching_foods = food_data[
            (food_data['BMI_Category'] == bmi_category) &
            (food_data['Sugar_Level_Category'] == sugar_level_category) &
            (food_data['Diet_Type'] == diet_type)
        ]
        if not matching_foods.empty:
            st.dataframe(matching_foods[['Food', 'Calories']].reset_index(drop=True))
        else:
            st.warning("❌ No exact matches found. Try ML mode or adjust input.")

    export_df = recommended_foods if use_ml else matching_foods
    st.subheader("📤 Export Recommendations")
    csv = convert_df_to_csv(export_df)
    st.download_button("📥 Download as CSV", data=csv, file_name='recommendations.csv', mime='text/csv')
    st.markdown(generate_pdf(export_df), unsafe_allow_html=True)

    # Charts
    if not export_df.empty:
        st.subheader("🥧 Calorie Distribution")
        fig_pie = px.pie(export_df, names='Food', values='Calories', title='Calories by Food Item')
        st.plotly_chart(fig_pie)
