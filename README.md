# 🌾 AI-ML Based Agri-Horticultural Commodity Price Forecasting

This project is a deep learning-powered web application that forecasts the next 3-day prices of selected agricultural commodities using historical pricing data. It uses LSTM models for time series prediction and integrates LLMs (Large Language Models) to generate human-like insights from the predicted results. The interface is multilingual, making it farmer-friendly across diverse regions in India.

## 🚀 Features
- Upload CSVs containing commodity price data  
- Preprocess data: filter by commodity & state, handle missing values  
- Normalize and generate time series sequences  
- LSTM-based model to forecast next 3-day prices  
- Natural language inference using LLMs  
- Multilingual UI (English, Hindi, Tamil, Malayalam)  
- Visual insights: actual vs predicted prices + forecast tables  

## 📊 Sample CSV Format
Date, Commodity, State, Modal Price (Rs./Quintal)
2024-01-01, Tomato, Tamil Nadu, 1200
2024-01-02, Tomato, Tamil Nadu, 1250

## 🛠️ Tech Stack
- Python, Streamlit  
- TensorFlow/Keras  
- Pandas, NumPy, Scikit-learn  
- Google Translate API  
- LLM-based insight generation (e.g., GPT-powered natural language)  

## 💻 How to Run
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
Run the app:
streamlit run app.py
Upload your CSV, select a commodity and state, and view interactive predictions!

## 📈 Output

📉 Line chart: Actual vs Predicted Prices

📅 Forecast Table: 3-day predicted prices in Rs./Quintal and Rs./Kg

🧠 Insight Text: Auto-generated interpretation of future price trends using LLMs

## 👥 Team
Built during a Hackathon to address real-world challenges in agricultural pricing using AI/ML. This tool aims to empower farmers and traders with short-term pricing insights, supporting informed decisions in their agricultural journey.

![image](https://github.com/user-attachments/assets/78ac7489-d771-4a5e-b6c4-61724c3803ff)
![image](https://github.com/user-attachments/assets/e54dbd93-8773-4c18-a4b2-f33138012d9a)



