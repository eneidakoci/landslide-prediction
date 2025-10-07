**Landslide Risk Prediction using Machine Learning**

This project aims to create a machine learning system for predicting landslide risk in Tirana, Albania.


**Dependencies needed:**

  pip install streamlit fastapi uvicorn requests folium streamlit-folium pandas numpy scikit-learn xgboost joblib matplotlib seaborn


**Start the backend:**

uvicorn api:app --reload --port 8502


**Start the frontend:** 

streamlit run app.py


**Project Structure:**

  ModelTraining.py - Trains Random Forest, SVM, and XGBoost models
  
  api.py - FastAPI backend for predictions
  
  app.py - Streamlit frontend interface
  
  xgboost_model.pkl - Trained model file (best performing)


**Features:**

  Predicts landslide risk based on terrain and environmental factors
  
  Interactive map visualization
  
  Risk levels: Very Low, Low, Moderate, High, Very High
