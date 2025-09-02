# Weather Predictor 🌦️

A Python machine learning project that predicts whether it will rain tomorrow.  
Trains a logistic regression model on historical Kaggle weather data, then fetches live weather conditions from the OpenWeatherMap API to make real-time predictions.

---

## 🚀 Features
- Full ML pipeline: cleaning, imputing missing values, scaling, and train/test split
- Logistic regression baseline reaching **~85% accuracy** and **ROC-AUC ≈ 0.90**
- Visual outputs: confusion matrix and ROC curve plots
- Integration with **OpenWeatherMap API** for live predictions by city
- Models persisted with `joblib` for quick reloads and reproducibility

---

## ⚙️ Setup
Clone the repo:
```bash
git clone https://github.com/<your-username>/weather-predictor.git
cd weather-predictor
