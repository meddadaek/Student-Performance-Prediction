# 📊 Student Performance Prediction

This project is a **machine learning pipeline** for predicting student performance based on various features such as gender, parent income, and other academic data.  
It uses both **Linear Regression** and **Random Forest Regressor** to compare models and visualize feature importance.

## 🧰 Tech Stack
- pandas — Data loading & cleaning  
- numpy — Numerical operations  
- matplotlib & seaborn — Data visualization  
- scikit-learn — Machine learning models & metrics


## 🧠 Features
- ✅ Cleans and preprocesses data (handles duplicate columns, encodes gender, removes unnecessary columns)  
- 🧮 Adds simulated parent income feature for richer analysis  
- 🔥 Automatically selects the target column (first “Obtain marks” column)  
- 🤖 Trains Linear Regression & Random Forest Regressor models  
- 📊 Compares model performance using R² Score and MSE  
- 🌡️ Visualizes feature importance and correlation heatmap

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/meddadaek/student-performance-prediction.git
cd student-performance-prediction
2. Install Dependencies
bash
Copier le code
pip install -r requirements.txt
3. Add the Dataset
Place your file Student data 4.csv in the project folder.

4. Run the Project
bash
Copier le code
python main.py
📊 Model Results
Linear Regression

MSE: Displayed in terminal

R² Score: Displayed in terminal

Random Forest Regressor

MSE: Displayed in terminal

R² Score: Displayed in terminal

📈 Bar charts and heatmaps are displayed at the end of execution to visualize model comparison and feature correlations.

🧪 Customization
Add or remove columns in preprocessing

Replace the dataset with your own

Tune hyperparameters of the Random Forest model

Add more ML models to the models dictionary for comparison

🏆 Future Improvements
Add cross-validation

Implement hyperparameter tuning

Save the best model

Build a small UI or API for predictions
