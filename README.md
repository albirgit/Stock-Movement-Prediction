# Financial News Stock Movement Prediction 📈

### 📝 Project Overview
This project focuses on predicting the movement of the stock market (Up/Down) by analyzing daily financial news headlines. It utilizes **Natural Language Processing (NLP)** and **Machine Learning** to identify correlations between news sentiment and market trends.

### 🚀 Key Features
- **NLP Pipeline:** Implementation of text preprocessing including tokenization, stop-word removal, and lowercasing using the **NLTK** library.
- **Vectorization:** Transformation of raw text into numerical data using **CountVectorizer** or **TF-IDF**.
- **Classification Model:** Training of a machine learning classifier (e.g., Random Forest/Logistic Regression) to predict target movements.
- **Interactive Web App:** A user-friendly dashboard built with **Streamlit** for real-time predictions based on manual headline input.

### 🛠️ Tech Stack
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Natural Language Processing (NLP):** NLTK
- **Deployment:** Streamlit

### 📊 Dataset
The project utilizes the **[Daily Financial News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews)** dataset from Kaggle, covering 8 years of headlines and corresponding DJIA movements.

### ⚙️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/stock-movement-prediction.git](https://github.com/your-username/stock-movement-prediction.git)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit app:**
    ```bash
   streamlit run app.py

### 📈 Methodology
1. **Exploratory Data Analysis (EDA):** Visualizing class distribution and headline patterns using Matplotlib/Seaborn.
2. **Preprocessing:** Cleaning headlines (removing punctuation, special characters, and stop-words).
3. **Feature Engineering:** Applying **N-grams** and **Vectorization** to prepare data for modeling.
4. **Modeling & Evaluation:** Splitting data into training/testing sets and evaluating performance using Accuracy and Confusion Matrix.

---
*Developed as part of my BS Data Science (Semester 3) projects.*
   
