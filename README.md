# Content-Monetization-Modeler
A Linear Regression model that can accurately estimate YouTube ad revenue for individual videos based on various performance and contextual features, and implement the results in a simple Streamlit web application
# 📊 YouTube Ad Revenue Prediction (Content Monetization Modeler)

This project builds a **machine learning pipeline** to predict **YouTube ad revenue (USD)** from video/channel features such as views, likes, comments, category, country, etc.  
It includes **EDA**, **preprocessing**, **model training**, and a **Streamlit app** for interactive predictions.

---

## 🚀 Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Run `eda/run_eda.py`
   - Generates dataset summary, correlation heatmaps, distributions, scatter plots.
   - Saves plots in `eda_outputs/`.

2. **Preprocessing**
   - Cleans dataset (duplicates, missing values).
   - Adds engineered features (`engagement`, `upload_month`).
   - Builds a Scikit-learn `ColumnTransformer` pipeline.
   - Saves cleaned dataset as `data/preprocessed_data.csv`.

   ```bash
   python run_preprocess.py
Model Training

Trains multiple models: Linear Regression, Ridge, Lasso, RandomForest, GradientBoosting.

Evaluates using R², RMSE, MAE.

Saves the best pipeline as models/best_model.joblib.

Exports metrics to models/metrics.csv.

bash
Copy code
python train_model.py
Streamlit App

Loads saved model and preprocessed dataset.

Provides a sidebar form to enter feature values.

Predicts ad revenue in USD.

bash
Copy code
streamlit run app.py
📂 Project Structure
nginx
Copy code
Youtube data/
├─ data/
│  ├─ youtube_ad_revenue_dataset.csv    # raw dataset
│  └─ preprocessed_data.csv             # cleaned dataset (generated)
├─ eda/
│  └─ run_eda.py                        # EDA script
├─ models/
│  ├─ best_model.joblib                 # trained pipeline (saved)
│  └─ metrics.csv                       # evaluation metrics
├─ src/
│  ├─ __init__.py
│  ├─ config.py                         # paths, constants
│  ├─ data_load.py                      # CSV loader
│  ├─ preprocess.py                     # preprocessing functions
│  └─ train.py                          # training logic
├─ run_preprocess.py                    # driver script for preprocessing
├─ train_model.py                       # driver script for training
├─ app.py                               # Streamlit app
└─ requirements.txt                     # dependencies
🛠️ Setup Instructions
1. Clone the repo / copy project
bash
Copy code
git clone <your-repo-url>
cd Youtube\ data
2. Create & activate virtual environment
bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Mac/Linux
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Place raw dataset
Ensure your dataset is at:

bash
Copy code
data/youtube_ad_revenue_dataset.csv
📊 Usage
Run EDA
bash
Copy code
python eda/run_eda.py
Run Preprocessing
bash
Copy code
python run_preprocess.py
Train Models
bash
Copy code
python train_model.py
Launch Streamlit App
bash
Copy code
streamlit run app.py
📈 Example Output
EDA → correlation heatmap, distributions, scatter plots

Preprocessing → preprocessed_data.csv with cleaned & engineered features

Model Training → metrics table like:

Model	R²	RMSE	MAE
LinearRegression	0.71	123.4	98.7
RandomForest	0.92	55.3	41.8

Streamlit App → prediction form + estimated ad revenue (USD)

📦 Requirements
See requirements.txt. Key libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

streamlit

joblib

Install all with:

bash
Copy code
pip install -r requirements.txt
🧑‍💻 Author
Developed by Priyadharshini M
Project: Content Monetization Modeler – YouTube Ad Revenue Prediction
