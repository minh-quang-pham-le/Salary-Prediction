# Salary Prediction using Regression

This project aims to predict salary based on only one simple feature "years of experience" using regression models and compare it with a neural network model.
The dataset is downloaded from "https://www.kaggle.com/rsadiq/salary" with 35 examples (1 feature and 1 target) in total.
We usually split the data into 3 subsets: Training, Cross-validation, and test set. Due to the small size of the data set, I only divided the data into 2 subsets: 80% training set and 20% test set. And compare between models to see which model is the best.
## Project Structure

- `data`: Contains the dataset (`salary_data.csv`).
- `notebooks/`: Jupyter notebooks, scaler_X, poly, and scaler_poly : scaler_X and scaler_poly store the scaler for linear model and polynomial model, poly store the PolynomialFeature.
- `scripts/`: Python scripts for data loading, model training, and comparison.
- `README.md`: Project overview and usage guide.
- `requirements.txt`: Required Python libraries.

## How to Set Up and Run the Project
1. Clone the repository:
   `git clone https://github.com/yourusername/salary-prediction-regression.git`
   `cd salary-prediction-regression`
   
2. Create and activate a virtual environment
  `python -m venv venv`
  `source venv/bin/activate`    # On macOS/Linux
  `venv\Scripts\activate`       # On Windows
 
3. Install the required libraries:
   `pip install -r requirements.txt`

4. All commands are displayed in the notebook in the Scripts file:
- Data loading and preprocessing: `Scripts/data_loading.py`
- Model training: `Scripts/model_training.py`
- Model comparison: `Scripts/model_comparison.py`
- Predict: `Sciprts/predict.py`

## Results
- Linear Regression, Polynomial Regression and Neural Network models have been trained and compared for the small data set with only one single feature.
