# Regression ANN for Salary Prediction

This project demonstrates how to build an Artificial Neural Network (ANN) for regression using TensorFlow/Keras to predict `EstimatedSalary` from customer data.

## Features

- **Data Preprocessing**
  - Drops irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
  - Encodes categorical features (`Gender` with LabelEncoder, `Geography` with OneHotEncoder)
  - Scales features using `StandardScaler`
- **Model Architecture**
  - Sequential ANN with two hidden layers (`relu` activation)
  - Output layer for regression (linear activation)
- **Training**
  - Early stopping to prevent overfitting
  - TensorBoard logging for visualization
- **Evaluation**
  - Reports Mean Absolute Error (MAE) on test data
- **Persistence**
  - Saves trained model (`.h5`), encoders, and scaler (`.pkl`)

## Files

- `regression.ipynb`: Main notebook with all code and explanations
- `Churn_Modelling.csv`: Dataset
- `requirements.txt`: List of required Python packages

## Usage

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the notebook:**  
   Open `regression.ipynb` in Jupyter or VS Code and execute cells step by step.

3. **View training logs:**  
   Use TensorBoard:
    ```bash
    tensorboard --logdir logs/fit
    ```

## Output

- Trained ANN model (`model.h5`, `regression_model.h5`)
- Saved encoders and scaler (`encoder_gender.pkl`, `onehot_encoder_geography.pkl`, `scaler.pkl`)
- TensorBoard logs in `logs/fit/`

## Notes

- The target variable is `EstimatedSalary`.
- All preprocessing steps are included in the notebook.

---

**Author:** Lakshay Sindhu  