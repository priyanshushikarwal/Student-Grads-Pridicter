# Student-Grads-Pridicter


---

# Student Performance Prediction

## Overview
This project aims to predict student performance (final grades) based on various factors such as study time, attendance, family background, and other demographic and academic features. The project uses regression models to predict the final grade (`G3`) of students.

## Dataset
The dataset used in this project is the **Student Performance Dataset** from the UCI Machine Learning Repository. It contains information about students' academic performance in two subjects: Mathematics (`student-mat.csv`) and Portuguese (`student-por.csv`). This project uses the Mathematics dataset.

### Dataset Features
- **Demographic Features**: `school`, `sex`, `age`, `address`, `famsize`, etc.
- **Academic Features**: `Medu` (mother's education), `Fedu` (father's education), `studytime`, `failures`, etc.
- **Target Variable**: `G3` (final grade, ranging from 0 to 20).

## Project Steps
1. **Data Exploration**:
   - Load and inspect the dataset.
   - Check for missing values and data types.
   - Visualize the distribution of the target variable (`G3`).

2. **Data Preprocessing**:
   - Encode categorical variables using one-hot encoding.
   - Scale numerical features using `StandardScaler`.
   - Split the dataset into training and testing sets.

3. **Model Building**:
   - Train a **Linear Regression** model.
   - Train a **Random Forest Regressor** model.

4. **Model Evaluation**:
   - Evaluate models using **Mean Squared Error (MSE)** and **R-squared (R2)** metrics.
   - Compare the performance of the two models.

5. **Visualization**:
   - Plot the correlation heatmap to understand feature relationships.
   - Visualize actual vs. predicted grades.
   - Display feature importance for the Random Forest model.

## Results
- **Linear Regression**:
  - MSE: 
  - R2:
- **Random Forest Regressor**:
  - MSE:
  - R2: 

The Random Forest model outperformed the Linear Regression model in terms of both MSE and R2 scores.

## How to Run the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. **Install Dependencies**:
   Ensure you have the required Python libraries installed:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn
   ```

3. **Run the Code**:
   Execute the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook student_performance_prediction.ipynb
   ```
   or
   ```bash
   python student_performance_prediction.py
   ```

## Dependencies
- Python 3.x
- Libraries:
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`

## Files in the Repository
- `student_performance_prediction.ipynb`: Jupyter Notebook containing the code.
- `student-mat.csv`: Dataset file.
- `README.md`: Project documentation.

## Future Work
- Experiment with other regression models (e.g., Gradient Boosting, XGBoost).
- Perform hyperparameter tuning for better model performance.
- Explore feature engineering to improve predictions.

## Author
Priyanshu  
Priyanshushikarwal@gmail.com
https://github.com/priyanshushikarwal

---

