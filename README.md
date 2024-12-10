
# Predicting_Bank_Loan_Default_Using_Machine_Learning


## Overview
This project focuses on predicting whether a bank loan will default using a dataset containing 67,463 records and 35 features. The goal was to build a machine learning model that can predict loan defaults effectively by handling imbalanced data and optimizing the model’s performance.
## Key Tasks
- Exploratory Data Analysis (EDA): Performed an in-depth analysis to understand the dataset, identify patterns, and uncover any issues in the data.
- Data Preprocessing: Applied One-Hot Encoding to categorical variables and handled imbalanced data using the SMOTE technique.
- Feature Selection: Reduced the feature set from 35 to the top 20 features using the Embedded Method for feature importance.
- Modeling: Built a RandomForestClassifier model and tuned hyperparameters using GridSearchCV to optimize the model’s performance. Key hyperparameters tuned include: 
   - max_depth=25 
   - n_estimators=500
- Evaluation: Evaluated the model on imbalanced and balanced data to compare performance.
## Dependencies
To run this project, you’ll need the following Python packages:
- pandas
- NumPy
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

## Results

- Imbalanced Data Performance: The initial model trained on imbalanced data achieved 90% accuracy but had a precision of 0 and 0 recall, indicating poor performance in the minority class.
- Balanced Data Performance: After applying the SMOTE technique to balance the dataset, the model improved significantly, achieving 83% accuracy, a precision of 0.74 and recall score of 0.83, demonstrating much better performance predicting loan defaults.
## Conclusion
This project demonstrates the effectiveness of balancing imbalanced datasets using SMOTE and optimizing machine learning models through hyperparameter tuning. After these improvements, the RandomForestClassifier model showed significant gains in recall and overall predictive performance for the minority class.
## Future Improvements

- Explore additional feature engineering techniques.
- Test other classification models (e.g., XGBoost, GradientBoosting).
- Investigate further methods to handle class imbalance.