# Vehicle Insurance Claims Prediction Model

Insurance fraud is a significant problem that impacts the profitability of insurance companies and increases the cost of insurance for honest policyholders.

Everyone needs vehicle insurance and fraudulent claims can take many forms, they can be staged accidents, false damage claims, and inflated injury claims.

By developing an accurate fraud detection model, insurance companies can prevent losses due to fraudulent claims and improve their overall business performance.



### Data Description ###



The dataset chosen is the " Vehicle Insurance Fraud Detection" dataset found on Kaggle, a CSV files that contains 33 columns. It consists of 15,420 records of vehicle insurance claims, out of which 923 are labeled as fraudulent. It captures different aspects of the vehicle insurance claims, including the driver's age, the vehicle's make and price, the accident area, and the policy type and whether a claim was fraudulent or not.

Since only 923 of the 15,420 claims are fraudulent, the data is skewed towards non fraudulent claims, which is going to make it difficult for the model to detect the fraud cases accurately. To overcome this challenge, we can use techniques such as oversampling the minority class or using cost-sensitive learning algorithms.

The dataset also contains redundant features that can affect the model's performance. Feature selection and dimensionality reduction techniques are needed to find the relevant features for the model.

https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection



### Tools ###

- **Pandas** for accessing the data, and preparing the it for modeling.

- **Ensemble model (Random Forest & Gradient Boosting)** for training the model

- **Streamlit** for creating a dashboard
