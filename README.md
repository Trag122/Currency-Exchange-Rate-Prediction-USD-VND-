# Currency-Exchange-Rate-Prediction-USD-VND-

## Project Overview

This project focuses on forecasting the USD/VND exchange rate using historical data and machine learning techniques. Accurate predictions in the foreign exchange market allow businesses and investors to optimize cross-border pricing strategies, manage currency risks, and mitigate losses from market volatility.

## Dataset & Preprocessing

* **Target Variable:** The closing exchange rate (`Close`) of USD to VND.
* **Features:** Engineered features include lagging indicators (`Lag_1`, `Lag_2`), moving averages (`MA_5`, `MA_20`), and standard deviations (`STD_5`, `STD_20`) to capture trends and volatility.
* **Data Processing:**
* The date column was converted to datetime format and sorted chronologically.
* All input features were normalized using `MinMaxScaler` to scale values between 0 and 1, ensuring stable and faster convergence during training.



## Machine Learning Models & Fine-Tuning

Three primary regression models were trained and fine-tuned to capture the non-linear dynamics of the exchange rate:

### 1. Support Vector Regression (SVR)

* **Approach:** Utilized the Radial Basis Function (RBF) kernel to map data into higher-dimensional spaces for complex pattern recognition.
* **Tuning:** Applied `GridSearchCV` with 5-fold cross-validation to test 48 combinations of hyperparameters (`C`, `gamma`, and `epsilon`).

### 2. Artificial Neural Networks (ANN)

* **Architecture:** A multi-layer perceptron configured with hidden layers of 128, 64, and 32 units respectively.
* **Key Parameters:** Employed `ReLU` activation for non-linearity, a `Dropout` rate of 0.2 to prevent overfitting, and the Adam optimizer (learning rate = 0.01).
* **Training:** Implemented Early Stopping to halt training when validation loss hit its minimum (around epoch 3), preventing the model from memorizing noise.

### 3. Long Short-Term Memory (LSTM)

* **Approach:** Leveraged LSTM networks, which are explicitly designed for sequential time-series forecasting.
* **Tuning:** A Grid Search identified the optimal parameters: 32 LSTM units, learning rate of 0.005, batch size of 32, `tanh` activation, and 1 hidden layer.

## Comparison and Results

The models were evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the Coefficient of Determination ($R^{2}$).

* **SVR:** Achieved an $R^{2} \approx 0.79$ and an MAE of 54.08.
* **ANN:** Reached an out-of-sample $R^{2}$ of 0.803.
* **LSTM (Best Model):** Delivered the highest accuracy with $MSE=4233.26$, $RMSE=65.06$, $MAE=48.34$, and $R^{2}=0.8416$.

**Conclusion:** The LSTM model is the superior choice for this task. It effectively captured the complex fluctuations of the USD/VND exchange rate, providing the highest variance explanation ($R^{2}$) and the lowest prediction errors.

## Project Authors

* **Group's members**: Nguyen Pham Quynh Trang (Leader), Pham Hieu Ngan, Nguyen Thi Hong Nhung.
* **Course:** Machine Learning - VNU International School.
