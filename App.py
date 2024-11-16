import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class SalesForecastingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Sales Forecasting App")
        self.setGeometry(300, 300, 400, 300)
        
        # Initialize layout
        layout = QVBoxLayout()
        
        # Upload button
        self.upload_button = QPushButton("Upload Sales Data (.csv)")
        self.upload_button.clicked.connect(self.upload_data)
        layout.addWidget(self.upload_button)
        
        # Train Model button
        self.train_button = QPushButton("Train Random Forest Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)
        
        # Predict button
        self.predict_button = QPushButton("Make Prediction")
        self.predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_button)
        
        # Display Result button
        self.display_button = QPushButton("Show Results")
        self.display_button.clicked.connect(self.show_results)
        layout.addWidget(self.display_button)
        
        # Label for results
        self.result_label = QLabel("Results will be displayed here")
        layout.addWidget(self.result_label)
        
        # Set up central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Variables to hold data
        self.data = None
        self.model = None
        self.predictions = None
        self.actual_sales = None
        self.test_data = None

    def upload_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Sales Data CSV", "", "CSV Files (*.csv)", options=options)
        if file_path:
            self.data = pd.read_csv(file_path, parse_dates=['date'])
            self.data.set_index('date', inplace=True)
            self.result_label.setText("Data Loaded Successfully")

    def train_model(self):
        if self.data is None:
            self.result_label.setText("Please upload data first.")
            return
        
        # Feature Engineering
        self.data['sales_diff'] = self.data['sales'].diff()
        self.data.dropna(inplace=True)
        
        def create_supervised(data, lag=1):
            df = pd.DataFrame(data)
            columns = [df.shift(i) for i in range(1, lag + 1)]
            columns.append(df)
            df = pd.concat(columns, axis=1)
            df.fillna(0, inplace=True)
            return df

        supervised_data = create_supervised(self.data['sales_diff'], 12)
        
        # Splitting Data
        train_data = supervised_data[:-12]
        test_data = supervised_data[-12:]
        
        # Scaling
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        
        x_train, y_train = train_data[:, 1:], train_data[:, 0]
        x_test, y_test = test_data[:, 1:], test_data[:, 0]

        # Train Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=0)
        self.model.fit(x_train, y_train)
        self.test_data = (x_test, y_test, scaler)
        self.result_label.setText("Model Trained Successfully")

    def make_prediction(self):
        if self.model is None or self.test_data is None:
            self.result_label.setText("Please train the model first.")
            return
        
        x_test, y_test, scaler = self.test_data
        
        # Predictions
        rf_predict = self.model.predict(x_test)
        rf_predict = scaler.inverse_transform(np.concatenate((rf_predict.reshape(-1, 1), x_test), axis=1))[:, 0]
        
        # Actual sales for the last 13 months
        self.actual_sales = self.data['sales'].values[-13:]
        
        # Calculating predicted sales
        result_list = []
        for index in range(len(rf_predict)):
            result_list.append(rf_predict[index] + self.actual_sales[index])
        
        self.predictions = np.array(result_list)
        self.result_label.setText("Predictions made successfully.")

    def show_results(self):
        if self.predictions is None or self.actual_sales is None:
            self.result_label.setText("Please make predictions first.")
            return
        
        # Metrics
        rf_mse = np.sqrt(mean_squared_error(self.actual_sales[-12:], self.predictions))
        rf_mae = mean_absolute_error(self.actual_sales[-12:], self.predictions)
        rf_r2 = r2_score(self.actual_sales[-12:], self.predictions)

        # Display metrics
        result_text = (f"Random Forest MSE: {rf_mse:.2f}\n"
                       f"Random Forest MAE: {rf_mae:.2f}\n"
                       f"Random Forest R2 Score: {rf_r2:.2f}")
        self.result_label.setText(result_text)

        # Plotting actual vs predicted
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index[-13:], self.actual_sales, label='Actual Sales')
        plt.plot(self.data.index[-12:], self.predictions, label='Predicted Sales', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Actual vs Predicted Sales')
        plt.legend()
        plt.show()

# Run the application
app = QtWidgets.QApplication(sys.argv)
window = SalesForecastingApp()
window.show()
sys.exit(app.exec_())
