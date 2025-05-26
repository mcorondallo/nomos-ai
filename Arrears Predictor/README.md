# Arrears Payment Predictor

A machine learning application to predict the likelihood of arrears payments based on historical data and various factors.

## Features

- Prediction of payment likelihood as a percentage
- Support for CSV file uploads for both historical and prediction data
- Manual data entry through an intuitive web interface
- Visualization of prediction results
- PDF report generation with benchmark comparisons
- Feature importance analysis

## Installation

1. Clone this repository or download the files to your local machine.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. The application will open in your default web browser.

3. **Train Model Tab**:
   - Upload a CSV file with historical payment data or use the provided dummy data
   - Click "Train Model" to build the prediction model
   - View feature importance to understand what factors affect payment likelihood

4. **Make Predictions Tab**:
   - Upload a CSV file with data for prediction, OR
   - Enter data manually using the form
   - Generate predictions

5. **Results & Reports Tab**:
   - View the prediction results
   - Analyze the distribution of payment probabilities
   - Compare with benchmark
   - Generate and download a PDF report

## Data Format

Your CSV files should contain the following columns:

- `days_overdue`: Number of days payment is overdue
- `previous_late_payments`: Count of previous late payments
- `average_days_late`: Average days late for previous payments
- `lease_count`: Number of active leases
- `lease_amount`: Monetary value of lease(s)
- `property_size_sqm`: Size of property in square meters
- `payment_method`: Method of payment (bank_transfer, card)
- `country`: Country location
- `region`: Region within country
- `was_paid` (for historical data only): Target variable (1 if paid, 0 if not)

## Machine Learning Model

The application uses a Gradient Boosting Regressor model to predict payment likelihood. The model:

- Handles both numerical and categorical features
- Preprocesses data with appropriate scaling and encoding
- Provides feature importance analysis
- Returns predictions as probability percentages

## Customization

- Adjust the risk category thresholds in the `model.py` file
- Modify the dummy data generation in `utils.py` to better match your actual data
- Customize the PDF report generation in `utils.py`
