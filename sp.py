from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn import metrics

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    df = yf.download(symbol, start=start_date, end=end_date)
    df = df.dropna()

    df['Date'] = pd.to_datetime(df.index)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df = df.drop("Date", axis=1)

    X = df[["Year", "Month", "Day"]]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    yyyy = request.form['year']
    mm = request.form['month']
    dd = request.form['day']

    date_to_predict = pd.DataFrame([[int(yyyy), int(mm), int(dd)]], columns=["Year", "Month", "Day"])
    predicted_price = model.predict(date_to_predict)
    predicted_price = np.round(predicted_price, 2)

    model_info = 'model_info' in request.form
    if model_info:
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        result = f' Name : Random Forest, Model Score : {score*100:.2f}%'
    else:
        result = None


    show_error = 'show_error' in request.form
    if show_error:
        mse = mean_squared_error(y_test, model.predict(X_test))
        mae = metrics.mean_absolute_error(y_test, model.predict(X_test))
        rmse = np.sqrt(mse)
        error_message = f'Mean Absolute Error (MAE) is {mae:.2f}$, Mean Squared Error (MSE) is {mse:.2f}$, Root Mean Squared Error (RMSE) is {rmse:.2f}$'
    else:
        error_message = None



    data_view = 'data_view' in request.form
    if data_view:
        return render_template('stock_info.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    else:
        None


    return render_template('result.html', symbol=symbol, date=f'{yyyy}-{mm}-{dd}', predicted_price=predicted_price[0],error_message=error_message, model_message=result)

if __name__ == '__main__':
    app.run(debug=True)
