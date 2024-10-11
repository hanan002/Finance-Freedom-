from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# โหลดโมเดล
model = joblib.load('model/best_model2.pkl')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # ดึงข้อมูลจากฟอร์ม
    Open = float(data['Open'])
    High = float(data['High'])
    Low = float(data['Low'])
    Vol = float(data['Vol'])
    Change = float(data['Change'])
    Year = int(data['Year'])
    Month = int(data['Month'])
    Day = int(data['Day'])
    Season_Autumn = int(data['Season_Autumn'])
    Season_Spring = int(data['Season_Spring'])
    Season_Summer = int(data['Season_Summer'])
    Season_Winter = int(data['Season_Winter'])

    # เตรียมข้อมูลสำหรับการทำนาย
    input_data = np.array([[Open, High, Low, Vol, Change, Year, Month, Day, Season_Autumn, Season_Spring, Season_Summer, Season_Winter]])

    # ทำนายผล
    predicted_price = model.predict(input_data)[0]

    return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
