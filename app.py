import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
tcp = pickle.load(open('tcp.pkl', 'rb'))
udp = pickle.load(open('udp.pkl', 'rb'))
icmp = pickle.load(open('icmp.pkl', 'rb'))
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Server got hit")
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    tcp_pred = tcp.predict(final_features)[0]
    udp_pred = udp.predict(final_features)[0]
    icmp_pred = icmp.predict(final_features)[0]

    prediction = (tcp_pred or udp_pred or icmp_pred)
    print("Server got request.")
    output = "Attack possible" if prediction == 1 else "No attacks"

    return render_template('home.html', prediction_text='Prediction: {}'.format(output))
                          # tcp_text=tcp_pred, udp_text=udp_pred, icmp_text=icmp_pred)

if __name__ == "__main__":
    app.run(debug=True)
