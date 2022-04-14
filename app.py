import numpy as np
from flask import Flask, request, render_template
import pickle

app= Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    if(prediction[0]>0.5):
        prediction_t="Fake"
    else:
        prediction_t="Real"
    return render_template('index.html', prediction_text='The Bank note is {}'.format(prediction_t))


if __name__ == '__main__':
    app.run(debug=True)