from flask import Flask, jsonify, request
from keras.models import load_model

model = load_model('./model.h5')

print(model.summary())

# Define o objeto que faz referência a aplicação.
app = Flask(__name__)

# Declara uma rota do tipo POST.
@app.route('/predict', methods=['POST'])
def predict():
  payload = request.get_json()
  features = payload['features']
  result = model.predict([features])

  # attributes = payload['']

  # return jsonify(prediction_result=result[0])

  return jsonify(predictions=result)

# Inicia a aplicação.
app.run(debug=True)
