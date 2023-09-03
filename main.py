from flask import Flask, jsonify, request
from keras.models import load_model
import os

# Carrega o modelo de classificação antes de iniciar a API.
model = load_model('./model.h5')

# Define o objeto que faz referência a aplicação.
app = Flask(__name__)

# Declara uma rota do tipo POST.
@app.route('/predict', methods=['POST'])
def predict():
  payload = request.get_json()
  # Recupera o atributo com as features
  features = payload['features']

  # Classifica o exemplar
  result = model.predict([features])

  # É preciso converter o resultado do modelo em uma lista, pois o tipo do dado retornado pelo método
  # keras.Sequential.predict é um np.array e este objeto não pode ser serializado para JSON através
  # do método jsonify
  result_list = result.tolist()

  # Retorna uma lista com a probabilidade de cada uma das classes.
  return jsonify(predictions=result_list[0])

# Inicia a aplicação.
if __name__ == '__main__':
  app.run()
