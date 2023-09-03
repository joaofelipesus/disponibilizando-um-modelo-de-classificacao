# Descrição

Este repositório contém o código utilizado em um artigo que mostra o passo a passo para criação e deploy de uma API que serve como interface para um modelo de classificação.

### Classificação

Foi utilizado um modelo de classificação `Multilayer perceptron (MLP)` treinado para classificar exemplares da base `IRIS`. 

Model: "sequential"

| Layer (type)    | Output Shape | Param |
|-----------------|--------------|-------|
| dense (Dense)   | (None, 128)  | 640   |                                                        
| dense_1 (Dense) | (None, 64)   | 8256  |    
| dense_2 (Dense) | (None, 3)    | 195   |

---

### Flask API

Foi desenvolvida uma API em `Flask` para disponibilizar o modelo de classificação treinado. A API contém apenas a rota `/predict`, que baseado nos valores recebidos para cada um dos atributos retorna a probabilidade do indivíduo pertencer a cada uma das classes da base `IRIS`.

```cURL
curl --request POST \
  --url localhost:5000/predict \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: Insomnia/2023.5.7' \
  --data '{
        "features": [5.1, 3.5, 1.4, 0.2]
}'

```

#### Comandos 
  - iniciar `virtualenv`:
    - `pip install virtualenv`;
    - `virtualenv .`.
    - `source bin/activate`
  - instalar dependências: `pip install -r requirements.txt`
  - iniciar a aplicação: `python main.py`
