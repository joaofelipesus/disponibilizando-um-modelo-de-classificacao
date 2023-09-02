from sklearn.datasets import load_iris
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense, InputLayer
from keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split

# Reduz aleatoriedade
SEED = 42

# Carrega a base de dados
iris = load_iris()
X = iris.data
y = iris.target

# Converte y em valores categóricos
y = to_categorical(y)

# divide a base em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Define as camadas do modelo de classificação
model = Sequential(
    [
        InputLayer(input_shape=(4,)),
        Dense(128, activation='relu', kernel_initializer=RandomNormal(seed=SEED)),
        Dense(64, activation='relu', kernel_initializer=RandomNormal(seed=SEED)),
        Dense(units=3, activation='softmax')
    ]
)

# Compila o modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['categorical_accuracy']
)

# Treino o modelo
model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2
)

# Avalia grupo de teste, o primeiro valor retornado é a perda e o segundo a acurácia.
print('Evaluate: ', model.evaluate(X_test, y_test))

model.save('model.h5')
