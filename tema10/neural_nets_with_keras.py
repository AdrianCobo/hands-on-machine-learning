# code from: https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb
from pickletools import optimize
from statistics import mode
import sys

assert sys.version_info >= (3, 7)


from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")
import matplotlib.pyplot as plt

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

from pathlib import Path

IMAGES_PATH = Path() / "images" / "ann"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# The perceptron
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target == 0  # Iris setosa

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)  # predicts True and False for these 2 flowers
print(y_pred)

# The Perceptron is equivalent to a SGDClassifier with loss="perceptron", no regularization, and a constant learning rate equal to 1:
# extra code – shows how to build and train a Perceptron

# from sklearn.linear_model import SGDClassifier

# sgd_clf = SGDClassifier(loss="perceptron", penalty=None,
#                         learning_rate="constant", eta0=1, random_state=42)
# sgd_clf.fit(X, y)
# assert (sgd_clf.coef_ == per_clf.coef_).all()
# assert (sgd_clf.intercept_ == per_clf.intercept_).all()

# When the Perceptron finds a decision boundary that properly separates the classes, it stops learning. This means that the decision boundary is often quite close to one class:
# extra code – plots the decision boundary of a Perceptron on the iris dataset

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

a = -per_clf.coef_[0, 0] / per_clf.coef_[0, 1]
b = -per_clf.intercept_ / per_clf.coef_[0, 1]
axes = [0, 5, 0, 2]
x0, x1 = np.meshgrid(
    np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
    np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)
custom_cmap = ListedColormap(["#9898ff", "#fafab0"])

plt.figure(figsize=(7, 3))
plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris setosa")
plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris setosa")
plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="lower right")
plt.axis(axes)
plt.show()

# Regression MLPs
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(rmse)

# Classification MLPs
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)

pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_valid, y_valid)
print(accuracy)

# Keras
# Image classifier:
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
print("trainning set shape:", X_train.shape)
print("pixel intensity representation", X_train.dtype)
# scale the pixels intensity to a 0-1 range
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0

# extra code showing a train image

plt.imshow(X_train[0], cmap="binary")
plt.axis("off")
plt.show()

# Define the corresponding classes name:

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
print("class name first image:", class_names[y_train[0]])

# extra code – this cell generates and saves Figure 10–10

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.title(class_names[y_train[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)

save_fig("fashion_mnist_plot")
plt.show()

# creating the model
tf.random.set_seed(42)
model = (
    tf.keras.Sequential()
)  # Crea un modelo secuencial = Una sola pila de capas conectadas secuencialmente
model.add(
    tf.keras.layers.InputLayer(input_shape=[28, 28])
)  # Añadimos la capa de entrada e indicamos el tamaño de entrada
model.add(
    tf.keras.layers.Flatten()
)  # Agregamos una capa para convertir cada imagen en una entrada 1D
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(
    tf.keras.layers.Dense(10, activation="softmax")
)  # ultima capa con activacion softmax ya que hay 10 clases independientes

# extra code – clear the session to reset the name counters
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
print("model summary\n", model.summary())
# extra code – another way to display the model's architecture
tf.keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
print("capas del model", model.layers)

# compilando el model:
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
)

# Training and evaluating the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
print(history.params)
print(history.epoch)

import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(
    figsize=(8, 5),
    xlim=[0, 29],
    ylim=[0, 1],
    grid=True,
    xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"],
)
plt.legend(loc="lower left")  # extra code
save_fig("keras_learning_curves_plot")  # extra code
plt.show()

print(model.evaluate(X_test, y_test))

# haciendo predicciones con el modelo
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = y_proba.argmax(axis=-1)
print(y_pred)
print(np.array(class_names)[y_pred])

# extra code – this cell generates and saves Figure 10–12
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis("off")
    plt.title(class_names[y_test[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
save_fig("fashion_mnist_images_plot", tight_layout=False)
plt.show()

# Building a Regression MLP
# extra code – load and split the California housing dataset, like earlier
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential(
    [
        norm_layer,
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

# Buildin coplex models
# extra code – reset the name counters and make the code reproducible
tf.keras.backend.clear_session()
tf.random.set_seed(42)

normalization_layer = tf.keras.layers.Normalization()  # estanzariza las entradas
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)
# concatenamos las capas
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([input_, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)

# What if you want to send different subsets of input features through the wide or deep paths? We will send 5 features (features 0 to 4), and 6 through the
# deep path (features 2 to 7). Note that 3 features will go through both (features 2, 3 and 4).

tf.random.set_seed(42)  # extra code
# concatenamos las capas(indicamos como van a estar conectadas)
input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep),
    y_train,
    epochs=20,
    validation_data=((X_valid_wide, X_valid_deep), y_valid),
)
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))

# Adding an auxiliary output for regularization:

tf.keras.backend.clear_session()
tf.random.set_seed(42)
input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
aux_output = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    loss=("mse", "mse"),
    loss_weights=(0.9, 0.1),  # ponderamos las salidas
    optimizer=optimizer,
    metrics=["RootMeanSquaredError"],
)
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep),
    (
        y_train,
        y_train,
    ),  # como queremos que las 2 salidas traten de predecir las mismas etiquetas resultado, los dos argumentos son iguales
    epochs=20,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
)

# Uso de la api de creacion de subclases para crear modelos dinamicos
class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output


tf.random.set_seed(42)  # extra code – just for reproducibility
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    loss="mse",
    loss_weights=[0.9, 0.1],
    optimizer=optimizer,
    metrics=["RootMeanSquaredError"],
)
model.norm_layer_wide.adapt(X_train_wide)
model.norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep),
    (y_train, y_train),
    epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
)
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

# Guardando y restaurando un modelo
# extra code – delete the directory, in case it already exists

import shutil

shutil.rmtree("my_keras_model", ignore_errors=True)
model.save("my_keras_model", save_format="tf")
# extra code – show the contents of the my_keras_model/ directory
for path in sorted(Path("my_keras_model").glob("**/*")):
    print(path)

model = tf.keras.models.load_model("my_keras_model")
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
model.save_weights("my_weights")
model.load_weights("my_weights")
# extra code – show the list of my_weights.* files
for path in sorted(Path().glob("my_weights.*")):
    print(path)

# Uso de callbacks de llamadas
shutil.rmtree("my_checkpoints", ignore_errors=True)  # extra code
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "my_checkpoints", save_weights_only=True
)
history = model.fit(
    (X_train_wide, X_train_deep),
    (y_train, y_train),
    epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb],
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
history = model.fit(
    (X_train_wide, X_train_deep),
    (y_train, y_train),
    epochs=100,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb, early_stopping_cb],
)


class PrintValTrainRatioCallback(
    tf.keras.callbacks.Callback
):  # Creando nuestro propio callback
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")


val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(
    (X_train_wide, X_train_deep),
    (y_train, y_train),
    epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[val_train_ratio_cb],
    verbose=0,
)

# Usando tensorBoard para visualizacion
# pip install tensorboard-plugin-profile
# tensorboard --logdir=./my_logs
shutil.rmtree("my_logs", ignore_errors=True)
from pathlib import Path
from time import strftime


def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")


run_logdir = get_run_logdir()
# extra code – builds the first regression model we used earlier
tf.keras.backend.clear_session()
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential(
    [
        norm_layer,
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir, profile_batch=(100, 200))
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid),
    callbacks=[tensorboard_cb],
)

print("my_logs")
for path in sorted(Path("my_logs").glob("**/*")):
    print("  " * (len(path.parts) - 1) + path.parts[-1])

test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)

        data = (np.random.randn(100) + 2) * step / 100  # gets larger
        tf.summary.histogram("my_hist", data, buckets=50, step=step)

        images = np.random.rand(2, 32, 32, 3) * step / 1000  # gets brighter
        tf.summary.image("my_images", images, step=step)

        texts = ["The step is " + str(step), "Its square is " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)

        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

# Ajuste de hiperparametros de redes neuronaless
# pip install -q -U keras-tuner

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
tf.keras.backend.clear_session()
tf.random.set_seed(42)

import keras_tuner as kt

# pip install -q -U keras_tuner


def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


random_search_tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    overwrite=True,
    directory="my_fashion_mnist",
    project_name="my_rnd_search",
    seed=42,
)
random_search_tuner.search(
    X_train, y_train, epochs=10, validation_data=(X_valid, y_valid)
)

top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_params[0].values)  # best hyperparameter values

best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
print(best_trial.summary())
print(best_trial.metrics.get_last_value("val_accuracy"))
best_model.fit(X_train_full, y_train_full, epochs=10)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)


class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tf.keras.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, y, **kwargs)


hyperband_tuner = kt.Hyperband(
    MyClassificationHyperModel(),
    objective="val_accuracy",
    seed=42,
    max_epochs=10,
    factor=3,
    hyperband_iterations=2,
    overwrite=True,
    directory="my_fashion_mnist",
    project_name="hyperband",
)
root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping_cb, tensorboard_cb],
)

bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(),
    objective="val_accuracy",
    seed=42,
    max_trials=10,
    alpha=1e-4,
    beta=2.6,
    overwrite=True,
    directory="my_fashion_mnist",
    project_name="bayesian_opt",
)
bayesian_opt_tuner.search(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping_cb],
)

# Ejercicios:
# 1. TensorFlow Playground(https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28517&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
# es un práctico simulador de redes neuronales creado por el equipo de TensorFlow. En este ejercicio, entrenará varios clasificadores
# binarios con solo unos pocos clics y modificará la arquitectura del modelo y sus hiperparámetros para ganar cierta intuición sobre cómo funcionan las redes
# neuronales y qué hacen sus hiperparámetros. Tómese su tiempo para explorar lo siguiente :

# 1.1 Los patrones aprendidos por una red neuronal. Intente entrenar la red neuronal predeterminada haciendo clic en el botón Ejecutar (arriba a la izquierda).
# Observe cómo encuentra rápidamente una buena solución para la tarea de clasificación. Las neuronas de la primera capa oculta han aprendido patrones simples,
# mientras que las neuronas de la segunda capa oculta han aprendido a combinar los patrones simples de la primera capa oculta en patrones más complejos. En
# general, cuantas más capas haya, más complejos pueden ser los patrones.

# Funciones de activación. Intente reemplazar la función de activación de tanh con una función de activación de ReLU y vuelva a entrenar la red. Observe que
# encuentra una solución aún más rápido, pero esta vez los límites son lineales. Esto se debe a la forma de la función ReLU.

# El riesgo de mínimos locales. Modifique la arquitectura de la red para tener solo una capa oculta con tres neuronas. Entrénelo varias veces (para
# restablecer los pesos de la red, haga clic en el botón Restablecer junto al botón Reproducir). Tenga en cuenta que el tiempo de entrenamiento varía mucho y,
# a veces, incluso se atasca en un mínimo local.

# Qué sucede cuando las redes neuronales son demasiado pequeñas. Elimina una neurona para quedarte con solo dos. Tenga en cuenta que la red neuronal ahora es
# incapaz de encontrar una buena solución, incluso si lo intenta varias veces. El modelo tiene muy pocos parámetros y se ajusta sistemáticamente al conjunto
# de entrenamiento.

# ¿Qué sucede cuando las redes neuronales son lo suficientemente grandes? Establezca el número de neuronas en ocho y entrene la red varias veces. Tenga en
# cuenta que ahora es consistentemente rápido y nunca se atasca. Esto destaca un hallazgo importante en la teoría de redes neuronales: las redes neuronales
# grandes rara vez se atascan en los mínimos locales, e incluso cuando lo hacen, estos óptimos locales suelen ser casi tan buenos como el óptimo global. Sin
# embargo, aún pueden quedarse atascados en mesetas largas durante mucho tiempo.

# El riesgo de que desaparezcan los gradientes en las redes profundas. Seleccione el conjunto de datos en espiral (el conjunto de datos de la parte inferior
# derecha debajo de "DATOS") y cambie la arquitectura de la red para tener cuatro capas ocultas con ocho neuronas cada una. Tenga en cuenta que el
# entrenamiento lleva mucho más tiempo y, a menudo, se estanca durante largos períodos de tiempo. También observe que las neuronas en las capas más altas (a
# la derecha) tienden a evolucionar más rápido que las neuronas en las capas más bajas (a la izquierda). Este problema, llamado el problema de los
# "gradientes que se desvanecen", se puede aliviar con una mejor inicialización del peso y otras técnicas, mejores optimizadores (como AdaGrad o Adam) o
# Batch Normalization (discutido en el Capítulo 11 ).

# Ve más lejos. Tómese una hora más o menos para jugar con otros parámetros y tener una idea de lo que hacen, para construir una comprensión intuitiva sobre
# las redes neuronales.

# 2. Dibuje una ANN usando las neuronas artificiales originales (como las de la Figura 10-3 ) que calcula A ⊕ B (donde ⊕ representa la operación XOR). Pista: A ⊕ B = ( A ∧ ¬ B ) ∨ (¬ A ∧ B ).

# 3. ¿Por qué generalmente es preferible usar un clasificador de regresión logística en lugar de un perceptrón clásico (es decir, una sola capa de unidades
# lógicas de umbral entrenadas con el algoritmo de entrenamiento de perceptrón)? ¿Cómo puede modificar un perceptrón para que sea equivalente a un
# clasificador de regresión logística?
# A classical Perceptron will converge only if the dataset is linearly separable, and it won't be able to estimate class probabilities. In contrast, a
# Logistic Regression classifier will generally converge to a reasonably good solution even if the dataset is not linearly separable, and it will output
# class probabilities. If you change the Perceptron's activation function to the sigmoid activation function (or the softmax activation function if there are
# multiple neurons), and if you train it using Gradient Descent (or some other optimization algorithm minimizing the cost function, typically cross entropy),
# then it becomes equivalent to a Logistic Regression classifier.

# 4. ¿Por qué la función de activación sigmoidea fue un ingrediente clave en el entrenamiento de las primeras MLP?
# The sigmoid activation function was a key ingredient in training the first MLPs because its derivative is always nonzero, so Gradient Descent can always
# roll down the slope. When the activation function is a step function, Gradient Descent cannot move, as there is no slope at all

# 5. Nombre tres funciones de activación populares. ¿Puedes dibujarlos?
# Popular activation functions include the step function, the sigmoid function, the hyperbolic tangent (tanh) function, and the Rectified Linear Unit (ReLU)
# function (see Figure 10-8). See Chapter 11 for other examples, such as ELU and variants of the ReLU function

# 6. Suponga que tiene un MLP compuesto por una capa de entrada con 10 neuronas de paso, seguida de una capa oculta con 50 neuronas artificiales y, finalmente,
# una capa de salida con 3 neuronas artificiales. Todas las neuronas artificiales utilizan la función de activación ReLU.
# ¿Cuál es la forma de la matriz de entrada X ?
# The shape of the input matrix X is m × 10, where m represents the training batch size.
# ¿Cuáles son las formas de la matriz de peso de la capa oculta W h y el vector de polarización b h ?
# The shape of the hidden layer's weight matrix W_h_ is 10 × 50, and the length of its bias vector b_h_ is 50.
# ¿Cuáles son las formas de la matriz de ponderación W o y el vector de polarización b o de la capa de salida ?
# The shape of the output layer's weight matrix W_o_ is 50 × 3, and the length of its bias vector b_o_ is 3.
# ¿Cuál es la forma de la matriz de salida Y de la red ?
# The shape of the network's output matrix Y is m × 3.
# Escriba la ecuación que calcula la matriz de salida de la red Y como una función de X , W h , b h , W o y b o .
# Y = ReLU(ReLU(X W_h_ + b_h_) W_o_ + b_o_). Recall that the ReLU function just sets every negative number in the matrix to zero. Also note that when you are
# adding a bias vector to a matrix, it is added to every single row in the matrix, which is called broadcasting.

# 7. ¿Cuántas neuronas necesita en la capa de salida si desea clasificar el correo electrónico en spam o jamón? ¿Qué función de activación debería utilizar en
# la capa de salida? Si, en cambio, desea abordar MNIST, ¿cuántas neuronas necesita en la capa de salida y qué función de activación debe usar? ¿Qué hay de
# hacer que su red prediga los precios de la vivienda, como en el Capítulo 2 ?
# To classify email into spam or ham, you just need one neuron in the output layer of a neural network—for example, indicating the probability that the email
# is spam. You would typically use the sigmoid activation function in the output layer when estimating a probability. If instead you want to tackle MNIST,
# you need 10 neurons in the output layer, and you must replace the sigmoid function with the softmax activation function, which can handle multiple classes,
# outputting one probability per class. If you want your neural network to predict housing prices like in Chapter 2, then you need one output neuron, using
# no activation function at all in the output layer. Note: when the values to predict can vary by many orders of magnitude, you may want to predict the
# logarithm of the target value rather than the target value directly. Simply computing the exponential of the neural network's output will give you the
# estimated value (since exp(log v) = v).

# 8. ¿Qué es la retropropagación y cómo funciona? ¿Cuál es la diferencia entre retropropagación y autodiferenciación en modo inverso?
# Backpropagation is a technique used to train artificial neural networks. It first computes the gradients of the cost function with regard to every model
# parameter (all the weights and biases), then it performs a Gradient Descent step using these gradients. This backpropagation step is typically performed
# thousands or millions of times, using many training batches, until the model parameters converge to values that (hopefully) minimize the cost function. To
# compute the gradients, backpropagation uses reverse-mode autodiff (although it wasn't called that when backpropagation was invented, and it has been
# reinvented several times). Reverse-mode autodiff performs a forward pass through a computation graph, computing every node's value for the current training
# batch, and then it performs a reverse pass, computing all the gradients at once (see Appendix B for more details). So what's the difference? Well,
# backpropagation refers to the whole process of training an artificial neural network using multiple backpropagation steps, each of which computes gradients
# and uses them to perform a Gradient Descent step. In contrast, reverse-mode autodiff is just a technique to compute gradients efficiently, and it happens
# to be used by backpropagation.

# 9. ¿Puede enumerar todos los hiperparámetros que puede modificar en un MLP básico? Si el MLP sobreajusta los datos de entrenamiento, ¿cómo podría modificar
# estos hiperparámetros para tratar de resolver el problema?
# Here is a list of all the hyperparameters you can tweak in a basic MLP: the number of hidden layers, the number of neurons in each hidden layer, and the
# activation function used in each hidden layer and in the output layer. In general, the ReLU activation function (or one of its variants; see Chapter 11) is
# a good default for the hidden layers. For the output layer, in general you will want the sigmoid activation function for binary classification, the softmax
# activation function for multiclass classification, or no activation function for regression. If the MLP overfits the training data, you can try reducing
# the number of hidden layers and reducing the number of neurons per hidden layer.

# 10. Entrene un MLP profundo en el conjunto de datos MNIST (puede cargarlo usando tf.keras.datasets.mnist.load_data(). Vea si puede obtener más del 98 % de
# precisión ajustando manualmente los hiperparámetros. Intente buscar la tasa de aprendizaje óptima usando el enfoque presentado en este capítulo (es decir,
# aumentando la tasa de aprendizaje exponencialmente, trazando la pérdida y encontrando el punto donde la pérdida se dispara). A continuación, intente
# ajustar los hiperparámetros usando Keras Tuner con todas las campanas y silbatos: guarde los puntos de control, use la detención temprana y trace las
# curvas de aprendizaje usando TensorBoard.
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full.shape
X_train_full.dtype
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

plt.imshow(X_train[0], cmap="binary")
plt.axis("off")
plt.show()

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.title(y_train[index])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

K = tf.keras.backend


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(
            self.model.optimizer.learning_rate,
            self.model.optimizer.learning_rate * self.factor,
        )


tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
expon_lr = ExponentialLearningRate(factor=1.005)

history = model.fit(
    X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), callbacks=[expon_lr]
)

plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale("log")
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=3e-1)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
run_index = 1  # increment this at every run
run_logdir = Path() / "my_mnist_logs" / "run_{:03d}".format(run_index)
run_logdir

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "my_mnist_model", save_best_only=True
)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
)

model = tf.keras.models.load_model("my_mnist_model")  # rollback to best model
model.evaluate(X_test, y_test)

# in a terminal: tensorboard --logdir=./my_mnist_logs
