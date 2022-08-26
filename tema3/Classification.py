# code from https://github.com/ageron/handson-ml3/blob/main/03_classification.ipynb

from audioop import cross
from email.mime import multipart
from itertools import chain
from multiprocessing import dummy
import sys
assert sys.version_info >= (3, 7)
import sklearn
assert sklearn.__version__ >= "1.0.1"
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# MNIST

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
# extra code – it's a bit too long
print(mnist.DESCR)
print(mnist.keys())  # extra code – we only use data and target in this notebook

X, y = mnist.data, mnist.target
print(X)
print(X.shape) # 784 es el segundo valor ya que las imagenes son de 28*28 pixeles
print(y)
print(y.shape)

import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
save_fig("some_digit_plot") # extra code
plt.show()

print(y[0])

# extra code – this cell generates and saves Figure 3–2
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)
    plot_digit(image_data)
plt.subplots_adjust(wspace=0, hspace=0)
save_fig("more_digits_plot", tight_layout=False)
plt.show()

X_train, X_Test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Entrenando el clasificador binario

y_train_5 = (y_train == '5') # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier # clasificador Stochastic Gradient Descent (SGD)

sgd_clf = SGDClassifier(random_state=42)
print(sgd_clf.fit(X_train, y_train_5))
print(sgd_clf.predict([some_digit]))

# Medidas de desempeño

# Medir precision usando validacion cruzada

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# Crea tu propio validador cruzado
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True) # add shuffle=True if the dataset is not
                                      # already shuffled
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))

print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# Matriz de confusión:
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

y_train_perfect_predictions = y_train_5 # pretend we reached perfection
print(confusion_matrix(y_train_5, y_train_perfect_predictions))

# Precision y recuperacion
# TP = True positive, FP Falso positivo

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5,y_train_pred)) # == 3530/(687+3530)). Cuantos 5 afirma que ha detectado
# extra code – this cell also computes the precision: TP / (FP + TP)
cm[1, 1] / (cm[0, 1] + cm[1, 1])
print(recall_score(y_train_5, y_train_pred)) # == 3530 / (1891 + 3530)
# extra code – this cell also computes the recall: TP / (FN + TP)
cm[1, 1] / (cm[1, 0] + cm[1, 1])

from sklearn.metrics import f1_score
# relaciona la precision y la recuperacion. + Precision -> - Recuperacion. y viceversa. (Equilibrio entre precision/recuperacion)
print(f1_score(y_train_5, y_train_pred)) 
# extra code – this cell also computes the f1 score
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)

# Intercambio de precision/recuperacion
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
# extra code – just shows that y_scores > 0 produces the same result as
#              calling predict()
y_scores > 0

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv= 3,
                              method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.figure(figsize=(8,4)) # extra code - it's not needed, just formatting
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

# extra code – this section just beautifies and saves Figure 3–5
idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
save_fig("precision_recall_vs_threshold_plot")

plt.show()

import matplotlib.patches as patches  # extra code – for the curved arrow

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

# extra code – just beautifies and saves Figure 3–6
plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko",
         label="Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("precision_vs_recall_plot")

plt.show()                            

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precisioin = thresholds[idx_for_90_precision]
print(threshold_for_90_precisioin)

y_train_pred_90 = (y_scores >= threshold_for_90_precisioin)
print(precision_score(y_train_5, y_train_pred_90))

recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print(recall_at_90_precision)

# The ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precisioin).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

# extra code – just beautifies and saves Figure 3–7
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
save_fig("roc_curve_plot")

plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5, y_scores))

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
print(y_probas_forest[:2])
#These are estimated probabilities. Among the images that the model classified as positive with a 
# probability between 50% and 60%, there are actually about 94% positive images:
# Not in the code
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")# Not in the code
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")

y_scores_forest = y_probas_forest[:,1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

# extra code – just beautifies and saves Figure 3–8
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("pr_curve_comparison_plot")

plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
print(f1_score(y_train_5, y_train_pred_forest))
print(roc_auc_score(y_train_5, y_scores_forest))
print(precision_score(y_train_5, y_train_pred_forest))
print(recall_score(y_train_5, y_train_pred_forest))

# Multiclass Classification
# SVMs do not scale well to large datasets, so let's only train on the first 2,000 instances, or else this section will take a very long time to run:

from sklearn.svm import SVC
svm_clf = SVC(random_state=42)
print(svm_clf.fit(X_train[:2000], y_train[:2000])) # y_train, not y_train_5

svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))

class_id = some_digit_scores.argmax()
print(class_id)
print(svm_clf.classes_)
print(svm_clf.classes_[class_id])

# extra code – shows how to get all 45 OvO scores if needed
svm_clf.decision_function_shape = "ovo"
some_digit_scores_ovo = svm_clf.decision_function([some_digit])
some_digit_scores_ovo.round(2)

from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
print(ovr_clf.fit(X_train[:2000], y_train[:2000]))
print(ovr_clf.predict([some_digit]))
print(len(ovr_clf.estimators_))

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

print(sgd_clf.decision_function([some_digit]).round())
print(cross_val_score(sgd_clf,X_train,y_train, cv=3, scoring="accuracy"))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# # Analisis de error

# from sklearn.metrics import ConfusionMatrixDisplay

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# plt.rc('font', size=9) # extra code - make the text smaller
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# plt.show()

# plt.rc('font', size=10)  # extra code
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
#                                         normalize="true", values_format=".0%")
# plt.show()

# sample_weight = (y_train_pred != y_train)
# plt.rc('font', size=10)  # extra code
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
#                                         sample_weight=sample_weight,
#                                         normalize="true", values_format=".0%")
# plt.show()

# # extra code – this cell generates and saves Figure 3–9
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# plt.rc('font', size=9)
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0])
# axs[0].set_title("Confusion matrix")
# plt.rc('font', size=10)
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[1],
#                                         normalize="true", values_format=".0%")
# axs[1].set_title("CM normalized by row")
# save_fig("confusion_matrix_plot_1")
# plt.show()

# # extra code – this cell generates and saves Figure 3–10
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# plt.rc('font', size=10)
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0],
#                                         sample_weight=sample_weight,
#                                         normalize="true", values_format=".0%")
# axs[0].set_title("Errors normalized by row")
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[1],
#                                         sample_weight=sample_weight,
#                                         normalize="pred", values_format=".0%")
# axs[1].set_title("Errors normalized by column")
# save_fig("confusion_matrix_plot_2")
# plt.show()
# plt.rc('font', size=14)  # make fonts great again

# cl_a, cl_b = '3', '5'
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# # extra code – this cell generates and saves Figure 3–11
# size = 5
# pad = 0.2
# plt.figure(figsize=(size, size))
# for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
#                                        (X_aa, (0, 1)), (X_ab, (1, 1))]:
#     for idx, image_data in enumerate(images[:size*size]):
#         x = idx % size + label_col * (size + pad)
#         y = idx // size + label_row * (size + pad)
#         plt.imshow(image_data.reshape(28, 28), cmap="binary",
#                    extent=(x, x + 1, y, y + 1))
# plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
# plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
# plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
# plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
# plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
# plt.xlabel("Predicted label")
# plt.ylabel("True label")
# save_fig("error_analysis_digits_plot")
# plt.show()

# Clasificacion multietiqueta
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7') # numero mayor o igual que 7
y_train_odd  = (y_train.astype('int8') % 2 == 1) # numero impart
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
print(knn_clf.fit(X_train, y_multilabel))

print(knn_clf.predict([some_digit]))

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

# extra code – shows that we get a negligible performance improvement when we
#              set average="weighted" because the classes are already pretty
#              well balanced.
print(f1_score(y_multilabel, y_train_knn_pred, average="weighted"))

from sklearn.multioutput import ClassifierChain
chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
print(chain_clf.fit(X_train[:2000], y_multilabel[:2000]))
chain_clf.predict([some_digit])

# Multioutput classification. En este caso estamos limpiando el ruido de una imagen para que nos devuelva otra(multiples pixeles de salida = multioutput)
np.random.seed(42)  # to make this code example reproducible
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_Test), 784))
X_test_mod = X_Test + noise
y_train_mod = X_train
y_test_mod = X_Test
# extra code – this cell generates and saves Figure 3–12
plt.subplot(121); plot_digit(X_test_mod[0])
plt.subplot(122); plot_digit(y_test_mod[0])
save_fig("noisy_digit_example_plot")
plt.show()

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
save_fig("cleaned_digit_example_plot")  # extra code – saves Figure 3–13
plt.show()

# Ejercicios:
# 1. Intente crear un clasificador para el conjunto de datos MNIST que logre más del 97 % de precisión en el conjunto de prueba. 
# Sugerencia: KNeighborsClassifierfunciona bastante bien para esta tarea; solo necesita encontrar buenos valores de hiperparámetros 
# (pruebe una búsqueda de cuadrícula en los hiperparámetros weightsy ).n_neighbors
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
baseline_accuracy = knn_clf.score(X_Test, y_test)
print("close: ", baseline_accuracy)

from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train[:10_000], y_train[:10_000])
print(grid_search.best_params_)
print(grid_search.best_score_)

grid_search.best_estimator_.fit(X_train, y_train)
tuned_accuracy = grid_search.score(X_Test, y_test)
print("finnaly: ",tuned_accuracy)

# 2. Escriba una función que pueda desplazar una imagen MNIST en cualquier dirección (izquierda, derecha, arriba o abajo) en un píxel.⁠
# Luego, para cada imagen en el conjunto de entrenamiento, cree cuatro copias desplazadas (una por dirección) y agréguelas al conjunto de entrenamiento. 
# Finalmente, entrene a su mejor modelo en este conjunto de entrenamiento ampliado y mida su precisión en el conjunto de prueba. 
# ¡Debe observar que su modelo funciona aún mejor ahora! Esta técnica de hacer crecer artificialmente el conjunto de entrenamiento se denomina aumento de 
# datos o expansión del conjunto de entrenamiento .

from scipy.ndimage import shift

def shift_image(image, dx, dy):
    image = image.reshape((28,28))
    shifted_image = shift(image,[dy,dx],cval=0, mode="constant")
    return shifted_image.reshape([-1])

# veamos si funciona el metodo

image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original")
plt.imshow(image.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down")
plt.imshow(shifted_image_down.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left")
plt.imshow(shifted_image_left.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.show()

# Creamos el set de entrenamiento aumentado

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((-1,0),(1,0),(0,1),(0,-1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image,dx,dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Barajamos las imagenes movidas para que no esten todas juntas
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

# Entrenamos el modelo con los mejores hiperparametros que habiamos encontrado antes
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
print(knn_clf.fit(X_train_augmented, y_train_augmented))
augmented_accuracy = knn_clf.score(X_Test, y_test)
print(augmented_accuracy)

# gracias a aumentar los datos de entrenamiento hemos aumentado en un 0.5% la tasa de acerto reduciendo el ratio de error en un 17%
error_rate_change = (1 - augmented_accuracy) / (1 - tuned_accuracy) - 1
print(f"error_rate_change = {error_rate_change:.0%}")

# 3. Aborde el conjunto de datos del Titanic. Un buen lugar para comenzar es Kaggle . Alternativamente, puede descargar los datos de 
# https://homl.info/titanic.tgz y descomprimir este tarball como lo hizo con los datos de viviendas en el Capítulo 2 .
# Esto le dará dos archivos CSV: train.csv y test.csv que puede cargar usando pandas.read_csv(). El objetivo es entrenar un clasificador que pueda 
# predecir la Survivedcolumna en función de las otras columnas.

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
train_data, test_data = load_titanic_data()
print(train_data.head())

# establecemos la columna PassengerId como indice de la columna
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

print(train_data.info()) # observamos que hay valores nulos que habra que cambiar
print(train_data[train_data["Sex"]=="female"]["Age"].median())
print(train_data.describe())
print(train_data["Survived"].value_counts())
print(train_data["Pclass"].value_counts()) # vemos los atributos categoricos
print(train_data["Sex"].value_counts())
print(train_data["Embarked"].value_counts()) # C=Cherbourg, Q=Queenstown, S=Southampton

# Construimos nuestros pipelines de procesamiento
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

# Pipeline para atributos categoricos
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
cat_pipeline = Pipeline([
    ("ordinal_encoder", OrdinalEncoder()),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

# unimos los pipelines categoricos y numericos

from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train)
y_train = train_data["Survived"] # obtenemos las etiquetas

# Entrenamos un clasificador
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

# Hacemos las predicciones
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print(forest_scores.mean())

# Vamos a probar con un SVC

from sklearn.svm import SVC
svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print(svm_scores.mean())

# Observamos graficamente que el classificador SVM generaliza mejor 
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()

# Intentando mejorar el modelo con otros atributos
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(
    ['RelativesOnboard']).mean()

# Luego habría que reentrenar el modelo con esos datos y ver los resultados

# 4. Clasificador de spam

import tarfile

# Descargamos los datos
def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (("easy_ham","ham",ham_url),
                                    ("spam", "spam", spam_url)):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            tar_bz2_file = tarfile.open(path)
            tar_bz2_file.extractall(path=spam_path)
            tar_bz2_file.close()
    return [spam_path / dir_name for dir_name in ("easy_ham","spam")]

ham_dir, spam_dir = fetch_spam_data()

# Descargamos los emails
ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]
print("ham_files:", len(ham_filenames))
print("spam_files:", len(spam_filenames))

# Parseamos estos modulos emails con la libreria de paython email(esta maneja las cabezeras, codificados ...)

import email
import email.policy

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
ham_emails = [load_email(filepath) for filepath in ham_filenames]
spam_emails = [load_email(filepath) for filepath in spam_filenames]

# Veamos un ejemplo de cada uno
print(ham_emails[1].get_content().strip())
print(spam_emails[6].get_content().strip())

# Cada email tiene su estructura, con imagenes, archivos adjuntos...
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload() # payload = lo que es realmente importante del mensaje
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email)
                                for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()

from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

print("estructura de emails ham:", structures_counter(ham_emails).most_common())
print("estructura de emails ham:", structures_counter(spam_emails).most_common())

# miramos las cabezeras de los emails
for header, value in spam_emails[0].items():
    print(header, ":", value)
print(spam_emails[0]["Subject"])

# dividimos los datos en entrenamiento y prueba
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pasamos los emails que tienen html a texto plano(seria mejor usar la libreria BeautifulSoup)
import re 
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

# veamos si funciona
html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print("html: ",sample_html_spam.get_content().strip()[:1000], "...")
print("plain text: ", html_to_plain_text(sample_html_spam.get_content())[:1000], "...")

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)
print(email_to_text(sample_html_spam)[:100], "...")

# vamos a hacer un poco de stemming
import nltk

stemmer = nltk.PorterStemmer()
print("steming example: ")
for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
    print(word, "=>", stemmer.stem(word))

# Reemplazamos las URLS por la palabra "URL" usando la libraria urlextract. Tambien se podria haber usado expresiones regulares
import urlextract # may require an Internet connection to download root domain
                  # names
url_extractor = urlextract.URLExtract()
some_text = "Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"
print(url_extractor.find_urls(some_text))

# ponemos todo junto en un transformador para convertir los emails en contadores de palanbras. En este caso usamos el metodo split que usa espacios en 
# blanco para los huecos entre palabras, esto funciona para la mayoria de los idiomas a excepcion del chino o el japones por ejemplo. En este caso no 
# nos afecta ya que la mayoria de emails están en inglés
from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

# Probamos el transformador en algunos emails
X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
print(X_few_wordcounts)

# ahora que tenemos las palabras contadas, necesitamos convertirlas en vectores. Para esto vamosa construit otro transformador cuyo metodo fit()
# va a crear el vocabulario(lista con las palabras mas frecuentes) y cuya transformada va a ser las palabras contadas en un vector. Su salida va a ser
# una sparse matrix

from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1
                            for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)),
                            shape=(len(X), self.vocabulary_size + 1))

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
print(X_few_vectors)
print(X_few_vectors.toarray())
print(vocab_transformer.vocabulary_)

# entrenamos el primer clasificador de spam
from sklearn.pipeline import Pipeline
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector",WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3)
print(score.mean())

from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)
log_clf.fit(X_train_transformed, y_train)
y_pred = log_clf.predict(X_test_transformed)

print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")