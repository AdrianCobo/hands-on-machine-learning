# code from: https://github.com/ageron/handson-ml3/blob/main/06_decision_trees.ipynb
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

IMAGES_PATH = Path() / "images" / "decision_trees"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Entrenamiento y visualizacion de un arbol de decision
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

# graficamos el arbol
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file=str(IMAGES_PATH / "iris_tree.dot"),  # path differs in the book
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

from graphviz import Source

Source.from_file(IMAGES_PATH / "iris_tree.dot")  # path differs in the book
# extra code
# sudo dot -Tpng images/decision_trees/iris_tree.dot -o images/decision_trees/iris_tree.png convierte .dot to .png

# haciendo predicciones
import numpy as np
import matplotlib.pyplot as plt

# extra code – just formatting details
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
plt.figure(figsize=(8, 4))

lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)
for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
             style, label=f"Iris {name}")

# extra code – this section beautifies and saves Figure 6–2
tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_deeper.fit(X_iris, y_iris)
th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.plot([th0, th0], [0, 3], "k-", linewidth=2)
plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)
plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)
plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)
plt.text(th0 - 0.05, 1.0, "Depth=0", horizontalalignment="right", fontsize=15)
plt.text(3.2, th1 + 0.02, "Depth=1", verticalalignment="bottom", fontsize=13)
plt.text(th2a + 0.05, 0.5, "(Depth=2)", fontsize=11)
plt.axis([0, 7.2, 0, 3])
plt.legend()
save_fig("decision_tree_decision_boundaries_plot")

plt.show()
print("acces to the tree structure:",tree_clf.tree_)
# # help(sklearn.tree._tree.Tree) # documentation

# Estimacion de probabilidades de clase / Estimating Class Probabilities
print(tree_clf.predict_proba([[5, 1.5]]).round(3))
print(tree_clf.predict([[5, 1.5]]))

# Hiperparametros de regularización / Regularization Hyperparameters
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)

# extra code – this cell generates and saves Figure 6–3

def plot_decision_boundary(clf, X, y, axes, cmap):
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
    colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[cmap][idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf1, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title("No restrictions")
plt.sca(axes[1])
plot_decision_boundary(tree_clf2, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"min_samples_leaf = {tree_clf2.min_samples_leaf}")
plt.ylabel("")
save_fig("min_samples_leaf_plot")
plt.show()

X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)
print("no regularizado", tree_clf1.score(X_moons_test, y_moons_test))
print("regularizado", tree_clf2.score(X_moons_test, y_moons_test))

# Regresion / Regression
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5 # a single random input feature
y_quad = X_quad ** 2 + 0.025 * np.random.rand(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
# extra code – we've already seen how to use export_graphviz()
export_graphviz(
    tree_reg,
    out_file=str(IMAGES_PATH / "regression_tree.dot"),
    feature_names=["x1"],
    rounded=True,
    filled=True
)
Source.from_file(IMAGES_PATH / "regression_tree.dot")
# sudo dot -Tpng images/decision_trees/regression_tree.dot -o images/decision_trees/regression_tree.png

tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg2.fit(X_quad, y_quad)
print(tree_reg.tree_.threshold)
print(tree_reg2.tree_.threshold)

# extra code – this cell generates and saves Figure 6–5

def plot_regression_predictions(tree_reg, X, y, axes=[-0.5, 0.5, -0.05, 0.25]):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_regression_predictions(tree_reg, X_quad, y_quad)

th0, th1a, th1b = tree_reg.tree_.threshold[[0, 1, 4]]
for split, style in ((th0, "k-"), (th1a, "k--"), (th1b, "k--")):
    plt.plot([split, split], [-0.05, 0.25], style, linewidth=2)
plt.text(th0, 0.16, "Depth=0", fontsize=15)
plt.text(th1a + 0.01, -0.01, "Depth=1", horizontalalignment="center", fontsize=13)
plt.text(th1b + 0.01, -0.01, "Depth=1", fontsize=13)
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper center", fontsize=16)
plt.title("max_depth=2")

plt.sca(axes[1])
th2s = tree_reg2.tree_.threshold[[2, 5, 9, 12]]
plot_regression_predictions(tree_reg2, X_quad, y_quad)
for split, style in ((th0, "k-"), (th1a, "k--"), (th1b, "k--")):
    plt.plot([split, split], [-0.05, 0.25], style, linewidth=2)
for split in th2s:
    plt.plot([split, split], [-0.05, 0.25], "k:", linewidth=1)
plt.text(th2s[2] + 0.01, 0.15, "Depth=2", fontsize=13)
plt.title("max_depth=3")

save_fig("tree_regression_plot")
plt.show()

# extra code – this cell generates and saves Figure 6–6

tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X_quad, y_quad)
tree_reg2.fit(X_quad, y_quad)

x1 = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

plt.sca(axes[0])
plt.plot(X_quad, y_quad, "b.")
plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([-0.5, 0.5, -0.05, 0.25])
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper center")
plt.title("No restrictions")

plt.sca(axes[1])
plt.plot(X_quad, y_quad, "b.")
plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([-0.5, 0.5, -0.05, 0.25])
plt.xlabel("$x_1$")
plt.title(f"min_samples_leaf={tree_reg2.min_samples_leaf}")

save_fig("tree_regression_regularization_plot")
plt.show()

# Sensibilidad a la orientacion del eje / Sensitivity to axis orientation
# extra code – this cell generates and saves Figure 6–7

np.random.seed(6)
X_square = np.random.rand(100, 2) - 0.5
y_square = (X_square[:, 0] > 0).astype(np.int64)

angle = np.pi / 4  # 45 degrees
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)

tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf_square, X_square, y_square,
                       axes=[-0.7, 0.7, -0.7, 0.7], cmap="Pastel1")
plt.sca(axes[1])
plot_decision_boundary(tree_clf_rotated_square, X_rotated_square, y_square,
                       axes=[-0.7, 0.7, -0.7, 0.7], cmap="Pastel1")
plt.ylabel("")

save_fig("sensitivity_to_rotation_plot")
plt.show()

# vamos a entrenar el modelo pero rotando los datos para que se le haga mas facil
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)

# extra code – this cell generates and saves Figure 6–8

plt.figure(figsize=(8, 4))

axes = [-2.2, 2.4, -0.6, 0.7]
z0s, z1s = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                       np.linspace(axes[2], axes[3], 100))
X_iris_pca_all = np.c_[z0s.ravel(), z1s.ravel()]
y_pred = tree_clf_pca.predict(X_iris_pca_all).reshape(z0s.shape)

plt.contourf(z0s, z1s, y_pred, alpha=0.3, cmap=custom_cmap)
for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris_rotated[:, 0][y_iris == idx],
             X_iris_rotated[:, 1][y_iris == idx],
             style, label=f"Iris {name}")

plt.xlabel("$z_1$")
plt.ylabel("$z_2$", rotation=0)
th1, th2 = tree_clf_pca.tree_.threshold[[0, 2]]
plt.plot([th1, th1], axes[2:], "k-", linewidth=2)
plt.plot([th2, th2], axes[2:], "k--", linewidth=2)
plt.text(th1 - 0.01, axes[2] + 0.05, "Depth=0",
         horizontalalignment="right", fontsize=15)
plt.text(th2 - 0.01, axes[2] + 0.05, "Depth=1",
         horizontalalignment="right", fontsize=13)
plt.axis(axes)
plt.legend(loc=(0.32, 0.67))
save_fig("pca_preprocessing_plot")

plt.show()

# Los arboles de decision tienen una varianza alta / Decision Trees Have High Variance
# Afortunadamente, al promediar las predicciones de muchos árboles, es posible reducir significativamente la varianza. Tal conjunto de árboles se llama 
# Random Forest , y es uno de los tipos de modelos más poderosos disponibles en la actualidad, como veremos en el próximo capítulo.
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40) # distinto random state.
tree_clf_tweaked.fit(X_iris, y_iris)

# extra code – this cell generates and saves Figure 6–9
# incluso con los mismos datos, al cambiar la aleatoriedad que usa el algoritmo, da un modelo distinto al usar un algotitmo de entrenamiento estocastico

plt.figure(figsize=(8, 4))
y_pred = tree_clf_tweaked.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)

for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
             style, label=f"Iris {name}")

th0, th1 = tree_clf_tweaked.tree_.threshold[[0, 2]]
plt.plot([0, 7.2], [th0, th0], "k-", linewidth=2)
plt.plot([0, 7.2], [th1, th1], "k--", linewidth=2)
plt.text(1.8, th0 + 0.05, "Depth=0", verticalalignment="bottom", fontsize=15)
plt.text(2.3, th1 + 0.05, "Depth=1", verticalalignment="bottom", fontsize=13)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.axis([0, 7.2, 0, 3])
plt.legend()
save_fig("decision_tree_high_variance_plot")

plt.show()

# Material extra: Acceso a la estructura del arbol:
tree = tree_clf.tree_
print(tree)
print("nodos:",tree.node_count)
print("profundidad maxima:",tree.max_depth)
print("maximo numero de clases:",tree.max_n_classes)
print("atributos de los nodos:",tree.n_features)
print("numero de salidas del arbol",tree.n_outputs)
print("nodos hojas:", tree.n_leaves)
print("impurezas:",tree.impurity)
print("nodos a la izquierda y derecha de la raiz:",tree.impurity)
print("threshold del arbol:",tree.threshold)
# computar la profundidad de cada nodo
def compute_depth(tree_clf):
    tree = tree_clf.tree_
    depth = np.zeros(tree.node_count)
    stack = [(0, 0)]
    while stack:
        node, node_depth = stack.pop()
        depth[node] = node_depth
        if tree.children_left[node] != tree.children_right[node]:
            stack.append((tree.children_left[node], node_depth + 1))
            stack.append((tree.children_right[node], node_depth + 1))
    return depth

depth = compute_depth(tree_clf)
print(depth)

# Ejercicios

# 1. ¿Cuál es la profundidad aproximada de un árbol de decisión entrenado (sin restricciones) en un conjunto de entrenamiento con un millón de instancias?
# The depth of a well-balanced binary tree containing m leaves is equal to log₂(m), rounded up. log₂ is the binary log; log₂(m) = log(m) / log(2). 
# A binary Decision Tree (one that makes only binary decisions, as is the case with all trees in Scikit-Learn) will end up more or less well balanced at the 
# end of training, with one leaf per training instance if it is trained without restrictions. Thus, if the training set contains one million instances, 
# the Decision Tree will have a depth of log₂(106) ≈ 20 (actually a bit more since the tree will generally not be perfectly well balanced).

# 2. ¿Es la impureza de Gini de un nodo generalmente menor o mayor que la de su padre? ¿Es generalmente menor/mayor, o siempre menor/mayor?
# A node's Gini impurity is generally lower than its parent's. This is due to the CART training algorithm's cost function, which splits each node in a way 
# that minimizes the weighted sum of its children's Gini impurities. However, it is possible for a node to have a higher Gini impurity than its parent, as 
# long as this increase is more than compensated for by a decrease in the other child's impurity. For example, consider a node containing four instances of 
# class A and one of class B. Its Gini impurity is 1 – (1/5)² – (4/5)² = 0.32. Now suppose the dataset is one-dimensional and the instances are lined up in 
# the following order: A, B, A, A, A. You can verify that the algorithm will split this node after the second instance, producing one child node with 
# instances A, B, and the other child node with instances A, A, A. The first child node's Gini impurity is 1 – (1/2)² – (1/2)² = 0.5, which is higher than 
# its parent's. This is compensated for by the fact that the other node is pure, so its overall weighted Gini impurity is 2/5 × 0.5 + 3/5 × 0 = 0.2, which is 
# lower than the parent's Gini impurity.

# 3. Si un árbol de decisión está sobreajustando el conjunto de entrenamiento, ¿es una buena idea intentar disminuirlo max_depth?
# If a Decision Tree is overfitting the training set, it may be a good idea to decrease max_depth, since this will constrain the model, regularizing it.

# 4. Si un árbol de decisión no se ajusta al conjunto de entrenamiento, ¿es una buena idea intentar escalar las características de entrada?
# Decision Trees don't care whether or not the training data is scaled or centered; that's one of the nice things about them. So if a Decision Tree underfits 
# the training set, scaling the input features will just be a waste of time.

# 5. Si lleva una hora entrenar un árbol de decisiones en un conjunto de entrenamiento que contiene 1 millón de instancias, ¿cuánto tiempo llevará 
# aproximadamente entrenar otro árbol de decisiones en un conjunto de entrenamiento que contiene 10 millones de instancias? Sugerencia: considere la 
# complejidad computacional del algoritmo CART.
# The computational complexity of training a Decision Tree is O(n × m log₂(m)). So if you multiply the training set size by 10, the training time will be 
# multiplied by K = (n × 10 m × log₂(10 m)) / (n × m × log₂(m)) = 10 × log₂(10 m) / log₂(m). If m = 106, then K ≈ 11.7, so you can expect the training time 
# to be roughly 11.7 hours.

# 6. Si lleva una hora entrenar un árbol de decisión en un conjunto de entrenamiento determinado, ¿cuánto tiempo tardará aproximadamente si duplica la 
# cantidad de funciones?
# If the number of features doubles, then the training time will also roughly double.

# 7. Entrene y ajuste un árbol de decisiones para el conjunto de datos de lunas siguiendo estos pasos:

# 7.1 Úselo make_moons(n_samples=10000, noise=0.4)para generar un conjunto de datos de lunas.
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

# 7.2 Úselo train_test_split()para dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons,
                                                    test_size=0.2,
                                                    random_state=42)

# 7.3 Utilice la búsqueda en cuadrícula con validación cruzada (con la ayuda de la GridSearchCVclase) para encontrar buenos valores de hiperparámetros para 
# un archivo DecisionTreeClassifier. Sugerencia: pruebe varios valores para max_leaf_nodes.
from sklearn.model_selection import GridSearchCV
params = {
    'max_leaf_nodes' : list(range(2, 100)),
    'max_depth': list(range(1,7)),
    'min_samples_split': [2, 3, 4]
}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=3)
print(grid_search_cv.fit(X_train, y_train))
print(grid_search_cv.best_estimator_)
# 7.4 Entrénelo en el conjunto de entrenamiento completo utilizando estos hiperparámetros y mida el rendimiento de su modelo en el conjunto de prueba. 
# Deberías obtener una precisión aproximada del 85 % al 87 %.
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
print("accuracy score:",accuracy_score(y_test, y_pred))

# 8. Haz crecer un bosque siguiendo estos pasos:

# 8.1 Continuando con el ejercicio anterior, genere 1000 subconjuntos del conjunto de entrenamiento, cada uno con 100 instancias seleccionadas al azar. 
# Sugerencia: puede usar la ShuffleSplitclase de Scikit-Learn para esto.
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

# 8.2 Entrene un árbol de decisiones en cada subconjunto, utilizando los mejores valores de hiperparámetros encontrados en el ejercicio anterior. 
# Evalúe estos 1000 árboles de decisión en el conjunto de prueba. Dado que se entrenaron en conjuntos más pequeños, es probable que estos árboles de decisión 
# funcionen peor que el primer árbol de decisión, logrando solo un 80 % de precisión .
from sklearn.base import clone
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]
accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
print(np.mean(accuracy_scores))

# 8.3 Ahora viene la magia. Para cada instancia del conjunto de prueba, genere las predicciones de los 1000 árboles de decisión y conserve solo la predicción 
# más frecuente (puede usar la mode()función de SciPy para esto). Este enfoque le brinda predicciones de voto mayoritario sobre el conjunto de prueba.
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

# 8.4 Evalúe estas predicciones en el conjunto de prueba: debería obtener una precisión ligeramente superior a la de su primer modelo (alrededor de un 0,5 a 
# un 1,5 % más). ¡Felicitaciones, has entrenado a un clasificador de Random Forest!
print("forest accuracy:", accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))