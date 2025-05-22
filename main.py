import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

# ---- 1. Wczytaj dane i przetwarzanie ----
df = pd.read_csv("train_data.csv",sep=";")
df.dropna(inplace=True)
X = df.drop("Stay", axis=1)
y = df["Stay"]

# One-hot encoding dla cech kategorycznych
categorical_cols = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

X_processed = preprocessor.fit_transform(X)

# One-hot encoding dla y
y_cat = pd.Categorical(y)
class_names = y_cat.categories
y_encoded = pd.get_dummies(y_cat)

# ---- 2. Podział na zbiory ----
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)


# ---- 3. Skalowanie ----
scaler = StandardScaler(with_mean=False)  # with_mean=False bo mamy sparse matrix
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Konwersja do macierzy


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# ---- 4. Parametry sieci ----
input_size = X_train.shape[1]
hidden_size = 64
output_size = y_train.shape[1]
epochs = 200
learning_rate = 0.01

# ---- 5. Inicjalizacja wag ----
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# ---- 6. Funkcje pomocnicze ----
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stabilność numeryczna
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# ---- 7. Trening ----
for epoch in range(epochs):
    # Forward
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    # Błąd i strata
    loss = cross_entropy(y_train, a2)

    # Backpropagation
    dz2 = a2 - y_train
    dW2 = np.dot(a1.T, dz2) / X_train.shape[0]
    db2 = np.sum(dz2, axis=0, keepdims=True) / X_train.shape[0]

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(a1)
    dW1 = np.dot(X_train.T, dz1) / X_train.shape[0]
    db1 = np.sum(dz1, axis=0, keepdims=True) / X_train.shape[0]

    # Aktualizacja wag
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- 8. Testowanie ----
z1 = np.dot(X_test, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)

y_pred_labels = np.argmax(a2, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true_labels, y_pred_labels)
print(f"Test accuracy: {acc:.4f}")