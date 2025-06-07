import matplotlib.pyplot as plt

# 1. Dane – wartości accuracy z Twojego projektu
knn_k = [1, 3, 5, 7]
knn_train_acc = [1.000, 0.648, 0.489, 0.446]
knn_test_acc = [0.201, 0.205, 0.229, 0.240]

gnb_var_smooth = [1e-9, 1e-8, 1e-7, 1e-6]
gnb_train_acc = [0.369, 0.369, 0.369, 0.369]
gnb_test_acc = [0.359, 0.359, 0.359, 0.359]

pca_n_components = [5, 10, 15, 17]
pca_train_acc = [0.288, 0.323, 0.361, 0.362]
pca_test_acc = [0.286, 0.313, 0.352, 0.350]

rf_n_estimators = [10, 50, 100, 200]
rf_train_acc = [0.987, 1.000, 1.000, 1.000]
rf_test_acc = [0.340, 0.379, 0.391, 0.389]

# 2. Tworzenie wykresów
plt.figure(figsize=(14, 10))

# k-NN
plt.subplot(2, 2, 1)
plt.plot(knn_k, knn_train_acc, marker='o', label='Train Accuracy')
plt.plot(knn_k, knn_test_acc, marker='s', label='Test Accuracy')
plt.title('k-NN: Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Gaussian NB
plt.subplot(2, 2, 2)
plt.plot(gnb_var_smooth, gnb_train_acc, marker='o', label='Train Accuracy')
plt.plot(gnb_var_smooth, gnb_test_acc, marker='s', label='Test Accuracy')
plt.xscale('log')
plt.title('Gaussian NB: Accuracy vs. var_smoothing')
plt.xlabel('var_smoothing (log scale)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# GNB + PCA
plt.subplot(2, 2, 3)
plt.plot(pca_n_components, pca_train_acc, marker='o', label='Train Accuracy')
plt.plot(pca_n_components, pca_test_acc, marker='s', label='Test Accuracy')
plt.title('GNB + PCA: Accuracy vs. PCA components')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Random Forest
plt.subplot(2, 2, 4)
plt.plot(rf_n_estimators, rf_train_acc, marker='o', label='Train Accuracy')
plt.plot(rf_n_estimators, rf_test_acc, marker='s', label='Test Accuracy')
plt.title('Random Forest: Accuracy vs. n_estimators')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
