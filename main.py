from keras.datasets import cifar10
import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# GLOBAL VARIABLES

TRAIN_SIZE = 50000
TEST_SIZE = 10000
CROSS_VALIDATION_SPLIT = 0.1
REDUCED_TRAIN_SIZE = 5000
REDUCED_TEST_SIZE = 1000
coeff = 0.5
degree = 5
gamma = 1.0 / 100
C = 0.001
chosen_class = 0
pca_components = 100



#DATASET HANDLING
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=CROSS_VALIDATION_SPLIT, random_state=42)

class_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

X_train = X_train.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)
#X_val = X_val.reshape(-1, 32*32*3)
X_train = X_train.astype('float64') / 255
X_test = X_test.astype('float64') / 255
#X_val = X_val.astype('float64') / 255

y_test = np.reshape(y_test, TEST_SIZE)
y_train = np.reshape(y_train, TRAIN_SIZE)
#y_val = np.reshape(y_val, int(TRAIN_SIZE * CROSS_VALIDATION_SPLIT))

#"""
#5000 SAMPLES FOR TESTING PURPOSES
X_train = X_train[:REDUCED_TRAIN_SIZE,:]
y_train = y_train[:REDUCED_TRAIN_SIZE]
X_test = X_test[:REDUCED_TEST_SIZE,:]
y_test = y_test[:REDUCED_TEST_SIZE]
#"""

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

y_train[y_train > chosen_class] = -1.0
y_test[y_test > chosen_class] = -1.0
y_train[y_train == chosen_class] = 1.0
y_test[y_test == chosen_class] = 1.0
#print(f"ytrain before{y_train[:100]}")
#print(f"ytest before{y_test[:100]}")

pca = PCA(n_components=pca_components)
pca.fit(X_train)
X_train = pca.transform(X_train)
#pca.fit(X_test)
X_test = pca.transform(X_test)


#d = degree of polynomial | c = coefficient
def compute_polynomial_kernel(x, c, d, g):
    return np.power(np.add(np.multiply(g, np.matmul(x, x.T)), c), d)

def compute_k_matrix_on_test_data(x_data, x_new_cases, c, d, g):
    return np.power(np.add(np.multiply(g, np.matmul(x_data, x_new_cases.T)), c), d)

def quadratic_solution(k_matrix, y, C):
    y_size = len(y)
    y = np.reshape(y, (y_size,1))
    y = y.astype(np.float64)
    print(f"---inside quadratic_solution--- y shape: {y.shape}, y dtype: {y.dtype}")
    P = matrix(np.matmul(y, y.T) * k_matrix)
    q = matrix(-np.ones(y_size))
    G = matrix(np.vstack((-np.eye(y_size), np.eye(y_size))))
    h = matrix(np.hstack((np.zeros(shape=y_size), C * np.ones(shape=y_size))))
    A = matrix(np.transpose(y))
    b = matrix(0.0)
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])
    return alphas

def support_vectors_indices(alphas):
    return np.flatnonzero(alphas)

def alpha_y_multiplication(alphas, y):
    return np.multiply(alphas, y)

def compute_bias(alphas, y, k_matrix, sv_indice, ay):
    print(f"---inside compute_bias--- shape of k_matrix[sv_indice]: {k_matrix[sv_indice].shape}, shape of alphas: {alphas.shape}, shape of y: {y.shape}, shape of ay: {ay.shape}")
    return y[sv_indice] - np.sum(np.multiply(ay, k_matrix[sv_indice]))

def predict(X_test, c, d, g, b, X_train, ay):
    test_length = len(X_test)
    kernel = compute_k_matrix_on_test_data(X_train, X_test, c, d, g)
    #ay_tiled = np.tile(ay, test_length)
    ay = np.reshape(ay, (len(ay), 1))
    kernel_sum = np.sum(ay * kernel, axis=0)
    print(f"---inside predict--- shape of train*test kernel: {kernel.shape}, shape of ay: {ay.shape}, shape of kernel_sum: {kernel_sum.shape}")
    return np.sign(kernel_sum + b)


if __name__ == "__main__":
    print(f"REDUCED TRAINING SET: {X_train.shape}")
    print(f"DEGREE: {degree}")
    print(f"CHOSEN CLASS: {class_dict[chosen_class]}")
    print(f"x_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    kernel_matrix = compute_polynomial_kernel(X_train, coeff, degree, gamma)
    print(f"polynomial kernel matrix: {kernel_matrix.shape}")

    alphas_lagrange = quadratic_solution(kernel_matrix, y_train, C)
    print(f"alphas lagrange matrix shape: {alphas_lagrange.shape}")

    sv_indices = support_vectors_indices(alphas_lagrange)
    print(f"number of support vectors: {len(sv_indices)}")

    alpha_y_product = alpha_y_multiplication(alphas_lagrange, y_train)
    print(f"a*y shape: {alpha_y_product.shape}")

    bias = compute_bias(alphas_lagrange, y_train, kernel_matrix, sv_indices[0],alpha_y_product)
    print(f"bias shape: {bias.shape}, bias: {bias}")

    y_pred = predict(X_test, coeff, degree, gamma, bias, X_train, alpha_y_product)

    #print(f"y test after: {y_test[:100]}")
    #print(f"y pred after: {y_pred[:100]}")

    print(f"accuracy: {accuracy_score(y_test, y_pred)}")