from tensorflow.keras.datasets import mnist
import numpy as np

class ANN:

    def initialize_parameters(self, layer_dims):
        self.n_layers = len(layer_dims) - 1
        b = np.zeros(len(layer_dims))
        params = {}
        for i in range(1, len(layer_dims)):
            params[f'w{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt( 2 / layer_dims[i - 1])
            params[f'b{i}'] = np.random.randn(layer_dims[i], 1)
        return params

    def linear_forward(self, A, W, b):
        # Dot product result in output vector than add biases
        Z = np.dot(W, A) + b  # might be matmul?
        linear_cache = {'A': A, 'W': W, 'b': b}
        return Z, linear_cache


    def softmax(self, Z):
        Z = Z - np.max(Z, axis=0)
        exp_z = np.exp(Z)
        return exp_z / np.sum(exp_z, axis=0), Z

    # def softmax(self, Z):
    #     activation_cache = Z
    #     exp_func = np.vectorize(lambda x: np.exp(x))
    #     result = np.apply_along_axis(lambda row: exp_func(row) / exp_func(row).sum(), 0, Z)
    #     return result, activation_cache

    def relu(self, Z):
        activation_cache = Z
        relu_func = np.vectorize(lambda x: x if x > 0 else 0)
        return relu_func(Z), activation_cache

    def linear_activation_forward(self, A_prev, W, B, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, B)
        if activation == 'relu':
            Z, activation_cache = self.relu(Z)
        else:  # softmax
            Z, activation_cache = self.softmax(Z)
        linear_cache['Z'] = activation_cache
        return Z, linear_cache

    def L_model_forward(self, X, parameters, use_batchnorm=False):
        A_prev = X
        cache = []
        for layer_index in range(1, self.n_layers):
            w_i = parameters[f'w{layer_index}']
            b_i = parameters[f'b{layer_index}']
            A_prev, cache_i = self.linear_activation_forward(A_prev, w_i, b_i, 'relu')
            cache.append(cache_i)
        # output layer wasn't part of the main loop
        w_output = parameters[f'w{self.n_layers }']
        b_output = parameters[f'b{self.n_layers }']
        z_output, cache_output = self.linear_activation_forward(A_prev, w_output, b_output, 'softmax')
        cache.append(cache_output)
        return z_output, cache

    def compute_cost(self, AL, Y):
        y_pred = np.log(AL[Y > 0])
        cost = -(y_pred * Y).sum() * (1 / AL.shape[1])
        return cost

    def apply_batchnorm(self, A):
        mu = A.mean(1)
        sigma = ((A.T - mu).T ** 2).mean(1)
        z_i = ((A.T - mu) / np.sqrt(sigma + np.finfo(float).eps)).T
        return z_i

    def linear_backward(self, dZ, cache):
        m = cache['A'].shape[1]
        dW = (1 / m) * np.dot(dZ, cache['A'].T)  # A[i-1]
        db = (1 / m) * dZ.sum(1)
        dA_prev = np.dot(cache['W'].T, dZ)
        return dA_prev, dW, db

    def relu_backward(self, dA, activation_cache):
        Z = activation_cache['Z']
        d_relu = np.vectorize(lambda x: 1 if x > 0 else 0)
        dZ = d_relu(Z)
        return dA * dZ

    def softmax_backward(self, dA, activation_cache):
        Z = activation_cache['Z']
        Y = activation_cache['Y']
        A, _ = self.softmax(Z)
        return A - Y  # L_model_backward sends AL which is enough for AL - Y but we created A from Z

    def linear_activation_backward(self, dA, cache, activation):
        if activation == 'relu':
            dZ = self.relu_backward(dA, cache)
        else:  # softmax
            dZ = self.softmax_backward(dA, cache)
        return self.linear_backward(dZ,cache)

    def L_model_backward(self, AL, Y, caches):
        index = self.n_layers - 1
        grads = {}
        caches[index]['Y'] = Y
        cache = caches[index]
        dA_prev, dW, db = self.linear_activation_backward(AL, cache, 'softmax')
        grads[f'dA{self.n_layers}'] = dA_prev
        grads[f'dW{self.n_layers}'] = dW
        grads[f'db{self.n_layers}'] = db
        for i in range(1, index + 1):
            cache = caches[index - i]
            dA_prev, dW, db = self.linear_activation_backward(dA_prev, cache, 'relu')
            grads[f'dA{self.n_layers- i}'] = dA_prev
            grads[f'dW{self.n_layers- i}'] = dW
            grads[f'db{self.n_layers- i}'] = db
        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        layers_dim = int(len(parameters) / 2)
        for layer in range(1, layers_dim + 1):
            parameters[f'w{layer}'] -= learning_rate * grads[f'dW{layer}']
            parameters[f'b{layer}'] -= learning_rate * grads[f'db{layer}'].reshape(-1, 1)
        return parameters

    def L_layer_model(self, X, Y, layers_dims, learning_rate, num_iterations, batch_size):
        prev_val_loss = np.inf
        num_classes = len(np.unique(Y))
        training_step = 0
        ind = np.arange(X.shape[1])
        np.random.shuffle(ind)
        X_validation = X[:,ind[:int(np.ceil(0.2*X.shape[1]))]]
        Y_validation = Y[:,ind[:int(np.ceil(0.2*X.shape[1]))]]

        X_train = X[:,ind[int(np.ceil(0.2*X.shape[1])):]]
        Y_train = Y[:,ind[int(np.ceil(0.2*X.shape[1])):]]

        params = self.initialize_parameters(layers_dims)
        history = []
        accuracy_history = []
        for epoch in range(num_iterations):
            for index in range(0, X_train.shape[1], batch_size):
                batch_x = X_train[ :,index:min(index + batch_size, X_train.shape[1])]
                batch_y = Y_train[:, index:min(index + batch_size, Y_train.shape[1])]
                z_output, cache = self.L_model_forward(batch_x, params, False)
                if (index / 500)%100 == 0:
                    acc = self.predict(X_validation, Y_validation, params)
                    cost = self.compute_cost(z_output, batch_y)
                    history.append(cost)
                    accuracy_history.append(acc)
                    print(f"{training_step} Training Step - Accuracy = {acc}, cost = {cost}")
                    training_step += 1
                grads = self.L_model_backward(z_output, batch_y, cache)
                params = self.update_parameters(params, grads, learning_rate)
        return params, history

    def predict(self, X, Y, parameters):
        y_preds, cache = self.L_model_forward(X, parameters, False)
        a = np.argmax(y_preds, axis=0)
        y = np.argmax(Y, axis=0)
        acc = np.sum(np.equal(y, a)) / len(y)
        return acc

def convert_to_onehot_vector(Y):
    Y_oneHot = np.zeros((10, len(Y)))
    Y_oneHot[Y, np.arange(len(Y))] = 1
    return Y_oneHot


def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], int(train_X.shape[1] * train_X.shape[2])).T / 255
    test_X = test_X.reshape(test_X.shape[0], int(test_X.shape[1] * test_X.shape[2])).T / 255
    train_y = convert_to_onehot_vector(train_y)
    test_y = convert_to_onehot_vector(test_y)
    return train_X, test_X, train_y, test_y

net = ANN()
train_X, test_X, train_y, test_y = load_data()
print(train_X.shape)
params, history = net.L_layer_model(train_X, train_y, [784, 20, 7, 5, 10], 0.009, 30, 128)
print(history)
