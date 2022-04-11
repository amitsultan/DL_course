import numpy as np

class ANN:

    def initialize_parameters(self, layer_dims):
        self.n_layers = len(layer_dims)
        b = np.zeros(len(layer_dims))
        params = {}
        for i in range(1, len(layer_dims)):
            params[f'w{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1])
            params[f'b{i}'] = np.zeros((layer_dims[i], 1))
        return params

    def linear_forward(self, A, W, b):
        # Dot product result in output vector than add biases
        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)
        return Z, linear_cache

    def softmax(self, Z):
        activation_cache = Z
        exp_func = np.vectorize(lambda x: np.exp(x))
        result = np.apply_along_axis(lambda row: exp_func(row) / exp_func(row).sum(), 0, Z)
        return result, activation_cache

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
        print(f'W: {W.shape}  A: {A_prev.shape}  Z: {Z.shape}   {activation}')
        return Z, (linear_cache, activation_cache)

    def L_model_forward(self, X, parameters, use_batchnorm):
        A = X
        cache = []
        for layer_index in range(1, self.n_layers - 1):
            A_prev = A
            w_i = parameters[f'w{layer_index}']
            b_i = parameters[f'b{layer_index}']
            A, cache_i = self.linear_activation_forward(A_prev, w_i, b_i, 'relu')
            cache.append(cache_i)
        # output layer wasn't part of the main loop
        w_output = parameters[f'w{self.n_layers - 1}']
        b_output = parameters[f'b{self.n_layers - 1}']
        z_output, cache_output = self.linear_activation_forward(A, w_output, b_output, 'softmax')
        cache.append(cache_output)
        return z_output, cache

    def compute_cost(self, AL, Y):
        y_pred = np.log2(AL)
        cost = -(y_pred * Y).sum() * (1 / AL.shape[1])
        return cost

    def apply_batchnorm(self, A):
        mu = A.mean(1)
        sigma = ((A.T - mu).T ** 2).mean(1)
        z_i = ((A.T - mu) / np.sqrt(sigma + np.finfo(float).eps)).T
        return z_i

    def linear_back(self, dZ, cache):
        m = cache[0].shape[1]
        dW = (1 / m) * dZ*cache[0].T  # A[i-1]
        db = (1 / m) * dZ.sum(1)
        dA_prev = cache[1].T * dZ
        return dA_prev, dW, db

    def activation_derivative(self, dA, activation):
        d_relu = np.vectorize(lambda x: 1 if x > 0 else 0)
        if activation == 'relu':
            dZ = d_relu(dA)
        else:  # softmax

    def linear_activation_backward(self, dA, cache, activation):
        pass

net = ANN()
x = np.random.randn(2, 20)
# params = net.initialize_parameters([2, 5, 2])
# z, _ = net.L_model_forward(x, params, False)
# print(z.shape)
# x = np.random.randint(2, 20, (2, 20))
# x = np.array([[2, 4, 6, 8], [10, 2, 3, 1]])
# print(x)
# net.apply_batchnorm(x)
d_relu =np.vectorize(lambda x: 1 if x > 0 else 0)
print(d_relu([1, -2, 3 , 0]))