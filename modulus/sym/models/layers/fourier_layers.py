import paddle
import math
import numpy as np


class FourierLayer(paddle.nn.Layer):

    def __init__(self, in_features: int, frequencies) ->None:
        super().__init__()
        if isinstance(frequencies[0], str):
            if 'gaussian' in frequencies[0]:
                nr_freq = frequencies[2]
                np_f = np.random.normal(0, 1, size=(nr_freq, in_features)
                    ) * frequencies[1]
            else:
                nr_freq = len(frequencies[1])
                np_f = []
                if 'full' in frequencies[0]:
                    np_f_i = np.meshgrid(*[np.array(frequencies[1]) for _ in
                        range(in_features)], indexing='ij')
                    np_f.append(np.reshape(np.stack(np_f_i, axis=-1), (
                        nr_freq ** in_features, in_features)))
                if 'axis' in frequencies[0]:
                    np_f_i = np.zeros((nr_freq, in_features, in_features))
                    for i in range(in_features):
                        np_f_i[:, i, i] = np.reshape(np.array(frequencies[1
                            ]), nr_freq)
                    np_f.append(np.reshape(np_f_i, (nr_freq * in_features,
                        in_features)))
                if 'diagonal' in frequencies[0]:
                    np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq,
                        1, 1))
                    np_f_i = np.tile(np_f_i, (1, in_features, in_features))
                    np_f_i = np.reshape(np_f_i, (nr_freq * in_features,
                        in_features))
                    np_f.append(np_f_i)
                np_f = np.concatenate(np_f, axis=-2)
        else:
            np_f = frequencies
        frequencies = paddle.to_tensor(data=np_f, dtype=paddle.
            get_default_dtype())
        frequencies = frequencies.t()
        self.register_buffer(name='frequencies', tensor=frequencies)

    def out_features(self) ->int:
        return int(self.frequencies.shape[1] * 2)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x_hat = paddle.matmul(x=x, y=self.frequencies)
        x_sin = paddle.sin(x=2.0 * math.pi * x_hat)
        x_cos = paddle.cos(x=2.0 * math.pi * x_hat)
        x_i = paddle.concat(x=[x_sin, x_cos], axis=-1)
        return x_i


class FourierFilter(paddle.nn.Layer):

    def __init__(self, in_features: int, layer_size: int, nr_layers: int,
        input_scale: float) ->None:
        super().__init__()
        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)
        out_60 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_features, layer_size]).shape, dtype=paddle.empty(shape=[
            in_features, layer_size]).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.empty(shape=[in_features,
            layer_size])))
        out_60.stop_gradient = not True
        self.frequency = out_60
        out_61 = paddle.create_parameter(shape=paddle.empty(shape=
            layer_size).shape, dtype=paddle.empty(shape=layer_size).numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(paddle
            .empty(shape=layer_size)))
        out_61.stop_gradient = not True
        self.phase = out_61
        self.reset_parameters()

    def reset_parameters(self) ->None:
        paddle.nn.initializer.XavierNormal()(self.frequency)
        paddle.nn.initializer.Uniform()(self.phase, -math.pi, math.pi)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        frequency = self.weight_scale * self.frequency
        x_i = paddle.sin(x=paddle.matmul(x=x, y=2.0 * math.pi * frequency) +
            self.phase)
        return x_i


class GaborFilter(paddle.nn.Layer):

    def __init__(self, in_features: int, layer_size: int, nr_layers: int,
        input_scale: float, alpha: float, beta: float) ->None:
        super().__init__()
        self.layer_size = layer_size
        self.alpha = alpha
        self.beta = beta
        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)
        out_62 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_features, layer_size]).shape, dtype=paddle.empty(shape=[
            in_features, layer_size]).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.empty(shape=[in_features,
            layer_size])))
        out_62.stop_gradient = not True
        self.frequency = out_62
        out_63 = paddle.create_parameter(shape=paddle.empty(shape=
            layer_size).shape, dtype=paddle.empty(shape=layer_size).numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(paddle
            .empty(shape=layer_size)))
        out_63.stop_gradient = not True
        self.phase = out_63
        out_64 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_features, layer_size]).shape, dtype=paddle.empty(shape=[
            in_features, layer_size]).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.empty(shape=[in_features,
            layer_size])))
        out_64.stop_gradient = not True
        self.mu = out_64
        out_65 = paddle.create_parameter(shape=paddle.empty(shape=
            layer_size).shape, dtype=paddle.empty(shape=layer_size).numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(paddle
            .empty(shape=layer_size)))
        out_65.stop_gradient = not True
        self.gamma = out_65
        self.reset_parameters()

    def reset_parameters(self) ->None:
        paddle.nn.initializer.XavierNormal().xavier_uniform_(self.frequency)
        paddle.nn.initializer.Uniform()(self.phase, -math.pi, math.pi)
        paddle.nn.initializer.Uniform()(self.mu, -1.0, 1.0)
        with paddle.no_grad():
            paddle.assign(paddle.to_tensor(data=np.random.gamma(self.alpha,
                1.0 / self.beta, self.layer_size)), output=self.gamma)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        frequency = self.weight_scale * (self.frequency * self.gamma.sqrt())
        x_c = x.unsqueeze(axis=-1)
        x_c = x_c - self.mu
        x_c = paddle.square(x=x_c.norm(p=2, axis=-2))
        x_c = paddle.exp(x=-0.5 * x_c * self.gamma)
        x_i = x_c * paddle.sin(x=paddle.matmul(x=x, y=2.0 * math.pi *
            frequency) + self.phase)
        return x_i
