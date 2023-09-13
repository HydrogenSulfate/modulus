import paddle


class WeightNormLinear(paddle.nn.Layer):

    def __init__(self, in_features: int, out_features: int, bias: bool=True
        ) ->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        out_73 = paddle.create_parameter(shape=paddle.empty(shape=(
            out_features, in_features)).shape, dtype=paddle.empty(shape=(
            out_features, in_features)).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.empty(shape=(out_features,
            in_features))))
        out_73.stop_gradient = not True
        self.weight = out_73
        out_74 = paddle.create_parameter(shape=paddle.empty(shape=(
            out_features, 1)).shape, dtype=paddle.empty(shape=(out_features,
            1)).numpy().dtype, default_initializer=paddle.nn.initializer.
            Assign(paddle.empty(shape=(out_features, 1))))
        out_74.stop_gradient = not True
        self.weight_g = out_74
        if bias:
            out_75 = paddle.create_parameter(shape=paddle.empty(shape=
                [out_features]).shape, dtype=paddle.empty(shape=[out_features])
                .numpy().dtype, default_initializer=paddle.nn.initializer.
                Assign(paddle.empty(shape=[out_features])))
            out_75.stop_gradient = not True
            self.bias = out_75
        else:
            self.add_parameter(name='bias', parameter=None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        paddle.nn.initializer.XavierUniform(self.weight)
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(self.weight_g)
        if self.bias is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.bias)

    def forward(self, input: paddle.Tensor) ->paddle.Tensor:
        norm = self.weight.norm(axis=1, p=2, keepdim=True)
        weight = self.weight_g * self.weight / norm
        return paddle.nn.functional.linear(weight=weight.T, bias=self.bias,
            x=input)

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)
