import paddle
import pathlib
from typing import Dict, List, Set, Optional, Union, Callable
Tensor = paddle.Tensor


class FirstDerivO2_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, dx):
        ctx.c0 = 0.5 / dx
        ctx.c1 = -0.5 / dx
        return ctx.c0 * tensor0 + ctx.c1 * tensor1

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.c0 * grad_output, ctx.c1 * grad_output, None


class FirstDerivO4_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, tensor2, tensor3, dx):
        ctx.c0 = -1.0 / (dx * 12.0)
        ctx.c1 = 8.0 / (dx * 12.0)
        ctx.c2 = -8.0 / (dx * 12.0)
        ctx.c3 = 1.0 / (dx * 12.0)
        return (ctx.c0 * tensor0 + ctx.c1 * tensor1 + ctx.c2 * tensor2 + 
            ctx.c3 * tensor3)

    @staticmethod
    def backward(ctx, grad_output):
        return (ctx.c0 * grad_output, ctx.c1 * grad_output, ctx.c2 *
            grad_output, ctx.c3 * grad_output, None)


class SecondDerivO2_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, tensor2, dx):
        ctx.c0 = 1.0 / dx ** 2
        ctx.c1 = -2.0 / dx ** 2
        return ctx.c0 * tensor0 + ctx.c1 * tensor1 + ctx.c0 * tensor2

    @staticmethod
    def backward(ctx, grad_output):
        return (ctx.c0 * grad_output, ctx.c1 * grad_output, ctx.c0 *
            grad_output, None)


class SecondDerivO4_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, tensor2, tensor3, tensor4, dx):
        ctx.c0 = -1.0 / (12.0 * dx ** 2)
        ctx.c1 = 4.0 / (3.0 * dx ** 2)
        ctx.c2 = -5.0 / (2.0 * dx ** 2)
        return (ctx.c0 * tensor0 + ctx.c1 * tensor1 + ctx.c2 * tensor2 + 
            ctx.c1 * tensor3 + ctx.c0 * tensor4)

    @staticmethod
    def backward(ctx, grad_output):
        return (ctx.c0 * grad_output, ctx.c1 * grad_output, ctx.c2 *
            grad_output, ctx.c1 * grad_output, ctx.c0 * grad_output, None)


class MixedSecondDerivO2_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, tensor2, tensor3, dx):
        ctx.c0 = 0.25 / dx ** 2
        ctx.c1 = -0.25 / dx ** 2
        return (ctx.c0 * tensor0 + ctx.c1 * tensor1 + ctx.c1 * tensor2 + 
            ctx.c0 * tensor3)

    @staticmethod
    def backward(ctx, grad_output):
        return (ctx.c0 * grad_output, ctx.c1 * grad_output, ctx.c1 *
            grad_output, ctx.c0 * grad_output, None)


class ThirdDerivO2_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, tensor2, tensor3, dx):
        ctx.c0 = 0.5 / dx ** 3
        ctx.c1 = -1.0 / dx ** 3
        ctx.c2 = 1.0 / dx ** 3
        ctx.c3 = -0.5 / dx ** 3
        return (ctx.c0 * tensor0 + ctx.c1 * tensor1 + ctx.c2 * tensor2 + 
            ctx.c3 * tensor3)

    @staticmethod
    def backward(ctx, grad_output):
        return (ctx.c0 * grad_output, ctx.c1 * grad_output, ctx.c2 *
            grad_output, ctx.c3 * grad_output, None)


class ForthDerivO2_f(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, tensor0, tensor1, tensor2, tensor3, tensor4, dx):
        ctx.c0 = 1.0 / dx ** 4
        ctx.c1 = -4.0 / dx ** 4
        ctx.c2 = 6.0 / dx ** 4
        ctx.c3 = -4.0 / dx ** 4
        ctx.c4 = 1.0 / dx ** 4
        return (ctx.c0 * tensor0 + ctx.c1 * tensor1 + ctx.c2 * tensor2 + 
            ctx.c3 * tensor3 + ctx.c4 * tensor4)

    @staticmethod
    def backward(ctx, grad_output):
        return (ctx.c0 * grad_output, ctx.c1 * grad_output, ctx.c2 *
            grad_output, ctx.c3 * grad_output, ctx.c4 * grad_output, None)
