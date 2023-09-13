import paddle
""" Modulus nodes
"""
from sympy import Add
from .constants import diff_str
from .key import Key


class Node:
    """
    Base class for all nodes used to unroll computational graph in Modulus.

    Parameters
    ----------
    inputs : List[Union[str, Key]]
        Names of inputs to node. For example, `inputs=['x', 'y']`.
    outputs : List[Union[str, Key]]
        Names of outputs to node. For example, `inputs=['u', 'v', 'p']`.
    evaluate : Pytorch Function
        A pytorch function that takes in a dictionary of tensors whose keys are the above `inputs`.
    name : str
        Name of node for print statements and debugging.
    optimize : bool
        If true then any trainable parameters contained in the node will be optimized by the `Trainer`.
    """

    def __init__(self, inputs, outputs, evaluate, name='Node', optimize=False):
        super().__init__()
        self._inputs = Key.convert_list([x for x in inputs if diff_str not in
            str(x)])
        self._outputs = Key.convert_list(outputs)
        self._derivatives = Key.convert_list([x for x in inputs if diff_str in
            str(x)])
        self.evaluate = evaluate
        self._name = name
        self._optimize = optimize
        if not hasattr(self.evaluate, 'saveable'):
            self.evaluate.saveable = False
        if self._optimize:
            assert hasattr(self.evaluate, 'name'
                ), 'Optimizable nodes require model to have unique name'

    @classmethod
    def from_sympy(cls, eq, out_name, freeze_terms=[], detach_names=[]):
        """
        generates a Modulus Node from a SymPy equation

        Parameters
        ----------
        eq : Sympy Symbol/Exp
          the equation to convert to a Modulus Node. The
          inputs to this node consist of all Symbols,
          Functions, and derivatives of Functions. For example,
          `f(x,y) + f(x,y).diff(x) + k` will be converted
          to a node whose input is [`f,f__x,k`].
        out_name : str
          This will be the name of the output for the node.
        freeze_terms : List[int]
          The terms that need to be frozen
        detach_names : List[str]
          This will detach the inputs of the resulting node.

        Returns
        -------
        node : Node
        """
        from modulus.sym.utils.sympy.torch_printer import torch_lambdify, _subs_derivatives, SympyToTorch
        sub_eq = _subs_derivatives(eq)
        if bool(freeze_terms):
            print('the terms ' + str(freeze_terms) +
                ' will be frozen in the equation ' + str(out_name) + ': ' +
                str(Add.make_args(sub_eq)))
            print('Verify before proceeding!')
        else:
            pass
        evaluate = SympyToTorch(sub_eq, out_name, freeze_terms, detach_names)
        inputs = Key.convert_list(evaluate.keys)
        outputs = Key.convert_list([out_name])
        node = cls(inputs, outputs, evaluate, name='Sympy Node: ' + out_name)
        return node

    @property
    def name(self):
        return self._name

    @property
    def outputs(self):
        """
        Returns
        -------
        outputs : List[str]
            Outputs of node.
        """
        return self._outputs

    @property
    def inputs(self):
        """
        Returns
        -------
        inputs : List[str]
            Inputs of node.
        """
        return self._inputs

    @property
    def derivatives(self):
        """
        Returns
        -------
        derivatives : List[str]
            Derivative inputs of node.
        """
        return self._derivatives

    @property
    def optimize(self):
        return self._optimize

    def __str__(self):
        return 'node: ' + self.name + '\n' + 'evaluate: ' + str(self.
            evaluate.__class__.__name__) + '\n' + 'inputs: ' + str(self.inputs
            ) + '\n' + 'derivatives: ' + str(self.derivatives
            ) + '\n' + 'outputs: ' + str(self.outputs
            ) + '\n' + 'optimize: ' + str(self.optimize)
