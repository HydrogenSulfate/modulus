import paddle
""" Helper functions for unrolling computational graph
"""
from copy import copy
import logging
from typing import Dict, List, Optional
from .models.arch import Arch, FuncArch
from .node import Node
from .key import Key
from .constants import diff_str
from .eq.derivatives import Derivative
from .manager import JitManager, GraphManager
logger = logging.getLogger(__name__)


class Graph(paddle.nn.Layer):
    """
    Torch Module that is constructed by unrolling a computational graph given
    desired inputs, outputs, and evaluatable nodes.

    Examples
    ========
    Here is a simple example of using `Graph` to unroll a two node graph.
    >>> import torch
    >>> from sympy import Symbol
    >>> from modulus.sym.node import Node
    >>> from modulus.sym.key import Key
    >>> from modulus.sym.graph import Graph
    >>> node_1 = Node.from_sympy(Symbol('x') + Symbol('y'), 'u')
    >>> node_2 = Node.from_sympy(Symbol('u') + 1.0, 'v')
    >>> graph = Graph([node_1, node_2], [Key('x'), Key('y')], [Key('v')])
    >>> graph.forward({'x': torch.tensor([1.0]), 'y': torch.tensor([2.0])})
    {'v': tensor([4.])}

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    invar : List[Key]
        List of inputs to graph.
    req_names : List[Key]
        List of required outputs of graph.
    diff_nodes : List[Node]
        List of specialty nodes to compute derivatives.
        By default this is not needed.
    func_arch : bool, Optional
        If True, find the computable derivatives that are part of the Jacobian and
        Hessian of the neural network. They will be calculated during the forward
        pass using FuncArch.
        If None (default), will use the GraphManager to get the global flag
        (default is False), which could be configured in the hydra config with key
        `graph.func_arch`.
    func_arch_allow_partial_hessian : bool, Optional
        If True, allow evaluating partial hessian to save some unnecessary computations.
        For example, when the input is x, outputs are [u, p], and the needed derivatives
        are `[u__x, p__x, u__x__x]`, func_arch needs to evaluate the full hessian rows
        to be able to extract jacobian `p__x`. When this flag is on, func_arch will
        only output `[u__x, u__x__x]`, and `p__x` will be evaluated later by the autograd.
        If None (default), will use the GraphManager to get the global flag
        (default is True), which could be configured in the hydra config with key
        `graph.func_arch_allow_partial_hessian`.
    """

    def __init__(self, nodes: List[Node], invar: List[Key], req_names: List
        [Key], diff_nodes: List[Node]=[], func_arch: Optional[bool]=None,
        func_arch_allow_partial_hessian: Optional[bool]=None):
        super().__init__()
        graph_manager = GraphManager()
        func_arch = (func_arch if func_arch is not None else graph_manager.
            func_arch)
        func_arch_allow_partial_hessian = (func_arch_allow_partial_hessian if
            func_arch_allow_partial_hessian is not None else graph_manager.
            func_arch_allow_partial_hessian)
        self.req_names = req_names
        self.computable_names = set(_computable_names(nodes, invar))
        req_names_no_diff = [Key(x.name) for x in req_names]
        if not set(req_names_no_diff).issubset(self.computable_names):
            _print_graph_unroll_error(nodes, invar, req_names)
            raise RuntimeError('Failed Unrolling Graph')
        nodes = copy(nodes)
        necessary_nodes = []
        needed_names = [Key(x.name, derivatives=x.derivatives) for x in
            req_names] + [Key(x.name) for x in req_names]
        while True:
            finished = True
            for i, node in enumerate(nodes):
                if not set(node.outputs).isdisjoint(set(needed_names)):
                    needed_names += node.inputs + [Key(x.name, derivatives=
                        x.derivatives) for x in node.derivatives] + [Key(x.
                        name) for x in node.derivatives]
                    necessary_nodes.append(node)
                    nodes.pop(i)
                    finished = False
            if finished:
                break
        needed_names = set(needed_names)
        if func_arch:
            for i, node in enumerate(necessary_nodes):
                if isinstance(node.evaluate, Arch):
                    if node.evaluate.supports_func_arch:
                        computable_derivatives = (node.evaluate.
                            _find_computable_deriv_with_func_arch(
                            needed_names, func_arch_allow_partial_hessian))
                        if len(computable_derivatives):
                            node_name = necessary_nodes[i].name
                            necessary_nodes[i] = FuncArch(node.evaluate,
                                computable_derivatives).make_node(node_name)
                            logger.info(
                                f'{node_name} has been converted to a FuncArch node.'
                                )
                    else:
                        logger.warning(
                            f'Arch {type(node.evaluate)} currently does not support FuncArch'
                            )
        self.node_evaluation_order = []
        outvar = copy(invar)
        while True:
            while True:
                finished = True
                for i, node in enumerate(necessary_nodes):
                    if set(node.inputs + node.derivatives).issubset(set(outvar)
                        ):
                        self.node_evaluation_order.append(node)
                        outvar += node.outputs
                        necessary_nodes.pop(i)
                        finished = False
                if finished:
                    break
            needed_derivatives = []
            for node in necessary_nodes:
                needed_derivatives += node.derivatives
            needed_derivatives += [x for x in req_names if x.derivatives]
            needed_derivatives = [diff for diff in needed_derivatives if 
                diff not in outvar]
            if len(needed_derivatives) > 0:
                try_auto_diff = True
                for dn in diff_nodes:
                    if not set(dn.outputs).isdisjoint(set(needed_derivatives)
                        ) and set(dn.inputs).issubset(set(outvar)):
                        self.node_evaluation_order.append(dn)
                        outvar += dn.outputs
                        try_auto_diff = False
                if try_auto_diff:
                    dnode = Derivative.make_node(outvar, needed_derivatives,
                        jit=JitManager().enabled and JitManager().
                        autograd_nodes)
                    self.node_evaluation_order.append(dnode)
                    outvar += dnode.outputs
            if set(req_names).issubset(set(outvar)):
                break
        self.evaluation_order = paddle.nn.LayerList(sublayers=[n.evaluate for
            n in self.node_evaluation_order])
        self.node_names: List[str] = [n.name for n in self.
            node_evaluation_order]
        self.optimizer_list = paddle.nn.LayerList(sublayers=[n.evaluate for
            n in self.node_evaluation_order if n.optimize])
        if graph_manager.debug:
            print(self)

    def forward(self, invar: Dict[str, paddle.Tensor]) ->Dict[str, paddle.
        Tensor]:
        outvar = invar
        for i, e in enumerate(self.evaluation_order):
            paddle.framework.core.nvprof_nvtx_push(self.node_names[i])
            outvar.update(e(outvar))
            paddle.framework.core.nvprof_nvtx_pop()
        outvar = {key: value for key, value in outvar.items() if Key(key) in
            self.req_names}
        return outvar

    def __str__(self):
        repr = '=' * 100 + '\n'
        for node in self.node_evaluation_order:
            repr += '-' * 50 + '\n'
            repr += str(node) + '\n'
        return repr


def _print_graph_unroll_error(nodes, invar, req_names):
    print('####################################')
    print('could not unroll graph!')
    print(
        'This is probably because you are asking to compute a value that is not an output of any node'
        )
    print('####################################')
    print('invar: ' + str(list(invar)))
    print('requested var: ' + str(req_names))
    print('computable var: ' + str(_computable_names(nodes, invar)))
    print('####################################')
    print('Nodes in graph: ')
    for node in nodes:
        print(node)
    print('####################################')


def _computable_names(nodes, invar):
    nodes = copy(nodes)
    computable_names = copy(invar)
    while True:
        finished = True
        for i, node in enumerate(nodes):
            if set(node.inputs).issubset(set(computable_names)):
                computable_names += node.outputs
                nodes.pop(i)
                finished = False
        if finished:
            return computable_names
