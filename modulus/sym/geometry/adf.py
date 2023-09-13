import paddle
import numpy as np
from typing import List, Tuple, Dict


class ADF(paddle.nn.Layer):
    """
    Used for hard imposition of boundary conditions.
    Currently supports 2d geometries and Dirichlet boundary conditions.
    Contributors: M. A. Nabian, R. Gladstone, H. Meidani, N. Sukumar, A. Srivastava
    Reference: "Sukumar, N. and Srivastava, A., 2021.
        Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.
        Computer Methods in Applied Mechanics and Engineering, p.114333."
    """

    def __init__(self):
        super().__init__()
        self.mu: float = 2.0
        self.m: float = 2.0
        self.eps: float = 1e-08

    def forward(self, invar):
        raise RuntimeError(
            'No forward method was defined for ADF or its child class')

    @staticmethod
    def r_equivalence(omegas: List[paddle.Tensor], m: float=2.0
        ) ->paddle.Tensor:
        """
        Computes the R-equivalence of a collection of approximate distance functions

        Parameters
        ----------
        omegas : List[torch.Tensor]
          List of ADFs used to compute the R-equivalence.
        m: float
          Normalization order

        Returns
        -------
        omega_E : torch.Tensor
          R-equivalence distance

        """
        omega_E = paddle.zeros_like(x=omegas[0])
        for omega in omegas:
            omega_E += 1.0 / omega ** m
        omega_E = 1.0 / omega_E ** (1.0 / m)
        return omega_E

    @staticmethod
    def transfinite_interpolation(bases: List[paddle.Tensor], indx: int,
        eps: float=1e-08) ->paddle.Tensor:
        """
        Performs transfinite interpolation of the boundary conditions

        Parameters
        ----------
        bases: List[torch.Tensor]
          List of ADFs used for the transfinite interpolation.
        indx: int
          index of the interpolation basis
        eps: float
          Small value to avoid division by zero

        Returns
        -------
        w : torch.Tensor
          Interpolation basis corresponding to the input index
        """
        bases_reduced = [bases[i] for i in range(len(bases)) if i != indx]
        numerator = paddle.prod(x=paddle.stack(x=bases_reduced), axis=0)
        denominator = 0.0
        for j in range(len(bases)):
            denom_term = [bases[i] for i in range(len(bases)) if i != j]
            denominator += paddle.prod(x=paddle.stack(x=denom_term), axis=0)
        w = paddle.divide(x=numerator, y=paddle.to_tensor(denominator + eps))
        return w

    @staticmethod
    def infinite_line_adf(points: Tuple[paddle.Tensor], point_1: Tuple[
        float], point_2: Tuple[float]) ->paddle.Tensor:
        """
        Computes the pointwise approximate distance for an infinite line

        Parameters
        ----------
        points: Tuple[torch.Tensor]
          ADF will be computed on these points
        point_1: Tuple[float]
          One of the two points that form the infinite line
        point_2: Tuple[float]
          One of the two points that form the infinite line

        Returns
        -------
        omega : torch.Tensor
          pointwise approximate distance
        """
        L = ADF._distance(point_1, point_2)
        omega = ((points[0] - point_1[0]) * (point_2[1] - point_1[1]) - (
            points[1] - point_1[1]) * (point_2[0] - point_1[0])) / L
        return omega

    @staticmethod
    def line_segment_adf(points: Tuple[paddle.Tensor], point_1: Tuple[float
        ], point_2: Tuple[float]) ->paddle.Tensor:
        """
        Computes the pointwise approximate distance for a line segment

        Parameters
        ----------
        points: Tuple[torch.Tensor]
          ADF will be computed on these points
        point_1: Tuple[float]
          Point on one end of the line segment
        point_2: Tuple[float]
          Point on the other ned of the line segment

        Returns
        -------
        omega : torch.Tensor
          pointwise approximate distance
        """
        L = ADF._distance(point_1, point_2)
        center = ADF._center(point_1, point_2)
        f = ADF.infinite_line_adf(points, point_1, point_2)
        t = ADF.circle_adf(points, L / 2, center)
        phi = paddle.sqrt(x=t ** 2 + f ** 4)
        omega = paddle.sqrt(x=f ** 2 + ((phi - t) / 2) ** 2)
        return omega

    @staticmethod
    def circle_adf(points: Tuple[paddle.Tensor], radius: float, center:
        Tuple[float]) ->paddle.Tensor:
        """
        Computes the pointwise approximate distance for a circle

        Parameters
        ----------
        points: Tuple[torch.Tensor]
          ADF will be computed on these points
        radius: float
          Radius of the circle
        center: Tuple[float]
          Center of the circle

        Returns
        -------
        omega : torch.Tensor
          pointwise approximate distance
        """
        omega = (radius ** 2 - ((points[0] - center[0]) ** 2 + (points[1] -
            center[1]) ** 2)) / (2 * radius)
        return omega

    @staticmethod
    def trimmed_circle_adf(points: Tuple[paddle.Tensor], point_1: Tuple[
        float], point_2: Tuple[float], sign: int, radius: float, center: float
        ) ->paddle.Tensor:
        """
        Computes the pointwise approximate distance of a trimmed circle

        Parameters
        ----------
        points: Tuple[torch.Tensor]
          ADF will be computed on these points
        point_1: Tuple[float]
          One of the two points that form the trimming infinite line
        point_2: Tuple[float]
          One of the two points that form the trimming infinite line
        sign: int
          Specifies the trimming side
        radius: float
          Radius of the circle
        center: Tuple[float]
          Center of the circle

        Returns
        -------
        omega : torch.Tensor
          pointwise approximate distance
        """
        assert sign != 0, 'sign should be non-negative'
        f = ADF.circle_adf(points, radius, center)
        t = np.sign(sign) * ADF.infinite_line_adf(points, point_1, point_2)
        phi = paddle.sqrt(x=t ** 2 + f ** 4)
        omega = paddle.sqrt(x=f ** 2 + ((phi - t) / 2) ** 2)
        return omega

    @staticmethod
    def _distance(point_1: Tuple[float], point_2: Tuple[float]
        ) ->paddle.Tensor:
        """
        Computes the distance between two points

        point_1: Tuple[float]
          The first point
        point_2: Tuple[float]
          The second point

        Returns
        -------
        distance : torch.Tensor
            distance between the two points
        """
        distance = np.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] -
            point_1[1]) ** 2)
        return distance

    @staticmethod
    def _center(point_1: Tuple[float], point_2: Tuple[float]) ->Tuple[float]:
        """
        Computes the center of the two points

        point_1: Tuple[float]
          The first point
        point_2: Tuple[float]
          The second point

        Returns
        -------
        center : torch.Tensor
            Center the two points
        """
        center = (point_1[0] + point_2[0]) / 2, (point_1[1] + point_2[1]) / 2
        return center
