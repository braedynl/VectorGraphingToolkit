from __future__ import annotations

import math
from numbers import Real
from typing import Callable, List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from scipy.integrate import odeint
from sympy.utilities.lambdify import lambdastr, lambdify

NOTATION_STYLES = {
    'angled': '$\\vec{{{}}} = \\langle {}, {} \\rangle$',
    'parentheses': '$\\vec{{{}}} = ({}, {})$',
    'brackets': '$\\vec{{{}}} = [{}, {}]$',
    'bmatrix': '$\\vec{{{}}} = \\begin{{bmatrix}} {} \\\\ {} \\end{{bmatrix}}$',
    'pmatrix': '$\\vec{{{}}} = \\begin{{pmatrix}} {} \\\\ {} \\end{{pmatrix}}$',
    'basis': '$\\vec{{{}}} = ({}) \\hat{{i}} + ({}) \\hat{{j}}$'
}

ArrayLike = TypeVar('ArrayLike')
Pair = TypeVar('Pair')
Sympifyable = TypeVar('Sympifyable')

class Vector:
    """
    Represents a euclidean vector in R2 space.

    Parameters
    ----------
    u
        Horizontal scalar of the vector.
    v
        Vertical scalar of the vector.

    Raises
    ------
    TypeError
        If either scalars are not a real numeric type.
    """

    __slots__ = ['__scalars']

    def __init__(self, u: Real = 0, v: Real = 0):
        if not isinstance(u, Real) or not isinstance(v, Real):
            raise TypeError('u or v argument is not a real numeric type')
        self.__scalars = np.array([u, v], dtype=np.double)

    @property
    def u(self) -> float:
        """Horizontal scalar of the vector."""
        return self.__scalars[0]

    @u.setter
    def u(self, u: Real) -> None:
        self.__scalars[0] = u

    @property
    def v(self) -> float:
        """Vertical scalar of the vector."""
        return self.__scalars[1]

    @v.setter
    def v(self, v: Real) -> None:
        self.__scalars[1] = v

    @property
    def mag(self) -> float:
        """Magnitude of the vector."""
        return np.linalg.norm(self.__scalars)

    def __pos__(self) -> Vector:
        return Vector(*(+self.__scalars))

    def __neg__(self) -> Vector:
        return Vector(*(-self.__scalars))

    def __inv__(self) -> Vector:
        m = self.mag
        if math.isclose(m, 0):
            raise ZeroDivisionError(
                'cannot convert zero vector to unit vector')
        return Vector(*(self.__scalars / m))

    def __add__(self, other: Vector) -> Vector:
        return Vector(*(self.__scalars + other.__scalars))

    def __iadd__(self, other: Vector) -> self:
        self.__scalars += other.__scalars
        return self

    def __sub__(self, other: Vector) -> Vector:
        return Vector(*(self.__scalars - other.__scalars))

    def __isub__(self, other: Vector) -> self:
        self.__scalars -= other.__scalars
        return self

    def __mul__(self, a: Real) -> Vector:
        return Vector(*(self.__scalars * a))

    def __rmul__(self, a: Real) -> Vector:
        return self.__mul__(a)

    def __imul__(self, a: Real) -> self:
        self.__scalars *= a
        return self

    def __truediv__(self, a: Real) -> Vector:
        return Vector(*(self.__scalars / a))

    def __itruediv__(self, a: Real) -> self:
        self.__scalars /= a
        return self

    def __floordiv__(self, a: Real) -> Vector:
        return Vector(*(self.__scalars // a))

    def __ifloordiv__(self, a: Real) -> self:
        self.__scalars //= a
        return self

    def __and__(self, other: Vector) -> float:
        return self.u * other.v - other.u * self.v

    def __xor__(self, other: Vector) -> float:
        return np.dot(self.__scalars, other.__scalars)

    def __eq__(self, other: Vector) -> bool:
        if not isinstance(other, Vector):
            return False
        return np.allclose(self.__scalars, other.__scalars)

    def __ne__(self, other: Vector) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return 'Vector({:.2f}, {:.2f})'.format(*self.__scalars)

    def __repr__(self) -> str:
        return 'Vector({}, {})'.format(*self.__scalars)

    def __hash__(self) -> int:
        return hash((self.u, self.v))

    def unit(self) -> Vector:
        """
        Conversion to unit vector.

        Raises
        ------
        ZeroDivisionError
            If vector is a zero vector.
        """
        return self.__inv__()

    def dot(self, other: Vector) -> float:
        """
        Dot product of two vectors.

        Parameters
        ----------
        other
            Second operand vector.

        Returns
        -------
        float
            Dot product of operand vectors.
        """
        return self.__xor__(other)

    def cross(self, other: Vector) -> float:
        """
        Cross product of two vectors.

        Parameters
        ----------
        other
            Second operand vector.

        Returns
        -------
        float
            Cross product of operand vectors.
        """
        return self.__and__(other)

    def angle(self, other: Vector, degrees: bool = False) -> float:
        """
        Angle between two vectors in radians.

        Parameters
        ----------
        other
            Second operand vector.
        degrees
            Option to return angle in degrees.

        Returns
        -------
        float
            Angle between operand vectors.
        """
        theta = math.acos(self.__xor__(other) / (self.mag * other.mag))
        if not degrees:
            return theta
        return theta * (180 / math.pi)

    def is_opposite(self, other: Vector) -> bool:
        """
        Tests if two vectors are opposite to each other.

        Two vectors are opposite if they have the same magnitude
        but opposite direction.

        Parameters
        ----------
        other
            Second operand vector.

        Returns
        -------
        bool
            True if opposite, else False.
        """
        return math.isclose(self.u, -other.u) and math.isclose(self.v, -other.v)

    def is_parallel(self, other: Vector) -> bool:
        """
        Tests if two vectors are parallel to each other.

        Two vectors are parallel if they have the same direction
        but not necessarily the same magnitude.

        Parameters
        ----------
        other
            Second operand vector.

        Returns
        -------
        bool
            True if parallel, else False.
        """
        return math.isclose(self.angle(other), 0)

    def is_antiparallel(self, other: Vector) -> bool:
        """
        Tests if two vectors are antiparallel to each other.

        Two vectors are antiparallel if they have opposite direction
        but not necessarily the same magnitude.

        Parameters
        ----------
        other
            Second operand vector.

        Returns
        -------
        bool
            True if antiparallel, else False.
        """
        return math.isclose(self.angle(other), math.pi)

    def is_perpendicular(self, other: Vector) -> bool:
        """
        Tests if two vectors are perpendicular to each other.

        Parameters
        ----------
        other
            Second operand vector.

        Returns
        -------
        bool
            True if perpendicular, else False.
        """
        return math.isclose(self.angle(other), math.pi / 2)

    def to_numpy(self) -> np.ndarray:
        """Returns the vector components as a numpy array."""
        return self.__scalars

    def to_list(self) -> List[float]:
        """Returns the vector components as a Python list."""
        return list(self.__scalars)

    def to_latex(self, name: str = 'v', notation: str = 'basis', ndigits: int = 2) -> str:
        """
        Returns the vector represented as a LaTeX string.

        Parameters
        ----------
        name
            Name of the vector.
        notation
            Notation format of the string.
        ndigits
            nth decimal place to round components to.

        Returns
        -------
        str
            The vector represented as a LaTeX string.

        Notes
        -----
        Accepted notation arguments are "angled", "parentheses", "brackets", "bmatrix",
        "pmatrix", and "basis".

        Raises
        ------
        ValueError
            If notation is not recognized.
        """
        if notation not in NOTATION_STYLES:
            raise ValueError('invalid notation argument "{}"'.format(notation))
        return NOTATION_STYLES[notation].format(
            name, round(self.u, ndigits), round(self.v, ndigits))

    def plot(self, ax: Axes, x: Real = 0, y: Real = 0, **kwargs) -> Quiver:
        """
        Plots the vector on a matplotlib Axes.

        Parameters
        ----------
        ax
            Axes to plot vector.
        x
            x coordinate of the vector.
        y
            y coordinate of the vector.
        **kwargs
            Additional keyword arguments passed to ~Axes.quiver()

        Returns
        -------
        matplotlib.quiver.Quiver
            The created Quiver instance.
        """
        return ax.quiver(x, y, self.u, self.v, **kwargs)


class VectorField:
    """
    Represents a static vector field in R2 space.

    Parameters
    ----------
    u
        Horizontal scalar function of x and y.
    v
        Vertical scalar function of x and y.

    Notes
    -----
        The Sympifyable type is any expression accepted by
        the sympy.sympfiy() function. Read more here:

        https://docs.sympy.org/latest/modules/core.html#id1
    """

    __slots__ = ['__u_sym', '__v_sym', '__u_np', '__v_np']
    __x, __y = sym.symbols('x y')

    def __init__(self, u: Sympifyable, v: Sympifyable):
        self.__u_sym = sym.sympify(u)
        self.__v_sym = sym.sympify(v)
        self.__u_np = lambdify(
            (self.__x, self.__y), self.__u_sym, 'numpy')
        self.__v_np = lambdify(
            (self.__x, self.__y), self.__v_sym, 'numpy')

    @property
    def u(self) -> Callable[[Real, Real], Real]:
        """Horizontal scalar function of x and y."""
        return self.__u_np

    @u.setter
    def u(self, u: Sympifyable) -> None:
        self.__u_sym = sym.sympify(u)
        self.__u_np = lambdify(
            (self.__x, self.__y), self.__u_sym, 'numpy')

    @property
    def v(self) -> Callable[[Real, Real], Real]:
        """Vertical scalar function of x and y."""
        return self.__v_np

    @v.setter
    def v(self, v: Sympifyable) -> None:
        self.__v_sym = sym.sympify(v)
        self.__v_np = lambdify(
            (self.__x, self.__y), self.__v_sym, 'numpy')

    @property
    def mag(self) -> Callable[[Real, Real], Real]:
        """Magnitude function of the vector field."""
        return lambdify(
            (self.__x, self.__y),
            sym.sqrt(self.__u_sym**2 +
                     self.__v_sym**2),
            'numpy')

    @property
    def div(self) -> Callable[[Real, Real], Real]:
        """Divergence function of the vector field."""
        return lambdify(
            (self.__x, self.__y),
            sym.diff(self.__u_sym, self.__x) +
            sym.diff(self.__v_sym, self.__y),
            'numpy')

    @property
    def curl(self) -> Callable[[Real, Real], Real]:
        """Curl function of the vector field."""
        return lambdify(
            (self.__x, self.__y),
            sym.diff(self.__v_sym, self.__x) -
            sym.diff(self.__u_sym, self.__y),
            'numpy')

    @classmethod
    def from_grad(cls, f: Sympifyable) -> VectorField:
        """
        Create a VectorField from a gradient function.

        Parameters
        ----------
        f
            A gradient function of x and y.

        Returns
        -------
        VectorField
            The created vector field instance.
        """
        x, y = sym.symbols('x y')
        return VectorField(sym.diff(f, x), sym.diff(f, y))

    @classmethod
    def from_stream(cls, psi: Sympifyable) -> VectorField:
        """
        Create a VectorField from a stream function.

        Parameters
        ----------
        psi
            A stream function of x and y.

        Returns
        -------
        VectorField
            The created vector field instance.
        """
        x, y = sym.symbols('x y')
        return VectorField(sym.diff(psi, y), -sym.diff(psi, x))

    def __str__(self) -> str:
        return 'VectorField({}, {})'.format(self.__u_sym, self.__v_sym)

    def __repr__(self) -> str:
        return 'VectorField({}, {})'.format(
            lambdastr((self.__x, self.__y), self.__u_sym),
            lambdastr((self.__x, self.__y), self.__v_sym))

    def is_solenoidal(self) -> bool:
        """Tests if the vector field is solenoidal."""
        return (sym.diff(self.__u_sym, self.__x) +
                sym.diff(self.__v_sym, self.__y) == 0)

    def is_conservative(self) -> bool:
        """Tests if the vector field is conservative."""
        return (sym.diff(self.__v_sym, self.__x) -
                sym.diff(self.__u_sym, self.__y) == 0)

    def to_latex(self, name: str = 'F', notation: str = 'basis') -> str:
        """
        Returns the vector field represented as a LaTeX string.

        Parameters
        ----------
        name
            Name of the vector field.
        notation
            Notation format of the string.

        Returns
        -------
        str
            The vector field represented as a LaTeX string.

        Notes
        -----
            Accepted notation arguments are "angled", "parentheses", "brackets", 
            "bmatrix", "pmatrix", and "basis".

        Raises
        ------
        ValueError
            If notation is not recognized.
        """
        if notation not in NOTATION_STYLES:
            raise ValueError('invalid notation argument "{}"'.format(notation))
        return NOTATION_STYLES[notation].format(
            name, sym.latex(self.__u_sym), sym.latex(self.__v_sym))

    def plot(self, ax: Axes, xn: int = 25, yn: int = 25, **kwargs) -> Quiver:
        """
        Plots the vector field on a matplotlib Axes.

        Parameters
        ----------
        ax
            Axes to plot vector field.
        xn
            Number of vectors to plot along the x axis for every y.
        yn
            Number of vectors to plot along the y axis for every x.
        **kwargs
            Additional keyword arguments passed to ~Axes.quiver()

        Returns
        -------
        matplotlib.quiver.Quiver
            The created Quiver instance.
        """
        x, y = np.meshgrid(
            np.linspace(*ax.get_xlim(), xn),
            np.linspace(*ax.get_ylim(), yn)
        )

        u, v, c = self.u(x, y), self.v(x, y), self.mag(x, y)

        return ax.quiver(x, y, u, v, c, **kwargs)

    def animate(self, fig: Figure, ax: Axes, init_pts: ArrayLike[Pair[Real]] = None, dt: float = 0.01, interval: int = 1, frames: int = 150, blit: bool = False) -> FuncAnimation:
        """
        Animates particles modeled by the vector field on a matplotlib Axes.

        Parameters
        ----------
        fig
            Figure object required for re-drawing.
        ax
            Axes to model animation.
        init_pts
            An array of coordinate pairs that set the initial particle positions.
            If None, 50 randomly-placed particles will be initialized.
        dt
            Time step used when calculating particle displacements.
        interval
            Amount of time between each frame of the animation in milliseconds.
        frames
            Amount of frames to run the animation before looping.
        blit
            Option to blit the animation.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The created FuncAnimation instance.

        Notes
        -----
            A reference to the FuncAnimation instance must be kept, otherwise
            garbage collection will stop the animation.
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ln, = ax.plot([], [], 'o', color='k', alpha=0.7)
        if init_pts is None:
            init_pts = np.array(
                (np.random.uniform(*xlim, 50), np.random.uniform(*ylim, 50))).T
        pts = np.copy(init_pts)

        def remove(pts: np.ndarray) -> np.ndarray:
            if len(pts) == 0:
                return np.empty((0, 2))
            out_x = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
            out_y = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
            return pts[~(out_x | out_y)]

        def displace(pts: np.ndarray) -> np.ndarray:
            return np.array([odeint(lambda pt, _: [self.u(*pt), self.v(*pt)], pt, [0, dt])[-1] for pt in pts])

        def update(frame: int) -> List[Line2D]:
            nonlocal pts
            if frame == 0:
                pts = np.copy(init_pts)
            pts = remove(displace(pts))
            ln.set_data(*pts.T)
            return ln,

        return FuncAnimation(fig, update, interval=interval, frames=frames, blit=blit)
