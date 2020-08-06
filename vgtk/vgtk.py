'''
VectorGraphingToolkit
=====================

A simplistic vector/vector field visualization tool built on top of matplotlib.

Recommended import:

    from vgtk import Vector, VectorField

Version: 0.1.0-beta

GitHub/Docs: https://github.com/braedynl/VectorGraphingToolkit/

Author: Braedyn Lettinga

Collaborators: Ashu Acharya

Credits: Heiko Hergert, Ph.D., Tony S. Yu, Ph.D.

This project is licensed under the MIT License - see the page below for more details:

https://github.com/braedynl/VectorGraphingToolkit/blob/master/LICENSE
'''

from __future__ import annotations

from typing import Callable, Iterable, Union

import matplotlib
import numpy as np
import sympy as sym
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sympy.utilities.lambdify import lambdastr, lambdify

from vgtk.handlers import (_ParticleSimulationHandler, _VectorEventHandler,
                           _VectorFieldEventHandler)


class Vector(object):
    '''
    This class simulates a two-dimensional vector in R^2 space.

    Parameters
    ----------
        u : Base u scalar.
        v : Base v scalar.
        name : Name of the vector. Used for plot interactivity and string methods.
    
    Notes
    -----
        `u`, `v`, and `name` are all attributes, and have corresponding setters.
        There is also a `scalars` getter/setter, which returns a `numpy.ndarray`
        of the two scalars. It can be set with any iterable of shape (2, ).

        Scalars are kept un-rounded internally. You can use `repr()` to see the 
        scalars un-rounded. `__str__()` will always round to the second decimal
        place. 

        Conversion from an array is not inherently supported by the constructor. 
        Use the `*` operator if you want to unpack two scalar values from an array. 
    '''

    def __init__(self, u:float, v:float, name:str='v'):
        self.__scalar_handle(u, v)
        self.__scalars = np.array((u, v), dtype=np.dtype(float))
        self.name = str(name)

        self.__handler = None
    
    def __add__(self, other:Vector) -> Vector:
        '''Adds two vectors.''' 
        return Vector(*(self.__scalars + other.__scalars), '({}+{})'.format(self.name, other.name))

    def __iadd__(self, other:Vector) -> self:
        '''Adds two vectors in-place.'''
        self.__scalars += other.__scalars
        return self
    
    def __sub__(self, other:Vector) -> Vector:
        '''Subtracts two vectors.'''
        return Vector(*(self.__scalars - other.__scalars), '({}-{})'.format(self.name, other.name))

    def __isub__(self, other:Vector) -> self:
        '''Subtracts two vectors in-place.'''
        self.__scalars -= other.__scalars
        return self

    def __mul__(self, a:float) -> Vector:
        '''Multiplies vector by a scalar value.'''
        return Vector(*(self.__scalars * a), '({}*{})'.format(a, self.name))

    def __rmul__(self, a:float) -> Vector:
        '''Multiplies vector by a scalar value.'''
        return self.__mul__(a)

    def __imul__(self, a:float) -> self:
        '''Multiples vector by a scalar value in-place.'''
        self.__scalars *= a 
        return self

    def __truediv__(self, a:float) -> Vector:
        '''Divides vector by a scalar value.'''
        return Vector(*(self.__scalars / a), '({}/{})'.format(self.name, a))
    
    def __idiv__(self, a:float) -> self:
        '''Divides vector by a scalar value in-place.'''
        self.__scalars /= a 
        return self
    
    def __invert__(self):
        '''Converts Vector into its equivalent unit vector form. Returns self.'''
        return self.unit()
    
    def __xor__(self, other:Vector) -> float:
        '''
        Calculates the dot product between two Vector instances.

        Parameters
        ----------
            other : Another Vector instance.
        
        Returns
        -------
            float : Resultant dot product.
        '''
        return self.dot(other)

    def __eq__(self, other:Vector) -> bool:
        '''Tests if two vectors are equivalent.'''
        return all(self.__scalars == other.__scalars)
    
    def __ne__(self, other:Vector) -> bool:
        '''Tests if two vectors are not equivalent.'''
        return not self.__eq__(other)

    def __hash__(self) -> int:
        '''Returns id of self.'''
        return id(self)

    def __getitem__(self, index:int) -> float:
        '''Obtains the u/v scalar from the array of scalars.'''
        return self.__scalars[index]
    
    def __setitem__(self, index:int, value:float) -> None:
        '''Sets the u/v scalar from the array of scalars.'''
        self.__scalars[index] = value

    def __str__(self) -> str:
        '''Returns a string of the vector in angle-bracket notation.'''
        return '{} = <{:.2f}, {:.2f}>'.format(self.name, *self.__scalars)

    def __repr__(self) -> str:
        '''Returns a string of the vector in angle-bracket notation, un-rounded.'''
        return '{} = <{}, {}>'.format(self.name, *self.__scalars)

    @property
    def u(self) -> float:
        '''Gets u scalar.'''
        return self.__scalars[0]
    
    @u.setter
    def u(self, u:float) -> None:
        '''Sets u scalar.'''
        self.__scalars[0] = float(u)
    
    @property
    def v(self) -> float:
        '''Gets v scalar.'''
        return self.__scalars[1]

    @v.setter
    def v(self, v:float) -> None:
        '''Sets v scalar.'''
        self.__scalars[1] = float(v)

    @property
    def scalars(self) -> np.ndarray:
        '''Gets array of scalars.'''
        return self.__scalars
    
    @scalars.setter
    def scalars(self, arr:Iterable[float]) -> None:
        '''Sets array of scalars.'''
        if len(arr) != 2: raise ValueError('dimension mismatch')
        self.__scalar_handle(arr[0], arr[1])    
        self.__scalars = np.array(arr, dtype=np.dtype(float))

    @property
    def mag(self) -> float:
        '''Magnitude of the vector.'''
        return np.linalg.norm(self.__scalars)

    def dot(self, other:Vector) -> float:
        '''
        Calculates the dot product between two Vector instances.

        Parameters
        ----------
            other : Another Vector instance.
        
        Returns
        -------
            float : Resultant dot product.
        '''
        return np.dot(self.__scalars, other.__scalars)

    def angle(self, other:Vector, degrees:bool=False) -> float:
        '''
        Measures the radian angle between two Vector instances.

        Parameters
        ----------
            other : Another Vector instance.
            degrees : Returns angle measured in degrees if True.
        
        Returns
        -------
            float : Resultant angle between the two vectors.
        '''
        t = np.arccos( self.dot(other) / (self.mag * other.mag) )
        return t if not degrees else t * (180/np.pi)

    def unit(self) -> self:
        '''Converts Vector into its equivalent unit vector form. Returns self.'''
        self.__scalars = self.__scalars / self.mag
        return self

    def plot(self, fig:Figure, ax:Axes, x:float=0, y:float=0, color:str='skyblue', trace_scalars:bool=False, 
             interactive:bool=False, **kwargs) -> matplotlib.quiver.Quiver:
        '''
        Plots the vector on a given matplotlib Axes.

        Parameters
        ----------
            fig : A matplotlib.figure.Figure instance.
            ax : A matplotlib.axes.Axes instance.
            x : Starting x-coordinate of the vector.
            y : Starting y-coordinate of the vector.
            color : Color of the vector. Argument passed to ~Axes.quiver().
            trace_scalars : Option to plot dashed lines that represent the scalar values of the vector.
                - The u scalar is represented as blue (C0), the v scalar is represented as orange (C1)
            interactive : Option to make the vector plot interactable.
                - A point is plotted at the tip of the vector that allows the user to warp, shift and see
                  various details about the vector.
                - Holding left-click will drag the vector's tip to the mouse pointer's location, while the
                  base of the vector stays fixed.
                - Holding right-click will drag the entire vector to the mouse pointer's location, while the
                  magnitude and direction stays fixed.
                - Holding middle-click will show the vector's details without warping or shifting.
            **kwargs : Additional arguments passed to ~Axes.quiver().
        
        Returns
        -------
            matplotlib.quiver.Quiver : The created Quiver instance.
        
        References
        ----------
            color options: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
            kwargs options: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html
        
        Notes
        -----
            'scale_units', 'angles' and 'scale' are overwritten in `kwargs` to prevent warping.
        '''
        kwargs['color'] = color
        kwargs['scale_units'] = 'xy'
        kwargs['angles'] = 'xy'
        kwargs['scale'] = 1

        self.__handler = _VectorEventHandler(fig, ax, x, y, *self.__scalars, self.name, trace_scalars, interactive, **kwargs)
        
        return self.__handler.quiver

    def get_latex_str(self, notation:Union['angled', 'parentheses', 'unit']='angled') -> str:
        '''
        Returns a string of the Vector instance in LaTeX format.

        Parameters
        ----------
            notation : Changes the notation style of the string.    
                - 'angled' : Ordered set notation, angle-bracket variant.
                - 'parentheses' : Ordered set notation, parentheses variant.
                - 'unit' : Unit vector notation. 
        
        Returns
        -------
            str : Vector instance in LaTeX format.
        
        References
        ----------
            https://en.wikipedia.org/wiki/Vector_notation
        '''
        if notation == 'angled':
            return '$\\vec{{{}}} = <{}, {}>$'.format(self.name, self.__scalars[0], self.__scalars[1])
        elif notation == 'parentheses':
            return '$\\vec{{{}}} = ({}, {})$'.format(self.name, self.__scalars[0], self.__scalars[1])
        elif notation == 'unit':
            return '$\\vec{{{}}} = ({})\\hat{{i}} + ({})\\hat{{j}}$'.format(self.name, self.__scalars[0], self.__scalars[1])
        else:
            raise ValueError('notation, "{}", not recognized -- possible options are: "angled", "parentheses", "unit"'.format(notation))

    def __scalar_handle(self, u:float, v:float) -> None:
        '''Tests if all vectors are a numeric type.'''
        if not isinstance(u, (int, float)) or not isinstance(v, (int, float)):
            raise TypeError('not all scalars are a numeric type')


class VectorField(object):
    '''
    This class simulates a two-dimensional vector field in R^2 space.

    Parameters
    ----------
        u : Base u scalar function of two variables (must be x and y).
        v : Base v scalar function of two variables (must be x and y).
        name : Name of the vector field. Used for plot interactivity and string methods.
    
    Notes
    -----
        'expr' represents a general sympy expression.

        `u` and `v` arguments are passed to `sympy.sympify()`. Expressions are ran via the 
        `exec()` function, which means that formulae must be in proper Python syntax. 
        Many common mathematical functions can be written without the use of `sympy`, like 
        `'cos()'`, `'sin()'`, `'tan()'`, etc. To express e^x as a string, use `'exp()'`.

        `u`, `v`, and `name` are all attributes, and have corresponding setters.
    '''

    def __init__(self, u:Union[str, float, 'expr'], v:Union[str, float, 'expr'], name:str='F'):

        self.__usym = sym.sympify(u)
        self.__vsym = sym.sympify(v)

        x, y = sym.symbols('x y')
        self.__unp = lambdify((x, y), self.__usym, 'numpy' )
        self.__vnp = lambdify((x, y), self.__vsym, 'numpy' )

        self.name = str(name)

        self.__handler = None
        self.__particle_handler = None

    def __str__(self) -> str:
        '''Returns a string of the vector field in angle-bracket notation.'''
        return '{} = <{}, {}>'.format(self.name, self.__usym, self.__vsym)
    
    def __repr__(self) -> str:
        '''Returns a string of the vector field in angle-bracket notation with lambda parameters.'''
        x, y = sym.symbols('x y')
        return '{} = <{}, {}>'.format(self.name, lambdastr((x, y), self.__usym), lambdastr((x, y), self.__vsym))

    @property
    def u(self) -> Callable[[float, float], float]:
        '''Gets u scalar function.'''
        return self.__unp
    
    @u.setter
    def u(self, u:Union[str, float, 'expr']) -> None:
        '''Sets u scalar function.'''
        self.__usym = sym.sympify(u)
        x, y = sym.symbols('x y')
        self.__unp = lambdify((x, y), self.__usym, 'numpy')
    
    @property
    def v(self) -> Callable[[float, float], float]:
        '''Gets v scalar function.'''
        return self.__vnp
    
    @v.setter
    def v(self, v:Union[str, float, 'expr']) -> None:
        '''Sets v scalar function.'''
        self.__vsym = sym.sympify(v)
        x, y = sym.symbols('x y')
        self.__vnp = lambdify((x, y), self.__vsym, 'numpy')

    @classmethod
    def from_grad(cls, f:Union[str, float, 'expr'], name:str='F') -> VectorField:
        '''
        Creates a VectorField from the gradient of a given function.

        Parameters
        ----------
            f : A function of two variables (must be x and y).
            name : Name of the vector field. Used for plot interactivity and string methods.
        
        Returns
        -------
            VectorField : The instantiated VectorField. 
        
        References
        ----------
            https://en.wikipedia.org/wiki/Gradient
        '''
        x, y = sym.symbols('x y')
        return VectorField(sym.diff(f, x), sym.diff(f, y), name)

    @classmethod
    def from_stream(cls, psi:Union[str, float, 'expr'], name:str='F') -> VectorField:
        '''
        Creates a VectorField from a given stream function.

        Parameters
        ----------
            psi : A stream function of two variables (must be x and y).
            name : Name of the vector field. Used to plot interactivity and string methods.
        
        Returns
        -------
            VectorField : the instantiated VectorField.
        
        References
        ----------
            https://en.wikipedia.org/wiki/Stream_function
        '''
        x, y = sym.symbols('x y')
        return VectorField(sym.diff(psi, y), -sym.diff(psi, x), name)

    @property
    def mag(self) -> Callable[[float, float], float]:
        '''The magnitude function of the vector field.'''
        x, y = sym.symbols('x y')
        return lambdify((x, y), sym.sqrt((self.__usym)**2 + (self.__vsym)**2), 'numpy')

    @property
    def div(self) -> Callable[[float, float], float]:
        '''The divergence function of the vector field.'''
        x, y = sym.symbols('x y')
        return lambdify((x, y), sym.diff(self.__usym, x) + sym.diff(self.__vsym, y), 'numpy')

    @property
    def curl(self) -> Callable[[float, float], float]:
        '''The curl function of the vector field.'''
        x, y = sym.symbols('x y')
        return lambdify((x, y), sym.diff(self.__vsym, x) - sym.diff(self.__usym, y), 'numpy')

    def is_solenoidal(self) -> bool:
        '''Tests if the vector field is solenoidal. Returns bool.'''
        x, y = sym.symbols('x y')
        return True if sym.diff(self.__usym, x) + sym.diff(self.__vsym, y) == 0 else False

    def is_conservative(self) -> bool:
        '''Tests if the vector field is conservative. Returns bool.'''
        x, y = sym.symbols('x y')
        return True if sym.diff(self.__vsym, x) - sym.diff(self.__usym, y) == 0 else False

    def plot(self, fig:Figure, ax:Axes, scale:float=1, density:int=10, cmap:Union[str, ListedColormap]='Blues', 
             normalize:bool=True, colorbar:bool=True, interactive:bool=False, **kwargs) -> matplotlib.quiver.Quiver:
        '''
        Plots the vector field on a given matplotlib Axes.

        Parameters
        ----------
            fig : A matplotlib.figure.Figure instance.
            ax : A matplotlib.axes.Axes instance.
            scale : Scalar value applied to each vector. See notes below for more details.
            density : A measure of how many vectors to plot within the field.
                - An evenly-spaced grid of density*density vectors is plotted.
                - Value must be within range [4, 100].
            cmap : A matplotlib colormap applied to the field.
                - Can pass a built-in or custom colormap.
            normalize : Option to normalize vectors. See notes below for more details.
            colorbar : Option to display a colorbar of the color mapping.
                - The values shown, both on the colorbar ticks and the colorbar's label, will vary depending
                  on the chosen scale and density.
            interactive : Option to make the vector field plot interactable.
                - The plot will detect mouse clicks, and sliders will be added below the axes.
                - Clicking and holding on a point within the axes will display an annotation describing the
                  curl, divergence, and magnitude at that point.
                - The scale and density sliders adjust the field's scale and density in realtime. 
                - The upper-bound of the scale slider is calculated using the x and y limits of the axes.
            **kwargs : Additional arguments passed to ~Axes.quiver(). 
        
        Returns
        -------
            matplotlib.quiver.Quiver : The created Quiver instance.
        
        References
        ----------
            built-in cmap options: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            kwargs options: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html
        
        Notes
        -----
            When normalize=True, all vectors will be converted into their unit vector form. The scale argument 
            is then applied after. An auto-scaling algorithm could be implemented in the future, since extremely 
            small axes will have huge vectors and extremely large axes will have small vectors without adjusting 
            the scale from its default. 

            'scale_units', 'angles' and 'scale' are overwritten in `kwargs` to prevent warping.
        '''
        if not 4 <= density <= 100: raise ValueError('density argument must be within range [4, 100]')

        kwargs['cmap'] = cmap
        kwargs['scale_units'] = 'xy'
        kwargs['angles'] = 'xy'
        kwargs['scale'] = 1

        self.__handler = _VectorFieldEventHandler(fig, ax, self.__unp, self.__vnp, self.mag, self.div, self.curl, 
                                                  self.name, scale, density, normalize, colorbar, interactive, **kwargs)

        return self.__handler.quiver

    def particles(self, fig:Figure, ax:Axes, pts:Iterable[tuple]=None, frames:int=300, dt:float=0.01, blit:bool=True,
                  fmt:str='o', color:str='k', alpha:float=0.7, **kwargs) -> matplotlib.animation.FuncAnimation:
        '''
        Animates particles on a given matplotlib Axes, where relative velocities are modeled by the field.

        Parameters
        ----------
            fig : A matplotlib.figure.Figure instance.
            ax : A matplotlib.axes.Axes instance.
            pts : An array of coordinate pairs that set the initial particle positions.
                - If `None`, 50 randomly-placed particles will be plotted.
            frames : The amount of frames to run the animation for.
            dt : The change in time between each frame.
                - I recommend keeping this value small.
            blit : Option to blit particle animation. 
                - Should be set to `False` if the `plot()` method is active with
                  interactivity enabled.
            fmt : Marker style of the particles. Argument passed to ~Axes.plot().
            color : Color of the particles. Argument passed to ~Axes.plot().
            alpha : Transparency of the particles. Argument passed to ~Axes.plot().
            **kwargs : Additional arguments passed to ~Axes.plot(). 
        
        Returns
        -------
            matplotlib.animation.FuncAnimation : The created FuncAnimation instance.
        
        References
        ----------
            color options: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
            kwargs options: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.plot.html
        
        Credits
        -------
            Heiko Hergert, Ph.D. - Helped with conceptualization.
            Tony S. Yu, Ph.D. - The only human being on the planet that has an example.

            Much of the particle animation work is based on Dr. Yu's article:
            https://tonysyu.github.io/animating-particles-in-a-flow.html
        '''

        self.__particle_handler = _ParticleSimulationHandler(fig, ax, self.__unp, self.__vnp, pts, frames, dt, blit, fmt, color, alpha, **kwargs)

        return self.__particle_handler.ani

    def get_latex_str(self, notation:Union['angled', 'parentheses', 'unit']='angled') -> str:
        '''
        Returns a string of the VectorField instance in LaTeX format.

        Parameters
        ----------
            notation : Changes the notation style of the string.    
                - 'angled' : Ordered set notation, angle-bracket variant.
                - 'parentheses' : Ordered set notation, parentheses variant.
                - 'unit' : Unit vector notation. 
        
        Returns
        -------
            str : VectorField instance in LaTeX format.
        
        References
        ----------
            https://en.wikipedia.org/wiki/Vector_notation
        '''
        if notation == 'angled':
            return '$\\vec{{{}}} = <{}, {}>$'.format(self.name, sym.latex(self.__usym), sym.latex(self.__vsym))
        elif notation == 'parentheses':
            return '$\\vec{{{}}} = ({}, {})$'.format(self.name, sym.latex(self.__usym), sym.latex(self.__vsym))
        elif notation == 'unit':
            return '$\\vec{{{}}} = ({})\\hat{{i}} + ({})\\hat{{j}}$'.format(self.name, sym.latex(self.__usym), sym.latex(self.__vsym))
        else:
            raise ValueError('notation, "{}", not recognized -- possible options are: "angled", "parentheses", "unit"'.format(notation))
