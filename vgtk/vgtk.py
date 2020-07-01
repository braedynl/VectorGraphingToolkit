'''
VectorGraphingToolkit
=====================

A simplistic vector/vector field visualization tool built on top of matplotlib.

Recommended import:

  from vgtk import Vector, VectorField

Version: 1.0.0

GitHub/Docs: https://github.com/braedynl/VectorGraphingToolkit/

Author: Braedyn Lettinga

Credits: Heiko Hergert, Ph.D., Tony S. Yu, Ph.D.

This project is licensed under the MIT License - see the page below for more details:
https://github.com/braedynl/VectorGraphingToolkit/blob/master/LICENSE
'''

from __future__ import annotations

from typing import Callable, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import odeint
from sympy.utilities.lambdify import lambdastr, lambdify

np.warnings.filterwarnings('ignore')


class Vector(object):
    '''
    This class simulates a two-dimensional vector in R^2 space.

    Parameters
    ----------
        u : Base u scalar.
        v : Base v scalar.
        name : Name of the vector. Used for plot interactivity and string methods.
    
    Raises
    ------
        TypeError : If not all scalars are a numeric type.
    
    Notes
    -----
        'u', 'v', and 'name' are all attributes, and have corresponding setters.
        There is also a 'scalars' getter/setter, which returns a numpy ndarray
        of the two scalars. It can be set with any iterable of shape (2, ).

        Scalars are kept un-rounded internally. You can use repr() to see the 
        scalars un-rounded. __str__() will always round to the second decimal
        place. 
    '''

    def __init__(self, u:float, v:float, name:str='v'):
        self.__scalar_handle(u, v)
        self.__scalars = np.array((u, v), dtype=np.dtype(float))
        self.name = str(name)

        # plot method "globals"
        self.__fig = None
        self.__ax = None

        self.__color = 'skyblue'
        self.__x0 = 0  # starting x-coordinate
        self.__y0 = 0  # starting y-coordinate
        self.__u = u  # u scalar
        self.__v = v  # v scalar
        self.__x1 = self.__x0 + self.__u  # ending x-coordinate
        self.__y1 = self.__y0 + self.__v  # ending y-coordinate
        self.__q = None  # Quiver instance for plot()
        self.__q_kwargs = None
        self.__xtrace = None  # xtrace Line2D instance
        self.__ytrace = None  # ytrace Line2D instance
        self.__annotation = '$\\angle x = {:.2f}^c, \\angle y = {:.2f}^c$\n$x_0 = {:.2f}, y_0 = {:.2f}$\n$x_1 = {:.2f}, y_1 = {:.2f}$\n$u = {:.2f}, v = {:.2f}$\n$mag(\\vec{{{name}}}) = {:.2f}$'
        self.__annot = None  # annotation
        self.__drag_pt = None  # Line2D drag point instance

        # state trackers
        self.__trace_state = False
        self.__dragging_pt = False
        self.__dragging_vec = False
    
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
        '''Reverse multiplies vector by a scalar value.'''
        return self.__mul__(a)

    def __imul__(self, a:float) -> self:
        '''In-place multiples vector by a scalar value.'''
        self.__scalars *= a 
        return self

    def __truediv__(self, a:float) -> Vector:
        '''Divides vector by a scalar value.'''
        return Vector(*(self.__scalars / a), '({}/{})'.format(self.name, a))
    
    def __idiv__(self, a:float) -> self:
        '''In-place divides vector by a scalar value.'''
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
        '''Obtains the u or v scalar from the array of scalars.'''
        return self.__scalars[index]
    
    def __setitem__(self, index:int, value:float) -> None:
        '''Sets the u or v scalar from the array of scalars.'''
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
        if len(arr) != 2:
            raise ValueError('dimension mismatch')
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
        self.__scalars = self.__scalars / self.mag  # `/=` doesn't work here for some reason
        return self

    def plot(self, fig:Figure, ax:Axes, x:float=0, y:float=0, color:str='skyblue', trace_scalars:bool=False, 
             interactive:bool=False, **kwargs) -> Quiver:
        '''
        Plots the vector on a given matplotlib Axes.

        Parameters
        ----------
            fig : A matplotlib Figure instance.
            ax : A two-dimensional matplotlib Axes instance.
            x : Starting x-coordinate of the vector.
            y : Starting y-coordinate of the vector.
            color : Color of the vector. Argument passed to ~Axes.quiver().
            trace_scalars : Option to plot dashed lines that represent the scalar values of the vector.
                - The u scalar is represented in blue ('C0'), the v scalar is represented in orange ('C1')
            interactive : Option to make the vector interactable.
                - A point is plotted at the tip of the vector that allows the user to warp, shift and view
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
            The ~Axes.quiver() method is called with arguments units='xy' and scale=1. This is to prevent warping
            and auto-scaling from matplotlib, and has the consequence of the user not being able to call these
            parameters in kwargs. 

            seaborn can be used, and makes the plots look a lot prettier. Can make interactability slow, however.
        '''
        self.__fig = fig 
        self.__ax = ax 
        self.__ax.set_aspect('equal')

        self.__x0 = x
        self.__y0 = y
        self.__x1 = self.__x0 + self.__u
        self.__y1 = self.__y0 + self.__v

        self.__color = color
        self.__q_kwargs = kwargs

        self.__trace_state = trace_scalars

        if self.__trace_state:
            self.__xtrace = self.__ax.plot(
                (self.__x0, self.__x1),
                (self.__y0, self.__y0),
                linestyle='--',
                color='C0'
            )

            self.__ytrace = self.__ax.plot(
                (self.__x1, self.__x1),
                (self.__y0, self.__y1),
                linestyle='--',
                color='C1'
            )

        self.__q = self.__ax.quiver(self.__x0, self.__y0, *self.__scalars, units='xy', scale=1, color=color, **self.__q_kwargs)
        
        if interactive:
            self.__drag_pt = self.__ax.scatter(self.__x1, self.__y1, color='grey', alpha=0.7)

            self.__annot = self.__ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points', 
                                              bbox={'boxstyle': 'round', 'fc': 'w', 'pad': 0.4, 'alpha': 0.7})
            self.__annot.set_visible(False)

            self.__fig.canvas.mpl_connect('button_press_event', self.__on_click)
            self.__fig.canvas.mpl_connect('button_release_event', self.__on_release)
            self.__fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)

        return self.__q

    def get_latex_str(self, notation:Union['angled', 'parentheses', 'unit']='angled') -> str:
        '''
        Returns a string of the Vector instance in LaTeX formatting.

        Parameters
        ----------
            notation : Changes the notation style of the string.    
                - 'angled' : Ordered set notation, angle-bracket variant.
                - 'parentheses' : Ordered set notation, parentheses variant.
                - 'unit' : Unit vector notation. 
        
        Returns
        -------
            str : Vector instance in LaTeX formatting.
        
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

    def __update_quiver(self, x:float, y:float, u:float, v:float) -> None:
        '''
        Updates the drawn quiver and subsequently re-draws the figure.

        Parameters
        ----------
            x : x-coordinate of the vector.
            y : y-coordinate of the vector.
            u : u scalar of the vector.
            v : v scalar of the vector.
        
        Notes
        -----
            The quiver, (self.__q), is removed from the figure entirely and
            re-drawn. The .set_UVC() method does not work here, since the vector's
            position will be changing, and .set_UVC() only configures the u and v
            scalars.
        '''
        self.__drag_pt.set_offsets([x + u, y + v])

        self.__q.remove()
        self.__q = self.__ax.quiver(x, y, u, v, units='xy', scale=1, color=self.__color, **self.__q_kwargs)

        self.__fig.canvas.draw_idle()

    def __update_annot(self, xdata:float, ydata:float) -> None:
        '''
        Updates the annotation with the given x and y data from the mouse pointer's location.

        Parameters
        ----------
            xdata : Mouse pointer's x position.
            ydata : Mouse pointer's y position.
        '''
        self.__annot.xy = (xdata, ydata)
        self.__annot.set_text(
            self.__annotation.format(
                    np.arctan(self.__v / self.__u),
                    np.arctan(self.__u / self.__v),
                    self.__x0, 
                    self.__y0, 
                    self.__x1, 
                    self.__y1,
                    self.__u, 
                    self.__v,
                    np.linalg.norm([self.__u, self.__v]), 
                    name=self.name
                )
            )

    def __on_release(self, event:MouseEvent) -> None:
        '''
        Halts all warping/shifting of the vector if the mouse button has been released. Additionally
        hides annotation. 

        Parameters
        ----------
            event : Mouse event object passed by ~mpl_connect().
        '''
        if event.inaxes == self.__ax:
            self.__annot.set_visible(False)
            self.__fig.canvas.draw_idle()

            # event.button != 2 ensures the mouse button is right/left-click,
            # causes middle-click bugs without this check
            if self.__drag_pt.contains(event)[0] and event.button != 2:
                self.__dragging_pt = False
                self.__dragging_vec = False

                self.__x1 = event.xdata
                self.__y1 = event.ydata
                self.__u = self.__x1 - self.__x0
                self.__v = self.__y1 - self.__y0

                self.__update_quiver(self.__x0, self.__y0, self.__u, self.__v)
            
    def __on_motion(self, event:MouseEvent) -> None:
        '''
        Calculates new scalars and initial positions in the event of a drag.

        Parameters
        ----------
          event : Mouse event object passed by ~mpl_connect().
        '''
        if event.inaxes == self.__ax and event.button != 2:
            self.__x1 = event.xdata
            self.__y1 = event.ydata

            if self.__dragging_pt:
                self.__u = self.__x1 - self.__x0
                self.__v = self.__y1 - self.__y0

            elif self.__dragging_vec:
                self.__x0 = self.__x1 - self.__u
                self.__y0 = self.__y1 - self.__v

            if self.__trace_state and (self.__dragging_pt or self.__dragging_vec):
                self.__xtrace[0].set_data((self.__x0, self.__x1), (self.__y0, self.__y0))
                self.__ytrace[0].set_data((self.__x1, self.__x1), (self.__y0, self.__y1))

            self.__update_annot(event.xdata, event.ydata)
            self.__update_quiver(self.__x0, self.__y0, self.__u, self.__v)

    def __on_click(self, event:MouseEvent) -> None:
        '''
        Calculates new scalars and initial positions in the event of a drag.

        Parameters
        ----------
            event : Mouse event object passed by ~mpl_connect().
        '''
        if self.__drag_pt.contains(event)[0] and event.inaxes == self.__ax:

            if event.button == 1:
                self.__dragging_pt = True

            elif event.button == 3:
                self.__dragging_vec = True

            self.__update_annot(event.xdata, event.ydata)
            self.__annot.set_visible(True)
            self.__fig.canvas.draw_idle()


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
        'expr' represents a sympy expression.

        'u', 'v', and 'name' are all attributes, and have corresponding setters.
    '''

    def __init__(self, u:Union[str, float, 'expr'], v:Union[str, float, 'expr'], name:str='F'):

        self.__usym = sym.sympify(u)
        self.__vsym = sym.sympify(v)

        x, y = sym.symbols('x y')
        self.__unp = lambdify((x, y), self.__usym, 'numpy' )
        self.__vnp = lambdify((x, y), self.__vsym, 'numpy' )

        self.name = str(name)

        # plot method "globals"
        self.__field_fig = None
        self.__field_ax = None
        self.__field_xlim = None
        self.__field_ylim = None

        self.__scale = 1
        self.__cmap = 'Blues'
        self.__cmap_func = 'mag'
        self.__q = None  # Quiver instance for plot()
        self.__q_kwargs = None
        self.__cbar = None
        self.__label = None  # label for cbar
        self.__annotation = '$x = {:.2f}, y = {:.2f}$\n$mag(\\vec{{{name}}})(x, y) = {:.2f}$\n$div(\\vec{{{name}}})(x, y) = {:.2f}$\n$curl(\\vec{{{name}}})(x, y) = {:.2f}$'
        self.__annot = None  # annotation
        self.__func_labels = {'mag' : 'Magnitude', 'div' : 'Divergence', 'curl': 'Curl'}  # proper name of cmap_func for cbar's label

        # particles method "globals"
        self.__ani_fig = None
        self.__ani_ax = None
        self.__ani_xlim = None
        self.__ani_ylim = None

        self.__ani = None  # FuncAnimation instance for particles()
        self.__ln = None  # Line2D array for all particles
        self.__pts = None  # array of particle positions

        # state trackers
        self.__normalize_state = True  
        self.__cbar_state = True
        self.__interactive_state = False

    def __str__(self) -> str:
        '''Returns a string of the vector field in angle-bracket notation.'''
        return '{} = <{}, {}>'.format(self.name, self.__usym, self.__vsym)
    
    def __repr__(self) -> str:
        '''Returns a string of the vector field in angle-bracket notation, with lambda parameters.'''
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
            f : A function of two variables (must be x and y)
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
    def curl(self) -> Callable[[float, float], float]:
        '''The curl function of the vector field.'''
        x, y = sym.symbols('x y')
        return lambdify((x, y), sym.diff(self.__vsym, x) - sym.diff(self.__usym, y), 'numpy')

    @property
    def div(self) -> Callable[[float, float], float]:
        '''The divergence function of the vector field.'''
        x, y = sym.symbols('x y')
        return lambdify((x, y), sym.diff(self.__usym, x) + sym.diff(self.__vsym, y), 'numpy')

    def is_solenoidal(self) -> bool:
        '''Tests if the vector field is solenoidal. Returns bool.'''
        x, y = sym.symbols('x y')
        return True if sym.diff(self.__usym, x) + sym.diff(self.__vsym, y) == 0 else False

    def is_conservative(self) -> bool:
        '''Tests if the vector field is conservative. Returns bool.'''
        x, y = sym.symbols('x y')
        return True if sym.diff(self.__vsym, x) - sym.diff(self.__usym, y) == 0 else False

    def plot(self, fig:Figure, ax:Axes, scale:float=1, density:int=10, cmap:Union[str, ListedColormap]='Blues', 
             cmap_func:Union['mag', 'div', 'curl']='mag', normalize:bool=True, colorbar:bool=True, 
             interactive:bool=False, **kwargs) -> Quiver:
        '''
        Plots the vector field on a given matplotlib Axes.

        Parameters
        ----------
            fig : A matplotlib Figure instance.
            ax : A two-dimensional matplotlib Axes instance.
            scale : Scalar value applied to each vector. See notes below for more details.
            density : A measure of how many vectors to plot within the field.
                - An evenly-spaced grid of density*density vectors is plotted on the axes.
                - Value must be within range [4, 100]. 
            cmap : A matplotlib colormap applied to the field.
                - Can pass a built-in or custom colormap. 
            cmap_func : One of the three VectorField properties to use in determination of the color mapping.
                - 'mag' : Maps colors to the vectors based on their respectives magnitudes.
                - 'div' : Maps colors to the vectors based on the divergence of the vectors' initial positions.
                - 'curl' : Maps colors to the vectors based on the curl of the vectors' initial positions.
            normalize : Option to normalize each vector. See notes below for more details.
            colorbar : Option to display a colorbar of the color mapping.
                - The values show, both on the colorbar ticks and the colorbar's label, will vary depending
                  on the chosen scale, density and cmap_func.
            interactive : Option to make the vector field interactable.
                - The plot will detect mouse clicks, and sliders will be added below the axes.
                - Clicking and holding on a point within the axes will display an annotation describing the
                  curl, divergence, and magnitude at that point.
                - The scale and density sliders allow for further adjustment of the field, and function just
                  as the scale and density parameters do.
                - The upper-bound of the scale slider is calculated using the x and y limits of the ax:
                  ```round(max(abs(val) for val in (xlim + ylim)) / 4)```
                - Values less than 1 will be rounded to 1.
            **kwargs : Additional arguments passed to ~Axes.quiver(). 
        
        Returns
        -------
            matplotlib.quiver.Quiver : The created Quiver instance.
        
        Raises
        ------
            ValueError : If density is not within range [4, 100], or cmap_func is none of the possible options.
        
        References
        ----------
            built-in cmap options: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            kwargs options: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html
        
        Notes
        -----
            When normalize=True, all vectors will be converted into their unit vector form (their scalars get
            divided by their magnitude). The scale argument is then applied after. An auto-scaling algorithm
            could be implemented in the future, since extremely small axes will have huge vectors and extremely
            large axes will have small vectors, without adjust the scale from its default, 1. 

            Like the Vector class's plot() method, ~Axes.quiver() is called with arguments units='xy' and scale=1
            to prevent warping and auto-scaling from matplotlib, and has the consequence of the user not being able
            to call these parameters in kwargs. This could be subject to change in the future, since matplotlib could
            be left to take care of the issue described above.

            seaborn can be used, and makes the plots look a lot prettier. Can make interactability slow, however.
        '''
        if not 4 <= density <= 100:
            raise ValueError('density argument must be within range [4, 100]')
        if cmap_func not in self.__func_labels:
            raise ValueError('cmap_func, "{}", not recognized -- possible options are: "mag", "curl", "div"'.format(cmap_func))

        self.__field_fig = fig
        self.__field_ax = ax
        self.__field_ax.set_aspect('equal')

        self.__field_xlim = self.__field_ax.get_xlim()
        self.__field_ylim = self.__field_ax.get_ylim()

        self.__normalize_state = normalize  
        self.__cbar_state = colorbar
        self.__interactive_state = interactive
        
        self.__q_kwargs = kwargs
        self.__cmap = cmap
        self.__cmap_func = cmap_func
        self.__scale = scale

        components = self.__create_field(scale, density)

        self.__q = self.__field_ax.quiver(*components, units='xy', scale=1, cmap=self.__cmap, **self.__q_kwargs)

        if self.__cbar_state:
            plt.subplots_adjust(left=0.05)
            c = components[-1]
            
            # an inset_axes is used here for a specific reason. if sliders are active, the
            # colorbar will stretch down to encapsulate the Axes height combined with the sliders below. 
            # an inset_axes circumvents this problem. This could be subject to change.
            cbar_ax = inset_axes(
                self.__field_ax,
                width='5%',
                height='100%',
                loc='lower left',
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=self.__field_ax.transAxes,
                borderpad=0,
                axes_kwargs={'zorder': -1}
            )
            
            self.__cbar = plt.colorbar(
                self.__q,
                cax=cbar_ax,
                cmap=self.__cmap,
                label=self.__label.format(self.__func_labels[self.__cmap_func], self.__scale),
                ticks=np.linspace(c.min(), c.max(), 5)
            )

        if self.__interactive_state:
            self.__annot = self.__field_ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points',
                                                    bbox={'boxstyle': 'round', 'fc': 'w', 'pad': 0.4, 'alpha': 0.7})
            self.__annot.set_visible(False)

            self.__field_fig.canvas.mpl_connect('button_press_event', self.__on_click)
            self.__field_fig.canvas.mpl_connect('button_release_event', self.__on_release)

            # very rudimentary scale range setter -- needs improvement
            scale_range = round(max(abs(val) for val in (self.__field_xlim + self.__field_ylim)) / 4)
            if scale_range < 1:
                scale_range = 1

            divider = make_axes_locatable(self.__field_ax)

            scale_ax = divider.append_axes('bottom', size='3%', pad=0.6) 
            self.__scale_slider = Slider(scale_ax, 'Scale', -scale_range, scale_range, valinit=self.__scale)
            self.__scale_slider.on_changed(self.__slider_update)

            density_ax = divider.append_axes('bottom', size='3%', pad=0.1)
            self.__density_slider = Slider(density_ax, 'Density', 4, 100, valinit=density, valstep=1)
            self.__density_slider.on_changed(self.__slider_update)

        return self.__q

    def particles(self, fig:Figure, ax:Axes, pts:Iterable[tuple]=None, frames:int=300, dt:float=0.01, 
                  fmt:str='o', color:str='k', alpha:float=0.7, **kwargs) -> FuncAnimation:
        '''
        Animates particles on a given matplotlib Axes, where relative velocities are modeled by the field.

        Parameters
        ----------
            fig : A matplotlib Figure instance.
            ax : A two-dimensional matplotlib Axes instance.
            pts : An array of coordinate pairs that set the initial particle positions.
                - If None, 50 randomly-placed particles will be plotted.
            frames : The amount of frames to run the animation for.
            dt : The change in time from one frame to the next.
                - I recommend keeping this value extremely small. Larger values may result in ludicrously
                  fast particle speeds.
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
            ~Axes.plot() arguments: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.plot.html

        Notes
        -----
            The velocities of the particles are relative. If the plot() method is active, its scale parameter
            will affect the speed of the particles. Additionally, if interactive=True, particle velocities will
            reflect the value set by the scale slider. The particle simulation will slow down significantly --
            unfortunately because matplotlib isn't the greatest at handling animations. When interactive=False,
            however, the animation will be blitted, and will perform considerably faster and smoother.
        
        Credits
        -------
            Heiko Hergert, Ph.D. - Helped with my conceptualization of the flow.
            Tony S. Yu, Ph.D. - The only human being on the planet that has an example.

            Much of the particle animation work is based on Dr. Yu's article, which can be found here:
            https://tonysyu.github.io/animating-particles-in-a-flow.html
        '''
        self.__ani_fig = fig 
        self.__ani_ax = ax

        self.__ani_xlim = self.__ani_ax.get_xlim()
        self.__ani_ylim = self.__ani_ax.get_ylim()

        self.__ln, = self.__ani_ax.plot([], [], fmt, color=color, alpha=alpha, **kwargs)

        if pts is None:
            pts = np.array((np.random.uniform(*self.__ani_xlim, 50), np.random.uniform(*self.__ani_ylim, 50))).transpose()

        # blit is importantly set to False if interactivity is enabled. if blit=True,
        # changes to the vector field won't appear, or will act undesirably when using
        # the sliders.
        if self.__interactive_state:
            self.__ani = FuncAnimation(self.__ani_fig, self.__particle_update, interval=1, frames=frames, blit=False, fargs=(pts, dt))
        else:
            self.__ani = FuncAnimation(self.__ani_fig, self.__particle_update, interval=1, frames=frames, blit=True, fargs=(pts, dt))        
        
        return self.__ani

    def get_latex_str(self, notation:Union['angled', 'parentheses', 'unit']='angled') -> str:
        '''
        Returns a string of the VectorField instance in LaTeX formatting.

        Parameters
        ----------
            notation : Changes the notation style of the string.    
                - 'angled' : Ordered set notation, angle-bracket variant.
                - 'parentheses' : Ordered set notation, parentheses variant.
                - 'unit' : Unit vector notation. 
        
        Returns
        -------
            str : VectorField instance in LaTeX formatting.
        
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

    def __solve_ode(self, f:Callable, pts:Iterable[tuple], dt:float) -> list:
        '''
        Solves for the displacement of each particle with respect to the change in time.

        Parameters
        ----------
            f : Function integrand. In our case, the u and v scalar functions. 
            pts : Array of coordinate pairs. 
            dt : The change in time from one frame to the next.
        
        Returns
        -------
            list : A list of all updated particle positions.
        '''
        return [odeint(f, pt, [0, dt])[-1] for pt in pts]  # using an ndarray here would be better -- currently figuring that out if possible

    def __vels(self, pt:tuple, _) -> list:
        '''
        Calculates the velocity of the particle at a specific point.

        Parameters
        ----------
            pt : An x, y coordinate pair.
            _ : Dummy time parameter required for odeint(). Unused since scalar functions don't depend on time.
        
        Returns
        -------
            list : The x and y velocity at the given point (index 0 and index 1, respectively).
        
        Notes
        -----
            If interactivity from the plot() method is enabled, the slider values will be multiplied by the
            output of the scalar functions. Otherwise, the scale parameter will be used.
        '''
        if self.__interactive_state:
            return [self.__scale_slider.val * self.__unp(*pt), self.__scale_slider.val * self.__vnp(*pt)]

        return [self.__scale * self.__unp(*pt), self.__scale * self.__vnp(*pt)]

    def __remove_pts(self, pts:np.ndarray) -> np.ndarray:
        '''
        Removes points that are outside the bounds of the axes. 

        Parameters
        ----------
            pts : Array of coordinate pairs.
        
        Returns
        -------
            np.ndarray : The same array but with out-of-boundary points removed.
        
        Credits
        -------
            Tony S. Yu, Ph.D.
        '''
        if len(pts) == 0:
            return []
        out_x = (pts[:, 0] < self.__ani_xlim[0]) | (pts[:, 0] > self.__ani_xlim[1])
        out_y = (pts[:, 1] < self.__ani_ylim[0]) | (pts[:, 1] > self.__ani_ylim[1])
        keep = ~(out_x | out_y)

        return pts[keep]

    def __particle_update(self, frame:int, *fargs) -> Line2D:
        '''
        Calculates particle displacements, removes out-of-axes points, and subsequently updates the 
        axes with new particle positions.

        Parameters
        ----------
            frame : Current frame of the animation.
            *fargs : A list of the intial particle positions at index 0 (type: Iterable), and the dt value at index 1 (type: float).
        
        Returns
        -------
            matplotlib.lines.Line2D : The updated Line2D array.
        '''
        # self.__pts needs to be a data member, though it may not look like it at first.
        # transition arrays would be reset back to the initial if not a data member. 

        if frame == 0:
            self.__pts = fargs[0]

        self.__pts = np.asarray(self.__solve_ode(self.__vels, self.__pts, fargs[1]))
        self.__pts = np.asarray(self.__remove_pts(self.__pts))

        self.__pts.shape = (self.__pts.shape[0], 2)  # rudimentary way of ensuring .transpose() runs properly
        x, y = self.__pts.transpose()

        self.__ln.set_data(x, y)

        return self.__ln,

    def __on_release(self, event:MouseEvent) -> None:
        '''
        Hides the annotation on release of the mouse button.

        Parameters
        ----------
            event : Mouse event object passed by ~mpl_connect().
        '''
        if event.inaxes == self.__field_ax:
            self.__annot.set_visible(False)
            self.__field_fig.canvas.draw_idle()

    def __on_click(self, event:MouseEvent) -> None:
        '''
        Updates and displays the annotation on click of the axes.

        Parameters
        ----------
            event : Mouse event object passed by ~mpl_connect().
        '''
        if event.inaxes == self.__field_ax:
            self.__annot.xy = (event.xdata, event.ydata)
            self.__annot.set_text(
                self.__annotation.format(
                    event.xdata, 
                    event.ydata, 
                    self.mag(event.xdata, event.ydata), 
                    self.div(event.xdata, event.ydata), 
                    self.curl(event.xdata, event.ydata), 
                    name=self.name
                    )
                )
            
            self.__annot.set_visible(True)
            self.__field_fig.canvas.draw_idle()

    def __slider_update(self, _) -> None:
        '''
        Re-draws vector field on update of a slider.

        Parameters
        ----------
            _ : Dummy parameter for the argument emitted by ~Slider.on_changed().
        '''
        components = self.__create_field(self.__scale_slider.val, self.__density_slider.val)

        self.__q.remove()
        self.__q = self.__field_ax.quiver(*components, units='xy', scale=1, cmap=self.__cmap, **self.__q_kwargs)

        if self.__cbar_state:
            self.__cbar.set_label(self.__label.format(self.__func_labels[self.__cmap_func], self.__scale_slider.val))

        self.__field_fig.canvas.draw_idle()

    def __create_field(self, scale:float, density:int) -> tuple:
        '''
        Creates a grid of coordinate points and calculates all corresponding vectors. Additionally
        maps colors from the cmap dependent on the cmap_func.

        Parameters
        ----------
            scale : Scalar value applied to each vector.
            density : A measure of how many vectors to plot within the field.
                - An evenly-spaced grid of density*density vectors is plotted within on the axes.
                - Value must be within range [4, 100]. 
        
        Returns
        -------
            tuple : Returns a tuple of the x, y, u, v and c arrays for each vector in the field. 
                - x : Array of x-coordinates.
                - y : Array of y-coordinates.
                - u : Array of u scalars.
                - v : Array of v scalars.
                - c : Array of values used in determining the mapping of colors.
        '''
        x, y = np.meshgrid(
            np.linspace(*self.__field_xlim, int(density)),
            np.linspace(*self.__field_ylim, int(density))
        )

        u, v = (
            self.__unp(x, y),
            self.__vnp(x, y)
        )

        m = np.sqrt(u**2 + v**2)

        if self.__normalize_state:
            with np.errstate(all='ignore'):
                u = (u / m) * scale
                v = (v / m) * scale
                self.__label = 'Sampled {} (Scale: {:.2f}, Normalized)'
        else:
            u = u * scale
            v = v * scale
            self.__label = 'Sampled {} (Scale: {:.2f})'
        
        if self.__cmap_func == 'mag':
            c = m
        elif self.__cmap_func == 'div':
            c = self.div(x, y)
        elif self.__cmap_func == 'curl':
            c = self.curl(x, y)
        
        c = np.full_like(x, c)

        return x, y, u, v, c
