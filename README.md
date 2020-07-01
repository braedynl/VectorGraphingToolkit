# VectorGraphingToolkit

*Current Version: 1.0.0*

A simplistic vector/vector field visualization tool built on top of `matplotlib`.

I made this purely for fun. Can it be used for anything practical? I have no idea - I'm an amateur programmer, not a physicist. If it does prove useful to anyone, I would love to hear about your use-case and your take on some improvements that could be made. I am extremely open to collaboration as well, if you want to contribute to the project, please get in contact. 

Feel free to submit bugs, feature requests, etc. to the issues page. Changelog can be found [here](CHANGELOG.md). The current version of this module, 1.0.0, is what I'm deeming the first "official" version -- I don't doubt the possibility of numerous bugs and oversights.

<p align="center"> 
  <img src=examples/ex_intro.gif>
</p>

[Source](examples/ex_intro.py)

## Installation/Requirements

Through terminal:

```
$ python setup.py install
```

Requirements can be found [here](requirements.txt). Currently not on PyPi. 

## Documentation

The following documentation expects that you know a little bit about `matplotlib`. You can learn the basics [here](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py).

The `VectorField` class supports expressions from the `sympy` library, though usage is mostly optional. You can learn how to create `sympy` expressions [here](https://docs.sympy.org/latest/tutorial/basic_operations.html). 

This documentation also assumes you are familiar with some of the basics of multivariable calculus. 

## Table of Contents
- [`Vector`](#vector)
  - [Constructor](#constructor)
  - [Getters/Setters](#getterssetters)
    - [`u`](#getterssetters)
    - [`v`](#getterssetters)
    - [`name`](#getterssetters)
    - [`scalars`](#getterssetters)
  - [Properties](#properties)
    - [`mag`](#properties)
  - [Overloads](#overloads)
    - [`__add__` / `__iadd__`](#overloads)
    - [`__sub__` / `__isub__`](#overloads)
    - [`__mul__` / `__imul__`](#overloads)
    - [`__truediv__` / `__idiv__`](#overloads)
    - [`__eq__` / `__ne__`](#overloads)
    - [`__getitem__` / `__setitem__`](#overloads)
    - [`__invert__`](#overloads)
    - [`__xor__`](#overloads)
  - [Method Functions](#method-functions)
    - [`dot`](#dot)
    - [`angle`](#angle)
    - [`unit`](#unit)
    - [`plot`](#plot)
    - [`get_latex_str`](#get_latex_str)
- [`VectorField`](#vectorfield)
  - [Constructor](#constructor-1)
  - [Getters/Setters](#getterssetters-1)
    - [`u`](#getterssetters-1)
    - [`v`](#getterssetters-1)
    - [`name`](#getterssetters-1)
  - [Properties](#properties-1)
    - [`mag`](#properties-1)
    - [`curl`](#properties-1)
    - [`div`](#properties-1)
  - [Class Methods](#class-methods)
    - [`from_grad`](#from_grad)
    - [`from_stream`](#from_stream)
  - [Method Functions](#method-functions-1)
    - [`is_solenoidal`](#is_solenoidal)
    - [`is_conservative`](#is_conservative)
    - [`plot`](#plot-1)
    - [`particles`](#particles)
    - [`get_latex_str`](#get_latex_str-1)


___

# `Vector`

This class simulates a two-dimensional vector in R^2 space. 

## Constructor
```python
Vector(u:float, v:float, name:str='v')
```
- `u` : Base u scalar.
- `v` : Base v scalar.
- `name` : Name of the vector. Used for plot interactivity and string methods. 

Example:
```python
from vgtk import Vector

v = Vector(1, 2)
w = Vector(-3.049, 4.5, 'w')

print(v)
print(w)
```
Out:
```
v = <1.00, 2.00>
w = <-3.05, 4.50>
```
Notes:

You'll notice that the scalars are always rounded to the second decimal place. Rounding only occurs during invocation of the `__str__` method, the scalars are kept un-rounded internally. You can view the un-rounded scalars by calling `repr()`. 

Conversion from an array is not inherently supported by the constructor. Use the `*` operator if you want to unpack two scalar values from an array. 

Example:
```python
import numpy as np
from vgtk import Vector

arr = np.array([3.141, 2.718])

v = Vector(*arr)

print(repr(v))
```
Out:
```
v = <3.141, 2.718>
```

Back to [table of contents](#table-of-contents).

## Getters/Setters

All getters include corresponding setters.

```python
from vgtk import Vector

v = Vector(1, 2)

print(v.name)  # name of the vector (str)

print(v.u)  # u scalar (float)
print(v.v)  # v scalar (float)

print(v.scalars)  # numpy ndarray of scalars (will always be shape: (2, ))
```
Out:
```
v
1.0
2.0
[1. 2.]
```
Notes:

The `scalars` setter can be any iterable of numeric types, as long as it's of shape `(2, )`. The iterable will be converted into a `numpy` `ndarray` internally.

Back to [table of contents](#table-of-contents).

## Properties

`Vector` has one property, `mag`, which returns the magnitude of the vector (type: `float`).

```python
from vgtk import Vector

# mag(v) = sqrt(u**2 + v**2)

v = Vector(3, 0)

print(v.mag)  # sqrt(3**2 + 0**2) = sqrt(3**2) = 3
```
Out:
```
3.0
```

Back to [table of contents](#table-of-contents).

## Overloads

The class has overloads for the `+`, `-`, `*`, and `/` operators, as well as their compound assignment counterparts. The compound assignments are treated as in-place methods. There are overloads for the `==` and `!=` operators as well, which simply compare component-by-component, exactly as you'd expect. 

When using the `+`, `-`, `*` and `/` operators, the resultant instance will have a name created from the other names/scalars involved in the arithemtic (see example). The compound assignments do not exhibit this behaviour. 

Subscripting is supported, albeit in a minuscule range. Index 0 is the u scalar, index 1 is the v scalar. 

```python
from vgtk import Vector

v1 = Vector(1, 2, name='v1')
v2 = Vector(3, 2, name='v2')

v3 = v1 + v2
print(v3)

v4 = v1 - v2
print(v4)

# * and / are overloaded for scalar values only

v5 = v1 * 2
print(v5)

v6 = v1 / 2
print(v6)

# recommended way of changing the default name 
v3 = v1 + v2; v3.name = 'v3'
print(v3)

print(v1 == v2)
print(v1 != v2) 

print(v1[0])
v1[0] = 2  # supports setting as well
print(v1)
```
Out:
```
(v1+v2) = <4.00, 4.00>
(v1-v2) = <-2.00, 0.00>
(2*v1) = <2.00, 4.00>
(v1/2) = <0.50, 1.00>
v3 = <4.00, 4.00>
False
True
1.0
v1 = <2.00, 2.00>
```

The `~` operator will transform the vector into its unit vector form, while the `^` operator will take the dot product between two `Vector` instances. There are method function equivalents for both, which can be seen in the next section below. 

```python
from vgtk import Vector

v = Vector(1, 2)

print(~v)  # (in-place, returns self -- see dot() method)

v1 = Vector(1, 0)
v2 = Vector(0, 1)

print(v1 ^ v2)
```
Out:
```
v = <0.45, 0.89>
0.0
```

Back to [table of contents](#table-of-contents).

## Method Functions

### `dot`
Calculates the dot product between two `Vector` instances. Returns `float`.
```python
dot(self, other:Vector) -> float
```
- `other` : Another `Vector` instance.

Example:
```python
from vgtk import Vector

v1 = Vector(1, 0)
v2 = Vector(0, 1)

dot_prod = v1.dot(v2)

print(dot_prod)
```
Out:
```
0.0
```

### `angle`
Measures the radian angle between two `Vector` instances. Returns `float`.
```python
angle(self, other:Vector, degrees:bool=False) -> float
```
- `other` : Another `Vector` instance. 
- `degrees` : Returns angle measured in degrees if `True`.

Example:
```python
from vgtk import Vector

v1 = Vector(1, 0)
v2 = Vector(0, 1)

in_radians = v1.angle(v2)
in_degrees = v1.angle(v2, degrees=True)

print(in_radians)
print(in_degrees)
```
Out:
```
1.5707963267948966
90.0
```

### `unit`
Converts the `Vector` into its equivalent unit vector form. Returns `self`. 
```python
unit(self) -> self
```
Example:
```python
from vgtk import Vector

v = Vector(11, -31)

v.unit()

print(v.mag)
print(v)
```
Out:
```
1.0
v = <0.33, -0.94>
```

### `plot`
Plots the vector on a given `matplotlib` `Axes`. Returns the created `~quiver.Quiver` instance. 
```python
plot(self, fig:Figure, ax:Axes, x:float=0, y:float=0, color:str='skyblue', trace_scalars:bool=False, interactive:bool=False, **kwargs) -> Quiver
```
- `fig` : A `matplotlib` `Figure` instance.
- `ax` : A two-dimensional `matplotlib` `Axes` instance. 
- `x` : Starting x-coordinate of the vector. 
- `y` : Starting y-coordinate of the vector. 
- `color` : Color of the vector. Argument passed to `~Axes.quiver()`. Options can be found [here.](https://matplotlib.org/3.1.0/gallery/color/named_colors.html)
- `trace_scalars` : Option to plot dashed lines that represent the scalar values of the vector. 
  - The u scalar is represented in blue (`'C0'`), the v scalar is represented in orange (`'C1'`). 
- `interactive` : Option to make the vector interactable. A point is plotted at the tip of the vector that allows the user to warp, shift, and view various details about the vector. 
  - Holding left-click will drag the vector's tip to the mouse pointer's location, while the base of the vector remains fixed. 
  - Holding right-click will drag the entire vector to the mouse pointer's location, while the magnitude and direction remain fixed. 
  - Holding middle-click will show the vector's details without warping or shifting.
- `kwargs` : Additional arguments passed to `~Axes.quiver()`. Options can be found [here.](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html)

Example:
```python
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from vgtk import Vector

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=3)

v = Vector(1, 2)

v.plot(fig, ax, color='k', trace_scalars=True)

ax.set_title(v.get_latex_str(True))
plt.show()
```
Out:

<p align="center"> 
  <img src=examples/ex_plot1.png>
</p>

Example:
```python
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from vgtk import Vector

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=3)

v = Vector(1, 2)

v.plot(fig, ax, color='k', trace_scalars=True, interactive=True)

plt.show()
```

[Video demonstration of this example.](examples/ex_plot2.mp4?raw=true)

Notes:

The `~Axes.quiver()` method is called with arguments `units='xy'` and `scale=1`. This is to prevent warping and auto-scaling from `matplotlib`, and has the consequence of the user not being able to call these parameters in `kwargs`. 

`seaborn` can be used, and makes the plots look a lot prettier. Can make interactability slow, however. 

### `get_latex_str`
Returns a `str` of the `Vector` instance in LaTeX formatting.
```python
get_latex_str(self, notation:Union['angled', 'parentheses', 'unit']='angled') -> str
```
- `notation` : Changes notation style of the string.
  - `'angled'` : [Ordered set notation, angle-bracket variant.](https://en.wikipedia.org/wiki/Vector_notation#Ordered_set_notation)
  - `'parentheses'` : [Ordered set notation, parentheses variant.](https://en.wikipedia.org/wiki/Vector_notation#Ordered_set_notation)
  - `'unit'` : [Unit vector notation.](https://en.wikipedia.org/wiki/Vector_notation#Unit_vector_notation)

Example:
```python
from vgtk import Vector

v = Vector(1, 2)

angled = v.get_latex_str('angled')
parentheses = v.get_latex_str('parentheses')
unit = v.get_latex_str('unit')

print(angled)
print(parentheses)
print(unit)
```
Out:
```
$\vec{v} = <1.0, 2.0>$
$\vec{v} = (1.0, 2.0)$
$\vec{v} = (1.0)\hat{i} + (2.0)\hat{j}$
```

Back to [table of contents](#table-of-contents).

___

# `VectorField`

This class simulates a two-dimensional vector field in R^2 space.

## Constructor
```python
VectorField(u:Union[str, float, 'expr'], v:Union[str, float, 'expr'], name:str='F')
```
- `u` : Base u scalar function of two variables (must be x and y).
- `v` : Base v scalar function of two variables (must be x and y). 
- `name` : Name of the vector field. Used for plot interactivity and string methods.

`'expr'` represents a `sympy` expression. 

Example:
```python
from vgtk import VectorField

F = VectorField('x', 'y')
G = VectorField('x*y + exp(y)', 'cos(x)*sin(y)', 'G')
H = VectorField(1, 1, 'H')

print(F)
print(repr(F))
print()

print(G)
print(repr(G))
print()

print(H)
print(repr(H))
print()
```
Out:
```
F = <x, y>
F = <lambda x,y: (x), lambda x,y: (y)>

G = <x*y + exp(y), sin(y)*cos(x)>
G = <lambda x,y: (x*y + math.exp(y)), lambda x,y: (math.sin(y)*math.cos(x))>

H = <1, 1>
H = <lambda x,y: (1), lambda x,y: (1)>
```
Notes:

`u` and `v` arguments are passed to `sympy.sympify()`, which you can read more about [here.](https://docs.sympy.org/latest/modules/core.html) Expressions are ran via the `exec()` function, which means that formulae must be in proper Python syntax. Many common mathematical functions can be written without the use of `sympy`, like `'cos()'`, `'sin()'`, `'tan()'`, etc. To express e^x as a string, use `'exp()'`.

Back to [table of contents](#table-of-contents).

## Getters/Setters

All of the following getters have corresponding setters.

```python
from vgtk import VectorField

F = VectorField('x', 'y')

print(F.name)

# Both functions should always have the signature:
# Callable[[float, float], float]

u_ret = F.u(2, 1)  # internally: lambda x, y : x
v_ret = F.v(1, 2)  # internally: lambda x, y : y

print(u_ret)
print(v_ret)
```
Out:
```
F
2
2
```
Setting the `u` and `v` properties is alike passing the `u` and `v` arguments to the constructor in terms of typing.

Back to [table of contents](#table-of-contents).

## Properties

There are three properties: `mag`, `curl` and `div`. All of which are exactly what you'd expect - the [magnitude](https://en.wikipedia.org/wiki/Magnitude_(mathematics)), [curl](https://en.wikipedia.org/wiki/Curl_(mathematics)) and [divergence](https://en.wikipedia.org/wiki/Divergence) functions of the vector field respectively.

```python
from vgtk import VectorField

F = VectorField('x', 'y')
G = VectorField('-y', 'x', 'G')

# mag(F) = sqrt(x**2 + y**2) = sqrt(1**2 + 0**2) = 1
magF = F.mag(1, 0)  # internally: lambda x, y : math.sqrt(x**2 + y**2)

# div(F) = ∂u/∂x + ∂v/∂y = 1 + 1 = 2
divF = F.div(1, 1)  # internally: lambda x, y : 2

# curl(G) = ∂v/∂x - ∂u/∂y = 1 - (-1) = 2
curlG = G.curl(1, 1)  # internally: lambda x, y : 2

print(magF)
print(divF)
print(curlG)
```
Out:
```
1.0
2
2
```

Back to [table of contents](#table-of-contents).

## Class Methods

### `from_grad`
Creates a `VectorField` from the [gradient](https://en.wikipedia.org/wiki/Gradient) of a function. 
```python
from_grad(cls, f:Union[str, float, 'expr'], name:str='F') -> VectorField
```
- `f` : A function of two variables (must be x and y). 
- `name` : Name of the vector field. Used for plot interactivity and string methods.

Example:
```python
from vgtk import VectorField

# f(x, y) = x*y
# F = <∂f/∂x, ∂f/∂y>
# F = <y, x>

F = VectorField.from_grad('x*y')

print(F)
```
Out:
```
F = <y, x>
```

### `from_stream`
Creates a `VectorField` from a given [stream function](https://en.wikipedia.org/wiki/Stream_function).
```python
from_stream(cls, psi:Union[str, float, 'expr'], name:str='F') -> VectorField
```
- `psi` : A stream function of two variables (must be x and y).
- `name` : Name of the vector field. Used for plot interactivity and string methods.

Example:
```python
from vgtk import VectorField

# ψ(x, y) = x*y
# F = <∂ψ/∂y, -∂ψ/∂x>
# F = <x, -y>

F = VectorField.from_stream('x*y')

print(F)
```
Out:
```
F = <x, -y>
```

Back to [table of contents](#table-of-contents).

## Method Functions

### `is_solenoidal`
Tests if the vector field is [solenoidal](https://en.wikipedia.org/wiki/Solenoidal_vector_field). Returns `bool`.
```python
is_solenoidal(self) -> bool
```

### `is_conservative`
Tests if the vector field is [conservative](https://en.wikipedia.org/wiki/Conservative_vector_field). Returns `bool`.
```python
is_conservative(self) -> bool
```

### `plot`
Plots the vector field on a given `matplotlib` `Axes`. Returns the created `~quiver.Quiver` instance. 
```python
plot(self, fig:Figure, ax:Axes, scale:float=1, density:int=10, cmap:Union[str, ListedColormap]='Blues', cmap_func:Union['mag', 'div', 'curl']='mag', normalize:bool=True, colorbar:bool=True, interactive:bool=False, **kwargs) -> Quiver
```
- `fig` : A `matplotlib` `Figure` instance.
- `ax` : A two-dimensional `matplotlib` `Axes` instance. 
- `scale` : Scalar value applied to each vector. See notes below for more details.
- `density` : A measure of how many vectors to plot within the field. An evenly-spaced grid of `density`\*`density` vectors is plotted on the axes.
  - Value must be within range `4 <= density <= 100`. 
- `cmap` : A `matplotlib` colormap applied to the field. Can pass a built-in or custom colormap. Built-in colormap options can be found [here.](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
- `cmap_func` : One of the three `VectorField` properties to use in determination of the color mapping.
  - `'mag'` : Maps colors to the vectors based on their respective magnitudes. 
  - `'div'` : Maps colors to the vectors based on the divergence of the vectors' initial positions. 
  - `'curl'` : Maps colors to the vectors based on the curl of the vectors' initial positions. 
- `normalize` : Option to normalize each vector. See notes below for more details.
- `colorbar` : Option to display a colorbar of the color mapping.
  - The values shown, both on the colorbar ticks and the colorbar's label, will vary depending on the chosen `scale`, `density`, and `cmap_func`.
- `interactive` : Option to make the vector field interactable. The plot will detect mouse clicks, and sliders will be added below the `ax`. 
  - Clicking and holding on a point within the `ax` will display an annotation describing the curl, divergence and magnitude at that point. 
  - The scale and density sliders allow for adjustment of the field, and function just as the `scale` and `density` parameters do.
    - The upper-bound of the scale slider is calculated using the x and y limits of the `ax`:
    - `round(max(abs(val) for val in (xlim + ylim)) / 4)`
    - Values less than 1 will be rounded to 1. 
- `kwargs` : Additional arguments passed to `~Axes.quiver()`. Options can be found [here.](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.quiver.html)

Example:
```python
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from vgtk import VectorField

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=10)

F = VectorField('x', 'y')

F.plot(fig, ax)

ax.set_title(F.get_latex_str(True))
plt.show()
```
Out:

<p align="center"> 
  <img src=examples/ex_plot3.png>
</p>


Example:
```python
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from vgtk import VectorField

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=3)

F = VectorField.from_grad('x*exp(-(x**2 + y**2))')

F.plot(fig, ax, cmap='inferno', interactive=True)

plt.show()
```
[Video demonstration of this example.](examples/ex_plot4.mp4?raw=true)

Notes:

When `normalize=True`, all vectors will be converted into their unit vector form (their scalars get divided by their magnitude). The `scale` argument is then applied *after*. An auto-scaling algorithm could be implemented in the future, since extremely small figures will have huge vectors and extremely large figures will have small vectors without adjusting the `scale` from its default, `1`. 

Like the [`Vector` class's `plot()` method](#plot), `~Axes.quiver()` is called with arguments `units='xy'` and `scale=1` to prevent warping and auto-scaling from `matplotlib`, and has the consequence of the user not being able to call these parameters in `kwargs`. This could be subject to change in the future, since `matplotlib` could be left to take care of the issue described above. 

### `particles`
Animates particles on a given `matplotlib` `Axes`, where relative velocities are modeled by the field. Returns the created `matplotlib.animation.FuncAnimation` instance. 
```python
particles(self, fig:Figure, ax:Axes, pts:Iterable[tuple]=None, frames:int=300, dt:float=0.01, fmt:str='o', color:str='k', alpha:float=0.7, **kwargs) -> FuncAnimation
```
- `fig` : A `matplotlib` `Figure` instance.
- `ax` : A two-dimensional `matplotlib` `Axes` instance. 
- `pts` : An array of coordinate pairs that set the initial particle positions.
  - If `None`, 50 randomly-placed particles will be plotted. 
- `frames` : The amount of frames to run the animation for.
- `dt` : The change in time from one frame to the next. 
  - I recommend keeping this value extremely small. Larger values may result in ludicrously fast particle speeds. 
- `fmt` : Marker style of the particles. Argument passed to `~Axes.plot()`. Options can be found [here.](https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.axes.Axes.plot.html)
- `color` : Color of the particles. Argument passed to `~Axes.plot()`. Options can be found [here.](https://matplotlib.org/3.1.0/gallery/color/named_colors.html)
- `alpha` : Transparency of the particles. Argument passed to `~Axes.plot()`. 
- `kwargs` : Additional arguments passed to `~Axes.plot()`. Options can be found [here.](https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.axes.Axes.plot.html)

Example:
```python
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn; seaborn.set()
from vgtk import VectorField

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=2)

F = VectorField.from_grad('10*x*exp(-(x**2 + y**2))')

F.plot(fig, ax, scale=0.1, density=25, cmap='viridis')
ax.set_title(F.get_latex_str(True))

ani = F.particles(fig, ax)

# ani.save('ex_plot5.gif', writer=animation.PillowWriter(fps=30))
plt.show()
```
Out:

<p align="center"> 
  <img src=examples/ex_plot5.gif>
</p>

Example:
```python
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn; seaborn.set()
from vgtk import VectorField

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=10)

F = VectorField('-y', 'x')

F.plot(fig, ax, density=15, cmap='plasma')
ax.set_title(F.get_latex_str(True))

ani = F.particles(fig, ax)

# ani.save('ex_plot6.gif', writer=animation.PillowWriter(fps=30))
plt.show()
```
Out:

<p align="center"> 
  <img src=examples/ex_plot6.gif>
</p>

Notes:

The velocities of the particles are relative. If the `plot()` method is active, its `scale` parameter *will* affect the speed of the particles. Additionally, if `interactive=True`, particle velocities will reflect the value set by the scale slider. The particle simulation will slow down significantly -- unfortunately because `matplotlib` isn't the greatest at handling animations. When `interactive=False`, however, the animation will be [blitted](https://en.wikipedia.org/wiki/Bit_blit), and perform considerably faster and smoother.

### `get_latex_str`
Returns a `str` of the `VectorField` instance in LaTeX formatting.
```python
get_latex_str(self, notation:Union['angled', 'parentheses', 'unit']='angled') -> str
```
- `notation` : Changes notation style of the string.
  - `'angled'` : [Ordered set notation, angle-bracket variant.](https://en.wikipedia.org/wiki/Vector_notation#Ordered_set_notation)
  - `'parentheses'` : [Ordered set notation, parentheses variant.](https://en.wikipedia.org/wiki/Vector_notation#Ordered_set_notation)
  - `'unit'` : [Unit vector notation.](https://en.wikipedia.org/wiki/Vector_notation#Unit_vector_notation)

Example:
```python
from vgtk import VectorField

F = VectorField('x*y', 'y**2 - x**2')

angled = F.get_latex_str('angled')
parentheses = F.get_latex_str('parentheses')
unit = F.get_latex_str('unit')

print(angled)
print(parentheses)
print(unit)
```
Out:
```
$\vec{F} = <x y, - x^{2} + y^{2}>$
$\vec{F} = (x y, - x^{2} + y^{2})$
$\vec{F} = (x y)\hat{i} + (- x^{2} + y^{2})\hat{j}$
```

Back to [table of contents](#table-of-contents).

___

## Credits

Author: Braedyn Lettinga

Contributors: Ashu Acharya

Credits: Heiko Hergert, Ph.D., Tony S. Yu, Ph.D.

Contribution resources are being worked on at the moment.  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
