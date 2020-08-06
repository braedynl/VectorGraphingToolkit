import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from matplotlib.animation import PillowWriter
from sympy.abc import x, y
from vgtk import VectorField

import seaborn; seaborn.set()

def cylinder(U=1, R=1):
    r = sym.sqrt(x**2 + y**2)
    psi = U * (r - R**2 / r) * sym.sin(sym.atan2(y, x))
    return psi

def settings(lim):
    fig, ax = plt.subplots()
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    return fig, ax

fig, ax = settings(lim=3)
ax.set_title('Potential Flow Around a Circular Cylinder')
ax.add_patch(plt.Circle((0, 0), radius=1, fc='k'))

pts = np.array((np.full(20, -3), np.random.uniform(-3, 3, 20))).transpose()

F = VectorField.from_stream(cylinder())
F.plot(fig, ax, scale=0.25, density=20, colorbar=False, cmap='Blues_r')
ani = F.particles(fig, ax, pts=pts, frames=800, color='k')

# ani.save('ex_intro.gif', writer=PillowWriter(fps=30))
plt.show()
