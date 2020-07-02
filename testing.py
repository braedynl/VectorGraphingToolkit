import matplotlib.pyplot as plt
from vgtk.vgtk import Vector, VectorField

fig, ax = plt.subplots()
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)

F = VectorField('x', 'y')

F.plot(fig, ax, interactive=True)

plt.show()