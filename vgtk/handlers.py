from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import odeint

np.warnings.filterwarnings('ignore')


class _MPLPlate(object):
    '''
    This class is a background plate used for all children
    class handlers. 

    Parameters
    ----------
        fig : A matplotlib.figure.Figure instance.
        ax : A matplotlib.axes.Axes instance. 
    '''

    def __init__(self, fig:Figure, ax:Axes):
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect('equal')

        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

        bbox = {'boxstyle': 'round', 'fc': 'w', 'pad': 0.4, 'alpha': 0.7}
        self.annotation = self.ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points', bbox=bbox)
        self.annotation.set_visible(False)


class _VectorEventHandler(_MPLPlate):

    def __init__(self, fig:Figure, ax:Axes, x0:float, y0:float, u:float, v:float, name:str, trace_scalars:bool, interactive:bool, **quiver_kwargs):
        
        super().__init__(fig, ax)

        self.x0 = x0
        self.y0 = y0
        self.x1 = x0 + u
        self.y1 = y0 + v
        self.u = u
        self.v = v
        self.quiver_kwargs = quiver_kwargs

        self.quiver = self.ax.quiver(self.x0, self.y0, self.u, self.v, **self.quiver_kwargs)

        self.trace_state = trace_scalars

        if self.trace_state:
            self.xtrace = self.ax.plot((self.x0, self.x1), (self.y0, self.y0), linestyle='--', color='C0')
            self.ytrace = self.ax.plot((self.x1, self.x1), (self.y0, self.y1), linestyle='--', color='C1')

        if interactive:
            self.dragging_point = False
            self.dragging_vector = False

            self.drag_point = self.ax.scatter(self.x1, self.y1, color='grey', alpha=0.7)
            self.annotation_text = '$\\angle x = {:.2f}^c, \\angle y = {:.2f}^c$\n$x_0 = {:.2f}, y_0 = {:.2f}$\n$x_1 = {:.2f}, y_1 = {:.2f}$\n$u = {:.2f}, v = {:.2f}$\n$mag(\\vec{{' + name + '}}) = {:.2f}$'

            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_quiver(self, x0:float, y0:float, u:float, v:float) -> None:
        '''
        Updates the drawn quiver and subsequently re-draws the figure.

        Parameters
        ----------
            x0 : x-coordinate of the vector.
            y0 : y-coordinate of the vector.
            u : u scalar of the vector.
            v : v scalar of the vector.
        
        Notes
        -----
            The quiver is removed from the figure entirely and re-drawn. The 
            .set_UVC() method does not work here, since the vector's position 
            will be changing, and .set_UVC() only configures the u and v
            scalars.
        '''
        self.drag_point.set_offsets([x0 + u, y0 + v])
        self.quiver.remove()
        self.quiver = self.ax.quiver(x0, y0, u, v, **self.quiver_kwargs)
        self.fig.canvas.draw_idle()

    def update_annotation(self, xdata:float, ydata:float) -> None:
        '''
        Updates the annotation with the given x and y data from the mouse pointer's location.

        Parameters
        ----------
            xdata : Mouse pointer's x position.
            ydata : Mouse pointer's y position.
        '''
        self.annotation.xy = (xdata, ydata)
        self.annotation.set_text(
            self.annotation_text.format(
                np.arctan(self.v / self.u),
                np.arctan(self.u / self.v),
                self.x0,
                self.y0,
                self.x1,
                self.y1,
                self.u,
                self.v,
                np.linalg.norm([self.u, self.v]),
            )
        )
        self.annotation.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_click(self, event:MouseEvent) -> None:
        '''
        Calculates new scalars and initial positions in the event of a drag.

        Parameters
        ----------
            event : Mouse event object emitted by ~Figure.canvas.mpl_connect().
        '''
        if self.drag_point.contains(event)[0] and event.inaxes == self.ax:
            if event.button == 1:
                self.dragging_point = True
            elif event.button == 3:
                self.dragging_vector = True
            self.update_annotation(event.xdata, event.ydata)
    
    def on_motion(self, event:MouseEvent) -> None:
        '''
        Calculates new scalars and initial positions in the event of a drag.

        Parameters
        ----------
          event : Mouse event object emitted by ~Figure.canvas.mpl_connect().
        '''
        if event.inaxes == self.ax and event.button != 2 and (self.dragging_point or self.dragging_vector):
            self.x1 = event.xdata
            self.y1 = event.ydata

            if self.dragging_point:
                self.u = self.x1 - self.x0
                self.v = self.y1 - self.y0
            elif self.dragging_vector:
                self.x0 = self.x1 - self.u
                self.y0 = self.y1 - self.v

            if self.trace_state:
                self.xtrace[0].set_data((self.x0, self.x1), (self.y0, self.y0))
                self.ytrace[0].set_data((self.x1, self.x1), (self.y0, self.y1))

            self.update_annotation(event.xdata, event.ydata)
            self.update_quiver(self.x0, self.y0, self.u, self.v)    
    
    def on_release(self, event:MouseEvent) -> None:
        '''
        Halts all warping/shifting of the vector if the mouse button has been released. Additionally
        hides annotation. 

        Parameters
        ----------
            event : Mouse event object emitted by ~Figure.canvas.mpl_connect().
        '''
        if event.inaxes == self.ax:
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()

            if self.drag_point.contains(event)[0] and event.button != 2:
                self.dragging_point = False
                self.dragging_vector = False

                self.x1 = event.xdata
                self.y1 = event.ydata
                self.u = self.x1 - self.x0
                self.v = self.y1 - self.y0

                self.update_quiver(self.x0, self.y0, self.u, self.v)


class _VectorFieldEventHandler(_MPLPlate):

    def __init__(self, fig:Figure, ax:Axes, u:Callable[[float, float], float], v:Callable[[float, float], float], 
                 mag:Callable[[float, float], float], div:Callable[[float, float], float], curl:Callable[[float, float], float],
                 name:str, mscale:float, density:int, normalize:bool, colorbar:bool, interactive:bool, **quiver_kwargs):
        
        super().__init__(fig, ax)

        self.u = u
        self.v = v

        self.mag = mag 
        self.div = div 
        self.curl = curl

        self.scale = mscale
        self.quiver_kwargs = quiver_kwargs

        self.normalize_state = normalize
        self.colorbar_state = colorbar

        components = self.create_field(self.scale, density)

        self.quiver = self.ax.quiver(*components, **self.quiver_kwargs)

        if self.colorbar_state:
            plt.subplots_adjust(left=0.05)
            c = components[-1]

            if self.normalize_state:
                self.colorbar_label = 'Sampled Magnitude (Scale: {:.2f}, Normalized)'
            else:
                self.colorbar_label = 'Sampled Magnitude (Scale: {:.2f})'

            cax = inset_axes(
                self.ax,
                width='5%',
                height='100%',
                loc='lower left',
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=self.ax.transAxes,
                borderpad=0,
                axes_kwargs={'zorder': -1}
            )
            
            self.colorbar = plt.colorbar(
                self.quiver,
                cax=cax,
                cmap=self.quiver_kwargs['cmap'],
                label=self.colorbar_label.format(self.scale),
                ticks=np.linspace(c.min(), c.max(), 5)
            )

        if interactive:
            self.annotation_text = '$x = {:.2f}, y = {:.2f}$\n$mag(\\vec{{' + name + '}})(x, y) = {:.2f}$\n$div(\\vec{{' + name + '}})(x, y) = {:.2f}$\n$curl(\\vec{{' + name + '}})(x, y) = {:.2f}$'

            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('button_release_event', self.on_release)

            scale_range = round(max(abs(val) for val in (self.xlim + self.ylim)) / 4)
            if scale_range < 1: scale_range = 1

            divider = make_axes_locatable(self.ax)

            scale_ax = divider.append_axes('bottom', size='3%', pad=0.6) 
            self.scale_slider = Slider(scale_ax, 'Scale', -scale_range, scale_range, valinit=mscale)
            self.scale_slider.on_changed(self.slider_update)

            density_ax = divider.append_axes('bottom', size='3%', pad=0.1)
            self.density_slider = Slider(density_ax, 'Density', 4, 100, valinit=density, valstep=1)
            self.density_slider.on_changed(self.slider_update)

    def on_release(self, event:MouseEvent) -> None:
        '''
        Clears annotation on release of the mouse button.

        Parameters
        ----------
            event : Mouse event emitted by ~Figure.canvas.mpl_connect()
        '''
        if event.inaxes == self.ax:
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()

    def on_click(self, event:MouseEvent) -> None:
        '''
        Displays and updates annotation on mouse click.

        Parameters
        ----------
            event : Mouse event emitted by ~Figure.canvas.mpl_connect()
        '''
        if event.inaxes == self.ax:
            self.annotation.xy = (event.xdata, event.ydata)
            self.annotation.set_text(
                self.annotation_text.format(
                    event.xdata, 
                    event.ydata, 
                    self.mag(event.xdata, event.ydata), 
                    self.div(event.xdata, event.ydata), 
                    self.curl(event.xdata, event.ydata),
                    )
                )
            self.annotation.set_visible(True)
            self.fig.canvas.draw_idle()

    def slider_update(self, _) -> None:
        '''
        Updates and re-draws field plot based on the scale and density slider values.
        '''

        components = self.create_field(self.scale_slider.val, self.density_slider.val)

        self.quiver.remove()
        self.quiver = self.ax.quiver(*components, **self.quiver_kwargs)

        if self.colorbar_state:
            self.colorbar.set_label(self.colorbar_label.format(self.scale_slider.val))

        self.fig.canvas.draw_idle()


    def create_field(self, scale:float, density:int) -> tuple:
        '''
        Calculates all vector and vector positions with a given scale and density.

        Parameters
        ----------
            scale : Scale value applied to vector lengths.
            density : Density of the field. 
        
        Returns
        -------
            tuple : A tuple of ndarrays, consisting of the x, y, u, v and c
                    values of the field respectively.
        '''

        x, y = np.meshgrid(
            np.linspace(*self.xlim, int(density)),
            np.linspace(*self.ylim, int(density))
        )

        u, v = (
            self.u(x, y),
            self.v(x, y)
        )

        m = np.sqrt(u**2 + v**2)

        if self.normalize_state:
            with np.errstate(all='ignore'):
                u = (u / m) * scale
                v = (v / m) * scale
        else:
            u = u * scale
            v = v * scale

        return x, y, u, v, m


class _ParticleSimulationHandler(_MPLPlate):

    def __init__(self, fig:Figure, ax:Axes, u:Callable[[float, float], float], v:Callable[[float, float], float], pts:Iterable[tuple], 
                 frames:int, dt:float, blit:bool, fmt:str, color:str, alpha:float, **plot_kwargs):
        
        super().__init__(fig, ax)

        self.u = u 
        self.v = v

        self.ln, = self.ax.plot([], [], fmt, color=color, alpha=alpha, **plot_kwargs)

        if pts is None: 
            pts = np.array((np.random.uniform(*self.xlim, 50), np.random.uniform(*self.ylim, 50))).transpose()

        self.ani = FuncAnimation(self.fig, self.particle_update, interval=1, frames=frames, blit=blit, fargs=(pts, dt))

    def solve_ode(self, f:Callable, pts:Iterable[tuple], dt:float) -> list:
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
        return [odeint(f, pt, [0, dt])[-1] for pt in pts]

    def get_vels(self, pt:tuple, _) -> list:
        '''
        Calculates the velocity of the particle at a specific point.

        Parameters
        ----------
            pt : An x, y coordinate pair.
            _ : Dummy time parameter required for odeint(). 
                - Unused since scalar functions don't depend on time.
        
        Returns
        -------
            list : The x and y velocity at the given point (index 0 and index 1, respectively).
        '''
        return [self.u(*pt), self.v(*pt)]

    def remove_pts(self, pts:np.ndarray) -> np.ndarray:
        '''
        Removes points that are outside the bounds of the axes. 

        Parameters
        ----------
            pts : Array of coordinate pairs.
        
        Returns
        -------
            numpy.ndarray : The same array but with out-of-boundary points removed.
        
        Credits
        -------
            Tony S. Yu, Ph.D.
        '''
        if len(pts) == 0: return []

        out_x = (pts[:, 0] < self.xlim[0]) | (pts[:, 0] > self.xlim[1])
        out_y = (pts[:, 1] < self.ylim[0]) | (pts[:, 1] > self.ylim[1])
        keep = ~(out_x | out_y)

        return pts[keep]

    def particle_update(self, frame:int, *fargs) -> Line2D:
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
        # self.pts needs to be a data member, though it may not look like it at first.
        # transition arrays would be reset back to the initial if not a data member. 

        if frame == 0: self.pts = fargs[0]

        self.pts = np.asarray(self.solve_ode(self.get_vels, self.pts, fargs[1]))
        self.pts = np.asarray(self.remove_pts(self.pts))

        self.pts.shape = (self.pts.shape[0], 2)  # rudimentary way of ensuring .transpose() runs properly
        x, y = self.pts.transpose()

        self.ln.set_data(x, y)

        return self.ln,
