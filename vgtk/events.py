from typing import Callable, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import odeint

np.warnings.filterwarnings('ignore')

FUNC_LABELS = {'mag' : 'Magnitude', 'div' : 'Divergence', 'curl': 'Curl'}


class MPLPlate(object):

    def __init__(self, fig:Figure, ax:Axes):
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect('equal')

        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

        bbox = {'boxstyle': 'round', 'fc': 'w', 'pad': 0.4, 'alpha': 0.7}
        self.annotation = self.ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points', bbox=bbox)
        self.annotation.set_visible(False)


class VectorEventHandler(MPLPlate):

    def __init__(self, fig:Figure, ax:Axes, x0:float, y0:float, u:float, v:float, name:str, trace_scalars:bool, interactive:bool, **quiver_kwargs):
        MPLPlate.__init__(self, fig, ax)

        self.trace_state = trace_scalars

        self.x0 = x0
        self.y0 = y0
        self.x1 = x0 + u
        self.y1 = y0 + v
        self.u = u
        self.v = v
        self.quiver_kwargs = quiver_kwargs

        self.quiver = self.ax.quiver(self.x0, self.y0, self.u, self.v, **self.quiver_kwargs)

        if self.trace_state:
            self.xtrace = self.ax.plot((self.x0, self.x1), (self.y0, self.y0), linestyle='--', color='C0')
            self.ytrace = self.ax.plot((self.x1, self.x1), (self.y0, self.y1), linestyle='--', color='C1')

        if interactive:
            self.dragging_point = False
            self.dragging_vector = False

            self.drag_point = self.ax.scatter(self.x1, self.y1, color='grey', alpha=0.7)
            self.annotation_text = '$\\angle x = {:.2f}^c, \\angle y = {:.2f}^c$\n$x_0 = {:.2f}, y_0 = {:.2f}$\n$x_1 = {:.2f}, y_1 = {:.2f}$\n$u = {:.2f}, v = {:.2f}$\n$mag(\\vec{{' + name +  '}}) = {:.2f}$'

            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_quiver(self, x0:float, y0:float, u:float, v:float) -> None:
        self.drag_point.set_offsets([x0 + u, y0 + v])
        self.quiver.remove()
        self.quiver = self.ax.quiver(x0, y0, u, v, **self.quiver_kwargs)
        self.fig.canvas.draw_idle()

    def update_annotation(self, xdata:float, ydata:float) -> None:
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
        if self.drag_point.contains(event)[0] and event.inaxes == self.ax:
            if event.button == 1:
                self.dragging_point = True
            elif event.button == 3:
                self.dragging_vector = True
            self.update_annotation(event.xdata, event.ydata)
    
    def on_motion(self, event:MouseEvent) -> None:
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


class VectorFieldEventHandler(MPLPlate):

    def __init__(self, fig:Figure, ax:Axes, u:Callable[[float, float], float], v:Callable[[float, float], float], mag:Callable[[float, float], float], curl:Callable[[float, float], float], div:Callable[[float, float], float], name:str, mscale:float, density:int, cmap_func:Union['mag', 'div', 'curl'], normalize:bool, colorbar:bool, interactive:bool, **quiver_kwargs):
        
        if not 4 <= density <= 100:
            raise ValueError('density argument must be within range [4, 100]')
        if cmap_func not in FUNC_LABELS:
            raise ValueError('cmap_func, "{}", not recognized -- possible options are: "mag", "curl", "div"'.format(cmap_func))
        
        MPLPlate.__init__(self, fig, ax)

        self.u = u
        self.v = v
        self.mag = mag
        self.curl = curl
        self.div = div

        self.cmap_func = cmap_func
        self.quiver_kwargs = quiver_kwargs

        self.normalize_state = normalize
        self.colorbar_state = colorbar

        components = self.create_field(mscale, density)

        self.quiver = self.ax.quiver(*components, **self.quiver_kwargs)

        if self.colorbar_state:
            plt.subplots_adjust(left=0.05)
            c = components[-1]
            label = FUNC_LABELS[self.cmap_func]

            if self.normalize_state:
                self.colorbar_label = 'Sampled ' + label + ' (Scale: {:.2f}, Normalized)'
            else:
                self.colorbar_label = 'Sampled ' + label + ' (Scale: {:.2f})'

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
                label=self.colorbar_label.format(mscale),
                ticks=np.linspace(c.min(), c.max(), 5)
            )

        if interactive:
            self.annotation_text = '$x = {:.2f}, y = {:.2f}$\n$mag(\\vec{{' + name + '}})(x, y) = {:.2f}$\n$div(\\vec{{' + name + '}})(x, y) = {:.2f}$\n$curl(\\vec{{' + name + '}})(x, y) = {:.2f}$'

            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('button_release_event', self.on_release)

            scale_range = round(max(abs(val) for val in (self.xlim + self.ylim)) / 4)
            if scale_range < 1:
                scale_range = 1

            divider = make_axes_locatable(self.ax)

            scale_ax = divider.append_axes('bottom', size='3%', pad=0.6) 
            self.scale_slider = Slider(scale_ax, 'Scale', -scale_range, scale_range, valinit=mscale)
            self.scale_slider.on_changed(self.slider_update)

            density_ax = divider.append_axes('bottom', size='3%', pad=0.1)
            self.density_slider = Slider(density_ax, 'Density', 4, 100, valinit=density, valstep=1)
            self.density_slider.on_changed(self.slider_update)

    def on_release(self, event:MouseEvent) -> None:
        if event.inaxes == self.ax:
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()

    def on_click(self, event:MouseEvent) -> None:
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
        components = self.create_field(self.scale_slider.val, self.density_slider.val)

        self.quiver.remove()
        self.quiver = self.ax.quiver(*components, **self.quiver_kwargs)

        if self.colorbar_state:
            self.colorbar.set_label(self.colorbar_label.format(self.scale_slider.val))

        self.fig.canvas.draw_idle()


    def create_field(self, scale:float, density:int) -> tuple:
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

        if self.cmap_func == 'mag':
            c = m
        elif self.cmap_func == 'div':
            c = self.div(x, y)
        elif self.cmap_func == 'curl':
            c = self.curl(x, y)
        
        c = np.full_like(x, c)

        return x, y, u, v, c