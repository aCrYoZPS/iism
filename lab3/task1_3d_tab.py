import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import analysis
from params import Params
from sampling import generate_pairs

DEFAULT_N = 10_000
BINS = 20
X_RANGE = (0.0, 6.0)
Y_RANGE = (0.0, 6.0)
SURFACE_POINTS = 60


class Task1Tab3D(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.rng = np.random.default_rng()

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(controls, text="n:").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value=str(DEFAULT_N))
        ttk.Entry(controls, textvariable=self.n_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Generate", command=self.run).pack(side=tk.LEFT, padx=4)
        self.info = ttk.Label(controls)
        self.info.pack(side=tk.LEFT, padx=12)

        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.run()

    def run(self):
        try:
            n = int(self.n_var.get())
        except ValueError:
            self.info.config(text="n must be an integer")
            return
        if n < 1:
            self.info.config(text="n must be positive")
            return

        x, y = generate_pairs(n, self.rng)
        heights, x_edges, y_edges = analysis.density_histogram_2d(x, y, X_RANGE, Y_RANGE, BINS)

        inside = heights.sum() * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        self.info.config(text=f"share of points outside {X_RANGE}x{Y_RANGE}: {1.0 - inside:.2%}")

        self.draw(heights, x_edges, y_edges)

    def draw(self, heights, x_edges, y_edges):
        self.figure.clear()
        ax = self.figure.add_subplot(projection="3d")

        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        xs, ys = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
        ax.bar3d(xs.ravel(), ys.ravel(), 0.0, dx, dy, heights.ravel(), alpha=0.5, shade=True)

        gx, gy = np.meshgrid(
            np.linspace(*X_RANGE, SURFACE_POINTS), np.linspace(*Y_RANGE, SURFACE_POINTS), indexing="ij"
        )
        ax.plot_wireframe(gx, gy, Params.f(gx, gy), color="C3", linewidth=0.5, rstride=2, cstride=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("density")
        ax.set_title("2D histogram (bars) and f(x, y) (red grid)")

        self.canvas.draw()
