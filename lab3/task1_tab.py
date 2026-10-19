import tkinter as tk
from tkinter import ttk

import analysis
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import report
from params import Params
from sampling import generate_pairs

DEFAULT_N = 10_000
BINS = 20
CLASSES = 20 
SLICES = 3 
SCATTER_POINTS = 2_000
X_RANGE = (0.0, 6.0)
Y_RANGE = (0.0, 6.0)
MIN_EXPECTED = 5


class Task1Tab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.rng = np.random.default_rng()

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(controls, text="n:").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value=str(DEFAULT_N))
        ttk.Entry(controls, textvariable=self.n_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Generate", command=self.run).pack(side=tk.LEFT, padx=4)
        self.warning = ttk.Label(controls, foreground="#b00020")
        self.warning.pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.report = tk.Text(body, width=58, font=("Consolas", 10), state=tk.DISABLED)
        self.report.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        self.run()

    def run(self):
        try:
            n = int(self.n_var.get())
        except ValueError:
            self.warning.config(text="n must be an integer")
            return
        if n < 4:
            self.warning.config(text="n must be at least 4")
            return

        x, y = generate_pairs(n, self.rng)

        edges_x = Params.cdf_x_inv(np.arange(1, CLASSES) / CLASSES)
        edges_y = Params.cdf_y_inv(np.arange(1, CLASSES) / CLASSES)
        test = analysis.independence_test(x, y, edges_x, edges_y)

        if test.min_expected < MIN_EXPECTED:
            self.warning.config(
                text=f"n/(r*c) = {test.min_expected:.1f} < {MIN_EXPECTED}: chi-square is unreliable"
            )
        else:
            self.warning.config(text="")

        self.draw(x, y)
        self.write_report(x, y, test)

    def draw(self, x, y):
        self.figure.clear()
        ax_x, ax_y, ax_cond, ax_scatter = self.figure.subplots(2, 2).flat

        for ax, sample, density, label, (low, high) in (
            (ax_x, x, Params.f_x, "x", X_RANGE),
            (ax_y, y, Params.f_y, "y", Y_RANGE),
        ):
            heights, edges = analysis.density_histogram(sample, low, high, BINS)
            ax.bar(edges[:-1], heights, width=np.diff(edges), align="edge", alpha=0.5, label="histogram")
            grid = np.linspace(low, high, 400)
            ax.plot(grid, density(grid), color="C3", label=f"f_{label.upper()}")
            ax.set_title(f"Component {label.upper()}")
            ax.legend()

        slice_of_x = analysis.class_indices(x, Params.cdf_x_inv(np.arange(1, SLICES) / SLICES))
        grid = np.linspace(*Y_RANGE, 400)
        for s in range(SLICES):
            ys = y[slice_of_x == s]
            if len(ys) == 0:
                continue
            heights, edges = analysis.density_histogram(ys, *Y_RANGE, 25)
            centers = (edges[:-1] + edges[1:]) / 2
            ax_cond.plot(centers, heights, ".", color=f"C{s}", label=f"empirical, slice {s + 1}")
            x_mid = Params.cdf_x_inv(np.array([(s + 0.5) / SLICES]))
            ax_cond.plot(grid, Params.f_y_given_x(grid, x_mid), color=f"C{s}", alpha=0.6)
        ax_cond.set_title("f(y|x) for x-slices")
        ax_cond.legend()

        ax_scatter.plot(x[:SCATTER_POINTS], y[:SCATTER_POINTS], ".", markersize=2)
        ax_scatter.set_xlim(*X_RANGE)
        ax_scatter.set_ylim(*Y_RANGE)
        ax_scatter.set_title(f"Sample (first {min(SCATTER_POINTS, len(x))} points)")

        self.canvas.draw()

    def write_report(self, x, y, test):
        lines = [f"n = {len(x)}", ""]
        lines += report.component_lines("X", x, Params.MEAN_X, Params.VAR_X)
        lines += report.component_lines("Y", y, Params.MEAN_Y, Params.VAR_Y)
        lines += report.correlation_lines(x, y, Params.RHO)
        lines += report.independence_lines(f"Independence (chi-square, {CLASSES}x{CLASSES} classes):", test)
        lines += [
            "Conditional density: f(y|x) = f(x,y)/f_X(x) = exp(-y)",
            "  does not depend on x.",
        ]

        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, "\n".join(lines))
        self.report.config(state=tk.DISABLED)
