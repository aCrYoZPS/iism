import tkinter as tk
from tkinter import ttk

import analysis
import numpy as np
import report
from discrete_params import MATRICES, VALUES_X, VALUES_Y, characteristics, marginals
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sampling import generate_pairs_discrete

DEFAULT_N = 10_000
MIN_EXPECTED = 5


class Task2Tab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.rng = np.random.default_rng()

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(controls, text="Matrix:").pack(side=tk.LEFT)
        self.matrix_var = tk.StringVar(value=next(iter(MATRICES)))
        matrix_box = ttk.Combobox(
            controls, textvariable=self.matrix_var, values=list(MATRICES), state="readonly", width=12
        )
        matrix_box.pack(side=tk.LEFT, padx=4)
        matrix_box.bind("<<ComboboxSelected>>", lambda _event: self.run())
        ttk.Label(controls, text="n:").pack(side=tk.LEFT, padx=(12, 0))
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

        matrix = MATRICES[self.matrix_var.get()]
        x, y = generate_pairs_discrete(n, matrix, VALUES_X, VALUES_Y, self.rng)
        observed = analysis.contingency_table(x, y, VALUES_X, VALUES_Y)
        test = analysis.independence_test_table(observed)

        if test is None:
            self.warning.config(text="sample has a single value of x or y: no table to test")
        elif test.min_expected < MIN_EXPECTED:
            self.warning.config(
                text=f"min expected count {test.min_expected:.1f} < {MIN_EXPECTED}: chi-square is unreliable"
            )
        else:
            self.warning.config(text="")

        self.draw(matrix, observed)
        self.write_report(matrix, x, y, test)

    def draw(self, matrix, observed):
        self.figure.clear()
        ax_x = self.figure.add_subplot(2, 2, 1)
        ax_y = self.figure.add_subplot(2, 2, 2)
        ax_cond = self.figure.add_subplot(2, 2, 3)
        ax_3d = self.figure.add_subplot(2, 2, 4, projection="3d")

        frequencies = observed / observed.sum()
        p_x, p_y = marginals(matrix)

        for ax, values, empirical, theory, label in (
            (ax_x, VALUES_X, frequencies.sum(axis=1), p_x, "X"),
            (ax_y, VALUES_Y, frequencies.sum(axis=0), p_y, "Y"),
        ):
            ax.bar(values, empirical, alpha=0.5, label="frequency")
            ax.plot(values, theory, "o", color="C3", label=f"p({label})")
            ax.set_title(f"Component {label}")
            ax.set_xticks(values)
            ax.legend()

        # P(Y = y_j | X = x_i): one line per value of X.
        row_counts = observed.sum(axis=1, keepdims=True)
        for i, x_i in enumerate(VALUES_X):
            if row_counts[i, 0] == 0:
                continue
            ax_cond.plot(VALUES_Y, observed[i] / row_counts[i, 0], "-o", color=f"C{i}", label=f"x = {x_i:g}")
            ax_cond.plot(VALUES_Y, matrix[i] / p_x[i], "--", color=f"C{i}", alpha=0.6)
        ax_cond.set_title("P(y | x): empirical (solid), theory (dashed)")
        ax_cond.set_xticks(VALUES_Y)
        ax_cond.legend(fontsize=7)

        xs, ys = np.meshgrid(VALUES_X, VALUES_Y, indexing="ij")
        ax_3d.bar3d(xs.ravel() - 0.4, ys.ravel() - 0.4, 0.0, 0.8, 0.8, frequencies.ravel(), alpha=0.5, shade=True)
        ax_3d.scatter(xs.ravel(), ys.ravel(), matrix.ravel(), color="C3", s=12, label="p_ij")
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_title("Frequencies (bars), p_ij (red)")

        self.canvas.draw()

    def write_report(self, matrix, x, y, test):
        mean_x, mean_y, var_x, var_y, rho = characteristics(matrix)

        lines = [f"Matrix: {self.matrix_var.get()}, n = {len(x)}", ""]
        lines += report.component_lines("X", x, mean_x, var_x)
        lines += report.component_lines("Y", y, mean_y, var_y)
        lines += report.correlation_lines(x, y, rho)
        if test is not None:
            lines += report.independence_lines("Independence (chi-square, sample marginals):", test)

        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, "\n".join(lines))
        self.report.config(state=tk.DISABLED)
