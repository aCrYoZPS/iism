import tkinter as tk
from tkinter import ttk

from task1_3d_tab import Task1Tab3D
from task1_tab import Task1Tab
from task2_tab import Task2Tab


def main():
    root = tk.Tk()
    root.title("Lab 3: systems of random variables")
    root.geometry("1400x760")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    notebook.add(Task1Tab(notebook), text="Continuous 2D")
    notebook.add(Task1Tab3D(notebook), text="Continuous 3D")
    notebook.add(Task2Tab(notebook), text="Discrete 2D")

    root.mainloop()


if __name__ == "__main__":
    main()
