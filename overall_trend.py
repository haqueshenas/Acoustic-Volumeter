#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Overall Trend Modeling Module
# Part of the Easy Phenotyping Lab (EPL)
# https://haqueshenas.github.io/EPL
#
# Copyright (C) 2025 Abbas Haghshenas
# Licensed under the MIT License
#
# Author:      Abbas Haghshenas <haqueshenas@gmail.com>
# Affiliation: Independent Researcher in Crop Production, Shiraz, Iran
# Version:     1.0
# This code is written with assistance of ChatGPT 4o, OpenAI.
# -----------------------------------------------------------------------------

"""
overall_trend_modeling.py

Build calibration curves mapping resonance peak frequency → known volume:
  - Upload JSON peak files + enter actual volumes
  - Fit Linear, Quadratic, Cubic, Logarithmic, and piecewise‐linear models
  - Preview scatter + overlayed fits, display equations
  - Export coefficients to JSON and plot to PNG

Usage:
    python overall_trend_modeling.py
"""

import tkinter as tk
import os
from tkinter import filedialog, messagebox, ttk
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import sys, os

def resource(rel_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller one-file.
    """
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

class OverallTrendModeling:
    """
    GUI for creating and exporting overall‐trend calibration models.
    """
    def __init__(self, root):
        """
        Initialize the window, center it on screen, set icon/size,
        and call setup_gui() to build all widgets.
        """
        self.root = root
        self.root.withdraw()
        self.root.title("Overall Trend Modeling - Acoustic Volumeter v. 1.0")
        self.root.geometry("1200x550")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.root.deiconify()

        self.root.iconbitmap(resource("AV.ico"))
        self.root.resizable(width=False, height=False)
        self.json_files = [None] * 10
        self.actual_volumes = [tk.DoubleVar() for _ in range(10)]

        self.setup_gui()
        self.coefficients = {}

    def setup_gui(self):
        """
        Lay out the left panel (file/volume inputs + buttons + equations)
        and the right panel (plot canvas), configuring grid weights.
        """
        left_frame = ttk.LabelFrame(self.root, text="Upload Files and Volumes", padding=(10, 10))
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        tk.Label(left_frame, text="Upload JSON Files and Enter Actual Volumes (µL)",
                 font=("Arial", 12), fg="darkblue").grid(row=0, column=0, columnspan=3, pady=10)

        self.file_buttons = []

        for i in range(5):
            row = i + 1
            label = tk.Label(left_frame, text=f"File {i + 1}: ")
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

            file_button = tk.Button(left_frame, text="Browse", command=lambda idx=i: self.browse_file(idx))
            file_button.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            self.file_buttons.append(file_button)

            volume_entry = tk.Entry(left_frame, textvariable=self.actual_volumes[i])
            volume_entry.grid(row=row, column=2, padx=5, pady=5, sticky="w")
            self.create_context_menu(volume_entry)

        for i in range(5, 10):
            row = i - 4
            label = tk.Label(left_frame, text=f"File {i + 1}: ")
            label.grid(row=row, column=3, padx=5, pady=5, sticky="w")

            file_button = tk.Button(left_frame, text="Browse", command=lambda idx=i: self.browse_file(idx))
            file_button.grid(row=row, column=4, padx=5, pady=5, sticky="w")
            self.file_buttons.append(file_button)

            volume_entry = tk.Entry(left_frame, textvariable=self.actual_volumes[i])
            volume_entry.grid(row=row, column=5, padx=5, pady=5, sticky="w")
            self.create_context_menu(volume_entry)

        self.clear_button = tk.Button(left_frame, text="Clear", command=self.clear_all, width=10)
        self.clear_button.grid(row=6, column=0, padx=10, pady=15, sticky="w")

        self.save_button = tk.Button(left_frame, text="Save List", command=self.save_list, width=10)
        self.save_button.grid(row=6, column=1, padx=10, pady=15, sticky="nes")

        self.load_button = tk.Button(left_frame, text="Load List", command=self.load_list, width=10)
        self.load_button.grid(row=6, column=2, padx=10, pady=15, sticky="w")

        self.analyze_button = tk.Button(left_frame, text="Fit", command=self.analyze_and_plot,
                                        width=12, height=2, font=("Arial", 10, "bold"), fg="darkblue")
        self.analyze_button.grid(row=7, column=0, columnspan=2, pady=7)

        self.export_button = tk.Button(left_frame, text="Export", command=self.export_data,
                                       width=12, height=2, font=("Arial", 10, "bold"), fg="darkgreen")
        self.export_button.grid(row=7, column=2, columnspan=2, pady=7, sticky="w")

        self.equations_label = tk.Label(left_frame, text="", font=("Arial", 10), fg="darkblue", justify="left")
        self.equations_label.grid(row=8, column=0, columnspan=6, pady=20, padx=10, sticky="w")

        plot_frame = ttk.LabelFrame(self.root,
                                    text="Preview",
                                    padding=(10, 10))
        plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.plot_frame = tk.Frame(plot_frame)
        self.plot_frame.pack(fill="both", expand=True)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(0, weight=1)

    def browse_file(self, idx):
        """Handles file browsing and clears selection automatically if a file is already selected."""
        if self.json_files[idx]:
            self.json_files[idx] = None
            self.file_buttons[idx].config(text="Browse", bg="SystemButtonFace", relief="raised")
            return

        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.json_files[idx] = file_path
            self.file_buttons[idx].config(text="Selected", bg="#ADD8E6", relief="sunken")

    def highlight_last_used_button(self, idx):
        for btn in self.file_buttons:
            btn.config(relief="raised", bg="SystemButtonFace", highlightbackground="SystemButtonFace")

        self.file_buttons[idx].config(bg="#ADD8E6")

    def create_context_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Cut", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"))

        def show_context_menu(event):
            menu.post(event.x_root, event.y_root)

        widget.bind("<Button-3>", show_context_menu)

    def clear_all(self):
        """Resets all file selections, volume entries, and UI elements."""
        self.json_files = [None] * 10

        for button in self.file_buttons:
            button.config(
                text="Browse",
                bg="SystemButtonFace",
                relief="raised"
            )

        for var in self.actual_volumes:
            var.set(0)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.equations_label.config(text="", fg="darkblue")

        self.last_used_button = None

    def save_list(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if save_path:
            with open(save_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["File Path", "Actual Volume"])
                for file_path, actual_volume in zip(self.json_files, self.actual_volumes):
                    writer.writerow([file_path or "", actual_volume.get()])

    def load_list(self):
        load_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if load_path:
            try:
                with open(load_path, mode="r") as file:
                    reader = csv.DictReader(file)
                    if "File Path" not in reader.fieldnames or "Actual Volume" not in reader.fieldnames:
                        messagebox.showerror("Invalid File",
                                             "The selected file does not contain the required columns: 'File Path' and 'Actual Volume'.")
                        return

                    for i, row in enumerate(reader):
                        if i >= 10:
                            break
                        self.json_files[i] = row.get("File Path", "")
                        try:
                            self.actual_volumes[i].set(float(row.get("Actual Volume", 0)))
                        except ValueError:
                            self.actual_volumes[i].set(0)
                        self.file_buttons[i].config(text="Selected" if self.json_files[i] else "Browse")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading the file: {e}")

    def analyze_and_plot(self):
        """
        Read each selected JSON’s peak frequency and corresponding
        user‐entered volume, then call perform_regressions() and
        display the resulting scatter + fit curves.
        """
        frequencies = []
        volumes = []

        for i, file_path in enumerate(self.json_files):
            if file_path:
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        frequency = data["Estimated Peak"]["Frequency (Hz)"]
                        frequencies.append(frequency)
                        volumes.append(self.actual_volumes[i].get())
                except Exception as e:
                    messagebox.showerror("Error", f"Error reading file {file_path}: {e}")
                    return

        if not frequencies or not volumes:
            messagebox.showwarning("Warning", "Please upload valid files and enter volumes.")
            return

        frequencies = np.array(frequencies).reshape(-1, 1)
        volumes = np.array(volumes)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(frequencies, volumes, color="blue", label="Data Points")

        self.perform_regressions(frequencies, volumes, ax)
        self.display_plot(fig)

    def perform_regressions(self, frequencies, volumes, ax):
        """
        Fit and plot:
          • Linear
          • Quadratic
          • Cubic
          • Logarithmic (V = a·log(F) + b)
          • Seg‐Linear (piecewise lines)
        Store all model coefficients in self.coefficients.
        """
        self.coefficients = {}

        linear_model = LinearRegression()
        linear_model.fit(frequencies, volumes)
        linear_pred = linear_model.predict(frequencies)
        ax.plot(frequencies, linear_pred, color="red", linestyle="--", label="Linear Fit")
        self.coefficients['Linear'] = {
            "Intercept": linear_model.intercept_,
            "Slope": linear_model.coef_[0]
        }

        poly2 = PolynomialFeatures(degree=2)
        frequencies_quad = poly2.fit_transform(frequencies)
        quad_model = LinearRegression()
        quad_model.fit(frequencies_quad, volumes)
        quad_pred = quad_model.predict(frequencies_quad)
        ax.plot(frequencies, quad_pred, color="green", linestyle="--", label="Quadratic Fit")
        self.coefficients['Quadratic'] = {
            "Intercept": quad_model.intercept_,
            "Coefficients": quad_model.coef_.tolist()
        }

        poly3 = PolynomialFeatures(degree=3)
        frequencies_cubic = poly3.fit_transform(frequencies)
        cubic_model = LinearRegression()
        cubic_model.fit(frequencies_cubic, volumes)
        cubic_pred = cubic_model.predict(frequencies_cubic)
        ax.plot(frequencies, cubic_pred, color="purple", linestyle="--", label="Cubic Fit")
        self.coefficients['Cubic'] = {
            "Intercept": cubic_model.intercept_,
            "Coefficients": cubic_model.coef_.tolist()
        }

        log_frequencies = np.log(frequencies).reshape(-1, 1)
        log_model = LinearRegression()
        log_model.fit(log_frequencies, volumes)
        log_pred = log_model.predict(log_frequencies)
        ax.plot(frequencies, log_pred, color="orange", linestyle="--", label="Logarithmic Fit")
        self.coefficients['Logarithmic'] = {
            "Intercept": log_model.intercept_,
            "Coefficient": log_model.coef_[0]
        }

        seg_linear_equations = []
        for i in range(len(frequencies) - 1):
            x1, x2 = frequencies[i][0], frequencies[i + 1][0]
            y1, y2 = volumes[i], volumes[i + 1]
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            seg_linear_equations.append({
                "Min Frequency": x1,
                "Max Frequency": x2,
                "Slope": slope,
                "Intercept": intercept
            })
        self.coefficients['Seg-Linear'] = seg_linear_equations

        equations_text = "Equations:\n\n"
        equations_text += f"Linear: Volume = {linear_model.intercept_:.3f} + {linear_model.coef_[0]:.3f} * Frequency\n"
        equations_text += f"Quadratic: Volume = {quad_model.intercept_:.3f} + {quad_model.coef_[1]:.3f} * Frequency + {quad_model.coef_[2]:.3f} * Frequency^2\n"
        equations_text += f"Cubic: Volume = {cubic_model.intercept_:.3f} + {cubic_model.coef_[1]:.3f} * Frequency + {cubic_model.coef_[2]:.3f} * Frequency^2 + {cubic_model.coef_[3]:.3f} * Frequency^3\n"
        equations_text += f"Logarithmic: Volume = {log_model.intercept_:.3f} + {log_model.coef_[0]:.3f} * log(Frequency)\n"
        equations_text += "Seg-Linear: (Piecewise Linear Equations will be included in the exported file)\n"
        self.equations_label.config(text=equations_text, fg="darkblue")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Volume (µL)")
        ax.legend()

    def display_plot(self, fig):
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def export_data(self):
        """
        Prompt for a JSON path, write out self.coefficients,
        save the plot as PNG alongside it, and notify the user.
        """
        save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as file:
                json.dump(self.coefficients, file, indent=4)
            plt.savefig(save_path.replace(".json", ".png"), dpi=600)
            messagebox.showinfo("Export Complete", f"Data and plot saved successfully.")

    def on_exit(self):
        """
        Ask the user to confirm exit; if yes, destroy the window
        and terminate the process cleanly.
        """
        if messagebox.askyesno("Exit", "Are you sure you want to quit?"):
            self.root.destroy()
            os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = OverallTrendModeling(root)
    root.mainloop()
