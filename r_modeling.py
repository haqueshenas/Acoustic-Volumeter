#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Resonance Modeling Module
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
resonance_modeling.py

GUI for fitting analytical models (Gaussian, Lorentzian, Asymm. Lorentzian,
Voigt, Quadratic) to an RMS‐vs‐frequency CSV, extracting resonance peak
parameters and exporting model specs.

Usage:
    python resonance_modeling.py
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Rectangle

import sys, os

def resource(rel_path):
    """
    Return the absolute path to a resource (image, icon, or data file),
    whether running from source or from a PyInstaller one‐file bundle.
    """
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)


class ResonanceModeling:
    """
    Main window for loading RMS data, selecting a fit model, visualizing
    the fit, and exporting model parameters.
    """
    def __init__(self, root):
        """
        Initialize the Tk window, set up plotting canvas, and call
        create_widgets() to build the UI.
        """
        self.root = root
        self.root.withdraw()
        self.root.title("Resonance Modeling - Acoustic Volumeter v.1.0. ")
        self.root.geometry("1200x620")
        self.root.resizable(width=False, height=False)

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        try:
            self.root.iconbitmap(resource("AV.ico"))
        except Exception as e:
            pass

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

        self.filepath = None
        self.data = None
        self.original_data = None
        self.scale_mode = tk.StringVar(value="linear")
        self.use_percentile = tk.BooleanVar(value=False)
        self.percentile_value = tk.StringVar(value="0")
        self.percentile_line = None
        self.show_marker = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.root.bind("<Return>", lambda event: self.update_plot())
        self.canvas = None
        self.dynamic_text = None
        self.vertical_line = None
        self.drag_rectangle = None
        self.drag_start = None
        self.drag_end = None

        self.create_widgets()

    def create_widgets(self):
        """
        Build and arrange all controls: file loader, model chooser,
        scale/range inputs, Apply/Reset/Export buttons, and plot area.
        """
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=0, column=0, sticky="news", padx=20, pady=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=0, column=1, rowspan=10, padx=5, pady=5, sticky="w")

        self.load_button = tk.Button(self.control_frame, text="Load CSV File", command=self.load_file)
        self.load_button.grid(row=0, column=0, padx=5, pady=20, ipadx=18, ipady=10, sticky="w")

        self.export_button = tk.Button(self.control_frame, text="     Export       ", command=self.export_plot)
        self.export_button.grid(row=1, column=0, padx=5, pady=5, ipadx=18, ipady=10, sticky="w")

        tk.Label(self.control_frame, text="Start Frequency (Hz):").grid(row=2, column=0, sticky="w", padx=5)
        self.x_start_entry = tk.Entry(self.control_frame, width=10)
        self.x_start_entry.grid(row=2, column=0, padx=130, pady=15, sticky="w")
        self.x_start_entry.bind("<KeyRelease>", lambda _: self.update_range_overlay())

        tk.Label(self.control_frame, text="End Frequency (Hz):").grid(row=3, column=0, sticky="w", padx=5)
        self.x_end_entry = tk.Entry(self.control_frame, width=10)
        self.x_end_entry.grid(row=3, column=0, padx=130, pady=5, sticky="w")
        self.x_end_entry.bind("<KeyRelease>", lambda _: self.update_range_overlay())

        tk.Label(self.control_frame, text="Scale:").grid(row=4, column=0, sticky="w", padx=5, pady=10)
        self.linear_radio = tk.Radiobutton(
            self.control_frame, text="Linear", variable=self.scale_mode, value="linear", command=self.update_scale
        )
        self.linear_radio.grid(row=5, column=0, padx=15, pady=0, sticky="w")

        self.log_radio = tk.Radiobutton(
            self.control_frame, text="Logarithmic", variable=self.scale_mode, value="log", command=self.update_scale
        )
        self.log_radio.grid(row=6, column=0, padx=15, pady=5, sticky="w")

        self.percentile_checkbox = tk.Checkbutton(
            self.control_frame, text="Use RMS Percentile", variable=self.use_percentile, command=self.toggle_percentile
        )
        self.percentile_checkbox.grid(row=7, column=0, sticky="w", padx=5, pady=10)

        self.percentile_entry = tk.Entry(self.control_frame, textvariable=self.percentile_value, state="disabled",
                                         width=4)
        self.percentile_entry.grid(row=8, column=0, padx=20, pady=0, ipady=4, sticky="w")
        self.percentile_entry.bind("<KeyRelease>", lambda _: self.update_percentile_range())

        self.percentile_slider = tk.Scale(
            self.control_frame,
            from_=0,
            to=100,
            orient="horizontal",
            command=self.slider_update_percentile,
            state="disabled"
        )
        self.percentile_slider.grid(row=8, column=0, padx=60, sticky="w")
        self.percentile_slider.set(100)

        self.display_model_var = tk.StringVar(value="Lorentzian")
        self.model_label = tk.Label(self.control_frame, text="Model:")
        self.model_label.grid(row=9, column=0, sticky="w", padx=5, pady=30)
        self.model_dropdown = ttk.Combobox(
            self.control_frame,
            textvariable=self.display_model_var,
            values=["Quadratic", "Gaussian", "Lorentzian", "Asymmetric Lorentzian", "Voigt"],
            state="readonly"
        )
        self.model_dropdown.grid(row=9, column=0, padx=60, pady=30, sticky="w")

        self.grid_checkbox = tk.Checkbutton(
            self.control_frame, text="Grid", variable=self.show_grid, command=self.toggle_grid
        )
        self.grid_checkbox.grid(row=10, column=0, padx=30, pady=5, sticky="w")

        self.marker_checkbox = tk.Checkbutton(
            self.control_frame, text="Values", variable=self.show_marker, command=self.toggle_marker
        )
        self.marker_checkbox.grid(row=10, column=0, sticky="w", padx=120, pady=5)

        self.update_button = tk.Button(self.control_frame, text="Apply", command=self.update_plot)
        self.update_button.grid(row=11, column=0, padx=10, pady=20, ipadx=20, ipady=5, sticky="w")

        self.reset_button = tk.Button(self.control_frame, text="Reset", command=self.reset_plot)
        self.reset_button.grid(row=11, column=0, padx=110, pady=20, ipadx=20, ipady=5, sticky="w")

        self.set_widget_states("disabled")

    def load_file(self):
        """
        Prompt for a CSV file, validate columns, sort & store the data,
        and initialize the plot range.
        """
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not self.filepath:
            return

        try:
            self.data = pd.read_csv(self.filepath)
            self.set_widget_states("normal")
            if "Frequency (Hz)" not in self.data.columns or "RMS (dB)" not in self.data.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'RMS (dB)' columns.")
            self.data = self.data.sort_values(by="Frequency (Hz)")
            self.original_data = self.data.copy()

            self.x_start_entry.delete(0, tk.END)
            self.x_start_entry.insert(0, str(self.data["Frequency (Hz)"].min()))

            self.x_end_entry.delete(0, tk.END)
            self.x_end_entry.insert(0, str(self.data["Frequency (Hz)"].max()))

            self.drag_rectangle = None

            self.plot_data(self.data["Frequency (Hz)"], self.data["RMS (dB)"])

            self.toggle_percentile()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def set_widget_states(self, state):
        """
        Enable or disable all widgets except the Load File button.
        :param state: 'normal' to enable, 'disabled' to disable
        """
        widgets = [
            self.x_start_entry, self.x_end_entry, self.percentile_checkbox,
            self.percentile_entry, self.percentile_slider, self.marker_checkbox,
            self.grid_checkbox, self.update_button, self.reset_button,
            self.export_button, self.linear_radio, self.log_radio, self.model_label, self.model_dropdown
        ]
        for widget in widgets:
            widget.config(state=state)

    @staticmethod
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def gaussian(x, a, x0, sigma, offset):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    @staticmethod
    def lorentzian(x, a, x0, gamma, offset):
        return a * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2) + offset

    @staticmethod
    def asymmetric_lorentzian(x, a, x0, gamma, offset, asymmetry):
        return (
                a * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)
                + offset
                + asymmetry * (x - x0)
        )

    @staticmethod
    def voigt(x, a, x0, sigma, gamma, offset):
        from scipy.special import wofz
        z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
        return a * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + offset

    def fit_data(self, x, y):
        """
        Curve-fit the selected analytical model (Quadratic, Gaussian, etc.)
        to x/y data. Returns a callable fit function, the parameters array,
        and the computed peak (frequency, amplitude).
        """
        if len(x) < 2 or len(y) < 2:
            raise ValueError("Not enough data points for fitting.")

        models = {
            "Gaussian": (self.gaussian, [max(y), np.mean(x), np.std(x), min(y)]),
            "Lorentzian": (self.lorentzian, [max(y), np.mean(x), np.std(x), min(y)]),
            "Asymmetric Lorentzian": (
                self.asymmetric_lorentzian, [max(y), np.mean(x), np.std(x), min(y), 0.1]
            ),
            "Voigt": (
                self.voigt,
                [
                    max(y),
                    np.mean(x),
                    (max(x) - min(x)) / 4,
                    (max(x) - min(x)) / 8,
                    min(y),
                ],
            ),
            "Quadratic": (self.quadratic, [1, 1, np.mean(y)]),
        }

        model = self.display_model_var.get()
        if model not in models:
            raise ValueError(f"Unknown model selected: {model}")

        model_func, initial_guess = models[model]

        maxfev = min(5000, len(x) * 500)

        try:
            popt, _ = curve_fit(model_func, x, y, p0=initial_guess, maxfev=maxfev)

            if model in ["Gaussian", "Lorentzian", "Asymmetric Lorentzian", "Voigt"]:
                peak_freq = popt[1]
                peak_amplitude = model_func(peak_freq, *popt)
            elif model == "Quadratic":
                peak_freq = -popt[1] / (2 * popt[0]) if popt[0] != 0 else None
                peak_amplitude = model_func(peak_freq, *popt) if peak_freq is not None else None
            else:
                peak_freq, peak_amplitude = None, None

            fit_func = lambda x: model_func(x, *popt)
            return fit_func, popt, (peak_freq, peak_amplitude)

        except Exception as e:
            raise RuntimeError(f"Failed to fit the {model} model: {e}")

    def plot_data(self, x, y, fitted_x=None, fitted_y=None, r2=None, rmse=None, equation=None, peak=None):
        """
        Render the RMS scatter plot, overlay the fit curve & peak marker,
        and hook up interactive mouse/drag events.
        """
        self.ax.clear()
        if self.scale_mode.get() == "log":
            self.ax.set_xscale("log")

        self.ax.plot(x, y, label="RMS", color="blue", marker='o', markersize=6)

        if self.show_grid.get():
            self.ax.grid(True)
        else:
            self.ax.grid(False)

        if fitted_x is not None and fitted_y is not None:
            self.ax.plot(fitted_x, fitted_y, label=f"{self.display_model_var.get()} Fit \n(R²={r2:.4f}, \nRMSE={rmse:.4f})",
                         color="red")
            if hasattr(self, 'model_label_text') and self.model_label_text:
                self.model_label_text.remove()
            self.model_label_text = self.ax.text(
                0.05, 0.95,
                f"{equation}\nPeak: {peak[0]:.2f} Hz, {peak[1]:.2f} dB",
                transform=self.ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7)
            )


        if peak is not None and peak[0] is not None:
            self.ax.scatter(peak[0], peak[1], color="green", label=f"Peak @ {peak[0]:.2f} Hz", s=100, marker='^')

        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("RMS (dB)")
        self.ax.legend()

        self.ax.set_xlim([x.min(), x.max()])

        if self.filepath:
            filename = os.path.basename(self.filepath)
            self.figure.suptitle(
                filename,
                fontsize=12,
                color='gray',
                horizontalalignment='center',
                weight = 'bold'
            )

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        if self.show_marker.get():
            self.canvas.mpl_connect("motion_notify_event", self.show_mouse_effects)
        self.canvas.mpl_connect("button_press_event", self.start_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_drag_overlay)
        self.canvas.mpl_connect("button_release_event", self.end_drag)

    def show_mouse_effects(self, event):
        """Show frequency near the mouse position and a vertical dashed line."""
        if event.xdata is not None and self.show_marker.get():
            self.dynamic_text = self.safe_remove(self.dynamic_text, self.ax.texts)
            self.vertical_line = self.safe_remove(self.vertical_line, self.ax.lines)

            freq_text = f"{event.xdata:.2f} Hz"
            self.dynamic_text = self.ax.text(event.xdata, event.ydata, freq_text, color="red", fontsize=9)
            self.vertical_line = self.ax.axvline(x=event.xdata, color="red", linestyle="--", linewidth=1)
            self.canvas.draw()
        else:
            self.dynamic_text = self.safe_remove(self.dynamic_text, self.ax.texts)
            self.vertical_line = self.safe_remove(self.vertical_line, self.ax.lines)
            self.canvas.draw()

    def toggle_percentile(self):
        """Enable or disable the percentile entry and slider based on checkbox state."""
        state = "normal" if self.use_percentile.get() else "disabled"
        self.percentile_entry.config(state=state)
        self.percentile_slider.config(state=state)

        if not self.use_percentile.get():
            self.clear_percentile_overlay()

    def update_percentile_range(self):
        """Update the X-Range based on the RMS percentile threshold."""
        if not self.use_percentile.get():
            return

        try:
            value = self.percentile_value.get().strip()
            if not value:
                return

            percentile = float(value)
            if percentile < 0 or percentile > 100:
                raise ValueError("Percentile must be between 0 and 100.")

            self.percentile_slider.set(percentile)

            rms_threshold = np.percentile(self.original_data["RMS (dB)"], percentile)

            if self.percentile_line:
                self.percentile_line = self.safe_remove(self.percentile_line, self.ax.lines)

            self.percentile_line = self.ax.axhline(
                y=rms_threshold, color="purple", linestyle="--", label=f"{percentile}th Percentile"
            )

            valid_indices = self.original_data["RMS (dB)"] >= rms_threshold
            x_range = self.original_data.loc[valid_indices, "Frequency (Hz)"]

            if x_range.empty:
                raise ValueError("No data found for the specified percentile.")

            x_start, x_end = x_range.min(), x_range.max()

            self.x_start_entry.delete(0, tk.END)
            self.x_start_entry.insert(0, f"{x_start:.2f}")
            self.x_end_entry.delete(0, tk.END)
            self.x_end_entry.insert(0, f"{x_end:.2f}")

            self.canvas.draw()

        except ValueError as e:
            self.percentile_slider.set(0)
            messagebox.showerror("Error", f"Invalid percentile value: {e}")

    def slider_update_percentile(self, value):
        """Synchronize slider and entry for percentile value."""
        self.percentile_value.set(value)
        self.update_percentile_range()

    def clear_percentile_overlay(self):
        """Clear the percentile dashed line from the plot."""
        if self.percentile_line:
            self.percentile_line = self.safe_remove(self.percentile_line, self.ax.lines)
        self.canvas.draw()

    @staticmethod
    def safe_remove(obj, container):
        """Safely remove a matplotlib object from a container."""
        if obj and obj in container:
            try:
                obj.remove()
            except ValueError:
                pass
            return None
        return obj

    def toggle_marker(self):
        """Enable or disable the marker feature dynamically."""
        if not self.show_marker.get():
            self.dynamic_text = self.safe_remove(self.dynamic_text, self.ax.texts)
            self.vertical_line = self.safe_remove(self.vertical_line, self.ax.lines)
            self.canvas.draw()
        else:
            self.canvas.mpl_connect("motion_notify_event", self.show_mouse_effects)

    def toggle_grid(self):
        if self.show_grid.get():
            self.ax.grid(True)
        else:
            self.ax.grid(False)

        self.canvas.draw()

    def start_drag(self, event):
        if event.xdata:
            self.drag_rectangle = self.safe_remove(self.drag_rectangle, self.ax.patches)
            self.drag_start = event.xdata
            self.drag_rectangle = self.ax.axvspan(self.drag_start, self.drag_start, color="red", alpha=0.3)
            self.canvas.draw()

    def update_drag_overlay(self, event):
        if self.drag_start is not None and event.xdata:
            x_min = min(self.drag_start, event.xdata)
            x_max = max(self.drag_start, event.xdata)
            if self.drag_rectangle:
                self.drag_rectangle.remove()
            self.drag_rectangle = self.ax.axvspan(x_min, x_max, color="red", alpha=0.3)
            self.canvas.draw()

    def end_drag(self, event):
        if event.xdata and self.drag_start:
            self.drag_end = event.xdata
            x_min = min(self.drag_start, self.drag_end)
            x_max = max(self.drag_start, self.drag_end)

            self.x_start_entry.delete(0, tk.END)
            self.x_start_entry.insert(0, f"{x_min:.2f}")
            self.x_end_entry.delete(0, tk.END)
            self.x_end_entry.insert(0, f"{x_max:.2f}")
            self.drag_start = None
            self.drag_end = None

    def update_range_overlay(self):
        """Draw a red transparent layer for the selected range."""
        if self.drag_rectangle:
            self.drag_rectangle = self.safe_remove(self.drag_rectangle, self.ax.patches)

        try:
            x_start = float(self.x_start_entry.get())
            x_end = float(self.x_end_entry.get())

            if x_start >= x_end:
                raise ValueError("Start frequency must be less than end frequency.")

            if self.scale_mode.get() == "log":
                if x_start <= 0 or x_end <= 0:
                    raise ValueError("Logarithmic scale requires positive x-range values.")
                x_start, x_end = np.log10([x_start, x_end])

            self.drag_rectangle = self.ax.axvspan(x_start, x_end, color="red", alpha=0.3)
            self.canvas.draw()

        except ValueError as e:
            if self.drag_rectangle:
                self.drag_rectangle = self.safe_remove(self.drag_rectangle, self.ax.patches)
            self.canvas.draw()

    def update_plot(self):
        """
        Read the user’s start/end frequencies, trim the dataset,
        run fit_data(), compute R² & RMSE, then call plot_data().
        Shows an error dialog on failure.
        """
        try:
            x_start = float(self.x_start_entry.get())
            x_end = float(self.x_end_entry.get())

            if x_start >= x_end:
                raise ValueError("Start frequency must be less than end frequency.")

            filtered_data = self.data[(self.data["Frequency (Hz)"] >= x_start) &
                                      (self.data["Frequency (Hz)"] <= x_end)]
            if filtered_data.empty:
                raise ValueError("No data in the selected range.")

            x = filtered_data["Frequency (Hz)"].values
            y = filtered_data["RMS (dB)"].values

            fit_func, popt, peak = self.fit_data(x, y)
            fitted_y = fit_func(x)

            residuals = y - fitted_y
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals ** 2))

            equation = f"Fit: {self.display_model_var.get()}\nParameters: {popt}\nR² = {r2:.4f}"

            self.plot_data(x, y, x, fitted_y, r2, rmse, equation, peak)

            if hasattr(self, 'model_label_text') and self.model_label_text:
                self.model_label_text.remove()
                self.model_label_text = None

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update plot: {e}")

    def update_scale(self):
        """
        Toggle between linear and logarithmic x-axis scaling,
        then redraw the plot.
        """
        if self.data is not None:
            self.drag_rectangle = self.safe_remove(self.drag_rectangle, self.ax.patches)

            self.drag_rectangle = None

            self.plot_data(self.data["Frequency (Hz)"], self.data["RMS (dB)"])

    def reset_plot(self):
        """
        Restore the original full dataset and default UI controls,
        clearing any overlays or percentile settings.
        """
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.plot_data(self.data["Frequency (Hz)"], self.data["RMS (dB)"])
            self.x_start_entry.delete(0, tk.END)
            self.x_end_entry.delete(0, tk.END)
            self.x_start_entry.insert(0, str(self.data["Frequency (Hz)"].min()))
            self.x_end_entry.insert(0, str(self.data["Frequency (Hz)"].max()))

            self.dynamic_text = self.safe_remove(self.dynamic_text, self.ax.texts)
            self.vertical_line = self.safe_remove(self.vertical_line, self.ax.lines)
            self.drag_rectangle = self.safe_remove(self.drag_rectangle, self.ax.patches)

            self.clear_percentile_overlay()

            self.percentile_value.set("0")
            self.percentile_slider.set(0)
            self.use_percentile.set(False)
            self.percentile_entry.config(state="disabled")
            self.percentile_slider.config(state="disabled")

            self.canvas.draw()

    def export_plot(self):
        """
        Save the current RMS plot as a high-res PNG and write a JSON
        alongside it containing model name, coefficients, peak, and RMSE.
        """
        if not self.filepath:
            messagebox.showerror("Error", "No data loaded to export.")
            return

        try:
            base_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            if not base_path:
                return

            plot_path = base_path if base_path.endswith(".png") else base_path + ".png"

            self.figure.savefig(plot_path, dpi=600)

            x_start = float(self.x_start_entry.get())
            x_end = float(self.x_end_entry.get())

            filtered_data = self.data[(self.data["Frequency (Hz)"] >= x_start) & (self.data["Frequency (Hz)"] <= x_end)]
            if filtered_data.empty:
                raise ValueError("No data in the selected range.")

            x = filtered_data["Frequency (Hz)"].to_numpy()
            y = filtered_data["RMS (dB)"].to_numpy()

            models = {
                "Gaussian": (self.gaussian, [max(y), np.mean(x), np.std(x), min(y)]),
                "Lorentzian": (self.lorentzian, [max(y), np.mean(x), np.std(x), min(y)]),
                "Asymmetric Lorentzian": (
                    self.asymmetric_lorentzian, [max(y), np.mean(x), np.std(x), min(y), 0.1]
                ),
                "Voigt": (
                    self.voigt,
                    [
                        max(y),
                        np.mean(x),
                        (max(x) - min(x)) / 4,
                        (max(x) - min(x)) / 8,
                        min(y),
                    ]
                ),
                "Quadratic": (self.quadratic, [1, 1, np.mean(y)]),
            }

            model_name = self.display_model_var.get()
            if model_name not in models:
                raise ValueError(f"Model '{model_name}' not supported.")

            model_func, initial_guess = models[model_name]

            maxfev = min(5000, len(x) * 500)

            popt, _ = curve_fit(model_func, x, y, p0=initial_guess, maxfev=maxfev)

            y_fit = model_func(x, *popt)

            residuals = y - y_fit
            rmse = np.sqrt(np.mean(residuals ** 2))

            if model_name in ["Gaussian", "Lorentzian", "Asymmetric Lorentzian", "Voigt"]:
                peak_freq = popt[1]
                peak_amplitude = model_func(peak_freq, *popt)
            elif model_name == "Quadratic":
                peak_freq = -popt[1] / (2 * popt[0]) if popt[0] != 0 else None
                peak_amplitude = model_func(peak_freq, *popt) if peak_freq else None

            if peak_freq is None or np.isnan(peak_amplitude):
                raise ValueError("Failed to calculate a valid peak for the selected model.")

            coefficients = {}
            if model_name == "Gaussian":
                coefficients = {"a": popt[0], "x0": popt[1], "sigma": popt[2], "offset": popt[3]}
            elif model_name == "Lorentzian":
                coefficients = {"a": popt[0], "x0": popt[1], "gamma": popt[2], "offset": popt[3]}
            elif model_name == "Asymmetric Lorentzian":
                coefficients = {"a": popt[0], "x0": popt[1], "gamma": popt[2], "offset": popt[3], "asymmetry": popt[4]}
            elif model_name == "Voigt":
                coefficients = {"a": popt[0], "x0": popt[1], "sigma": popt[2], "gamma": popt[3], "offset": popt[4]}
            elif model_name == "Quadratic":
                coefficients = {"a": popt[0], "b": popt[1], "c": popt[2]}

            data_to_export = {
                "Selected Model": model_name,
                "Equation Coefficients": coefficients,
                "Estimated Peak": {
                    "Frequency (Hz)": peak_freq,
                    "Amplitude (dB)": peak_amplitude,
                },
                "Metrics": {
                    "RMSE": rmse,
                },
            }

            json_path = os.path.splitext(plot_path)[0] + ".json"
            with open(json_path, "w") as json_file:
                json.dump(data_to_export, json_file, indent=4)

            messagebox.showinfo("Export Successful", f"Files saved successfully.")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def on_exit(self):
        """
        Confirm with the user, then cleanly destroy the window
        and exit the process.
        """
        if messagebox.askyesno("Exit", "Are you sure you want to quit?"):
            self.root.destroy()
            os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ResonanceModeling(root)
    root.mainloop()