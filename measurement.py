#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Measurement Module
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
measurement.py

Graphical interface for live acoustic volumetry:
  - Plays frequency sweeps, records audio
  - Computes RMS vs frequency and fits resonance models
  - Estimates unknown object volumes via calibration curves
  - Saves results as PNG, WAV, CSV, and JSON

Usage:
    python measurement.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import csv
import os
import io
from scipy.optimize import curve_fit
import threading
import sounddevice as sd
import numpy as np
import pandas as pd
import wave
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import threading

import sys, os

def resource(rel_path):
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

DEVICE_CONFIG_FILE = "device_config.json"
SINGLETON_CONFIG_FILE = "singletone_config.json"

def load_device_configuration():
    """Loads device configuration with fallback logic and validation."""
    config = {"input_device": "Unknown Device", "output_device": "Unknown Device"}
    global config_missing

    if os.path.exists(SINGLETON_CONFIG_FILE):
        try:
            with open(SINGLETON_CONFIG_FILE, 'r') as f:
                config.update(json.load(f))
            config_missing = False
            return validate_device_config(config)
        except Exception as e:
            print(f"Error loading {SINGLETON_CONFIG_FILE}: {e}")

    if os.path.exists(DEVICE_CONFIG_FILE):
        try:
            with open(DEVICE_CONFIG_FILE, 'r') as f:
                config.update(json.load(f))
            config_missing = False
            return validate_device_config(config)
        except Exception as e:
            print(f"Error loading {DEVICE_CONFIG_FILE}: {e}")

    config_missing = True
    return config

def validate_device_config(config):
    """Validates device configuration against available devices."""
    available_devices = sd.query_devices()
    input_names = [dev["name"] for dev in available_devices if dev["max_input_channels"] > 0]
    output_names = [dev["name"] for dev in available_devices if dev["max_output_channels"] > 0]

    if config["input_device"] not in input_names:
        config["input_device"] = "Unknown Device"
    if config["output_device"] not in output_names:
        config["output_device"] = "Unknown Device"

    return config

device_config = load_device_configuration()

def show_missing_config_warning(app):
    """Displays a warning if device configuration files are missing."""
    warning_window = tk.Toplevel(app)
    warning_window.title("Missing Configuration")
    warning_window.geometry("400x200")
    warning_window.transient(app)
    warning_window.grab_set()
    warning_window.lift()
    warning_window.attributes("-topmost", True)

    try:
        warning_window.iconbitmap(resource("AV.ico"))
    except Exception as e:
        print(f"Could not load icon: {e}")  # Log if there's an issue with the icon

    x = (app.winfo_screenwidth() - 400) // 2
    y = (app.winfo_screenheight() - 200) // 2
    warning_window.geometry(f"+{x}+{y}")

    tk.Label(
        warning_window,
        text=(
            "The device configuration file is missing or has not been created yet.\n\n"
            "Please use the 'Acoustic Profiling' tool to select and save the input/output devices.\n\n"
            "Alternatively, you can manually set the devices from the 'Detected Devices' section."
        ),
        wraplength=360,
        justify="left",
        padx=20,
        pady=10
    ).pack()

    tk.Button(warning_window, text="OK", command=warning_window.destroy, width=10).pack(pady=10)


CONFIG_FILE = resource("measurement_config.json")


class Measurement(tk.Tk):
    """
    Main application window for performing an acoustic sweep,
    plotting results, fitting models, and saving outputs.
    """
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Measurement - Acoustic Volumeter v. 1.0")
        self.geometry("720x730")
        self.iconbitmap(resource("AV.ico"))
        self.resizable(width=False, height=False)

        self.lift()
        self.attributes('-topmost', True)
        self.after(1000, lambda: self.attributes('-topmost', False))

        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        self.deiconify()

        # Initialize variables
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.bind_shortcuts()
        self.sound_frequency = tk.StringVar(value="600-1200")
        self.playback_level = tk.DoubleVar(value=0.005)
        self.frequency_resolution = tk.DoubleVar(value=100)
        self.sound_duration = tk.DoubleVar(value=3.0)
        self.silence_duration = tk.DoubleVar(value=0)
        self.initial_lag = tk.BooleanVar(value=True)
        self.lag_time = tk.IntVar(value=50)
        self.project_name = tk.StringVar()
        self.treatment_name = tk.StringVar()
        self.replicate_counter = tk.IntVar(value=1)
        self.project_folder = tk.StringVar()
        self.is_measuring = False
        self.trimmed_rms_values = []
        self.remaining_time = tk.StringVar(value="")

        self.load_last_settings()
        self._device_error_shown = False
        self._input_device_index = None
        self._output_device_index = None
        self.bind_project_events()
        style = ttk.Style(self)
        style.configure("TCombobox", arrowsize=12)
        style.configure("TCombobox", width=15)

        self.waveform_fig, self.waveform_ax = plt.subplots(figsize=(2, 1))
        self.waveform_ax.set_title("")
        self.waveform_ax.set_xlabel("")
        self.waveform_ax.set_ylabel("")
        self.wave_data = None

        self.waveform_ax.spines['top'].set_color('white')
        self.waveform_ax.spines['bottom'].set_color('white')
        self.waveform_ax.spines['left'].set_color('white')
        self.waveform_ax.spines['right'].set_color('white')

        self.waveform_ax.xaxis.label.set_color('white')
        self.waveform_ax.yaxis.label.set_color('white')
        self.waveform_ax.tick_params(axis='x', colors='white')
        self.waveform_ax.tick_params(axis='y', colors='white')

        self.rms_fig, self.rms_ax = plt.subplots(figsize=(3, 2))
        self.rms_fig.set_layout_engine('constrained')
        self.rms_ax.axis("off")

        self.create_widgets()

    def create_widgets(self):
        # Project Information Frame
        project_frame = ttk.LabelFrame(self, text="Project")
        project_frame.grid(row=0, column=0, columnspan=1, padx=10, pady=5, sticky="news")
        project_frame.columnconfigure(1, weight=1)
        ttk.Label(project_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(project_frame, textvariable=self.project_name, width=30).grid(row=0, column=1, columnspan=25, padx=5, pady=5,
                                                                      sticky="wns")
        ttk.Label(project_frame, text="Treatment:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(project_frame, textvariable=self.treatment_name, width=20).grid(row=1, column=1, columnspan=25, padx=5, pady=5,
                                                                        sticky="wns")
        ttk.Label(project_frame, text="Replicate:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(project_frame, textvariable=self.replicate_counter, width=10).grid(row=2, column=1, columnspan=25, padx=5,
                                                                           pady=5, sticky="wns")
        ttk.Label(project_frame, text="Save Folder:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(project_frame, textvariable=self.project_folder, state="readonly").grid(row=3, column=1, padx=5,
                                                                                          pady=5, sticky="ew")
        ttk.Button(project_frame, text="Browse", command=self.browse_folder).grid(row=3, column=2, padx=5, pady=5,
                                                                                  sticky="ew")

        # Calibration Frame
        calibration_frame = ttk.LabelFrame(self, text="Calibration")
        calibration_frame.grid(row=0, column=1, padx=(10, 10), pady=(84, 0), sticky="new")

        # Model label and Browse button
        ttk.Label(calibration_frame, text="Load File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_file_path = tk.StringVar()
        ttk.Button(calibration_frame, text="Browse", command=self.browse_model_file).grid(row=1, column=0, padx=5,
                                                                                          pady=5, sticky="w")

        # EST Model dropdown menu
        ttk.Label(calibration_frame, text="EST Model:").grid(row=0, column=1, padx=20, pady=5, sticky="news")
        self.est_model_selection = tk.StringVar(value="Lorentzian")  # Define as an instance attribute
        est_model_options = ttk.Combobox(
            calibration_frame,
            textvariable=self.est_model_selection,
            values=["None", "Quadratic", "Gaussian", "Lorentzian", "Asymmetric Lorentzian", "Voigt"],
            # Available EST Models
            state="readonly",
            style="TCombobox",
            width=20,
        )
        est_model_options.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        est_model_options.bind("<<ComboboxSelected>>", self.update_model_fit)

        # Overall Model dropdown menu
        ttk.Label(calibration_frame, text="Overall Model:").grid(row=1, column=1, padx=20, pady=5, sticky="news")
        self.model_selection = tk.StringVar(value="Seg-Linear")
        model_options = ttk.Combobox(
            calibration_frame, textvariable=self.model_selection, values=["Linear", "Seg-Linear", "Quadratic",
                                                                          "Cubic", "Logarithmic"],
            state="readonly", width=11
        )
        model_options.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        model_options.bind("<<ComboboxSelected>>", self.update_model_fit)

        # Detected Devices Frame
        device_frame = ttk.LabelFrame(self, text="Detected Devices")
        device_frame.grid(row=0, column=1, columnspan=1, padx=10, pady=3, sticky="new")
        device_frame.pack_propagate(False)

        input_device_name = str(device_config.get('input_device', "Unknown Device"))
        output_device_name = str(device_config.get('output_device', "Unknown Device"))

        input_device_name = input_device_name[:45] + "..." if len(input_device_name) > 45 else input_device_name
        output_device_name = output_device_name[:45] + "..." if len(output_device_name) > 45 else output_device_name

        self.input_device_label = ttk.Label(device_frame, text=f"{input_device_name}")
        self.input_device_label.grid(row=0, column=1, padx=0, pady=5, sticky="w")

        self.output_device_label = ttk.Label(device_frame, text=f"{output_device_name}")
        self.output_device_label.grid(row=1, column=1, padx=0, pady=5, sticky="w")

        style = ttk.Style()
        style.configure("Custom.TButton", font=("TkDefaultFont", 8))

        input_change_button = ttk.Button(device_frame, text="In", command=self.change_input_device,
                                         style="Custom.TButton")
        input_change_button.grid(row=0, column=0, padx=10, pady=3, sticky="w")

        output_change_button = ttk.Button(device_frame, text="Out", command=self.change_output_device,
                                          style="Custom.TButton")
        output_change_button.grid(row=1, column=0, padx=10, pady=3, sticky="w")

        playback_frame = ttk.LabelFrame(self, text="Playback Settings")
        playback_frame.grid(row=3, column=0, padx=10, pady=5, sticky="news")
        playback_frame.columnconfigure(1, weight=1)
        ttk.Label(playback_frame, text="Frequency Range (Hz):").grid(row=0, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.sound_frequency, width=15).grid(row=0, column=1, padx=5, pady=5, sticky="ns")
        ttk.Label(playback_frame, text="Playback Level (0-1):").grid(row=1, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.playback_level, width=15).grid(row=1, column=1, padx=5, pady=5, sticky="ns")
        ttk.Label(playback_frame, text="Frequency Resolution:").grid(row=2, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.frequency_resolution, width=15).grid(row=2, column=1, padx=5, pady=5,
                                                                               sticky="ns")
        ttk.Label(playback_frame, text="Playback Duration (s):").grid(row=3, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.sound_duration, width=15).grid(row=3, column=1, padx=5, pady=5, sticky="ns")
        ttk.Label(playback_frame, text="Silence Duration (s):").grid(row=4, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.silence_duration, width=15).grid(row=4, column=1, padx=5, pady=5, sticky="ns")
        ttk.Checkbutton(playback_frame, text="Exclude Initial Lag (ms):", variable=self.initial_lag).grid(row=5,
                                                                                                          column=0,
                                                                                                          padx=15,
                                                                                                          pady=5,
                                                                                                          sticky="w")
        ttk.Entry(playback_frame, textvariable=self.lag_time, width=15).grid(row=5, column=1, padx=5, pady=5, sticky="ns")

        # Waveform Display Frame
        waveform_frame = ttk.LabelFrame(self, text="Waveform")
        waveform_frame.grid(row=4, column=0, padx=10, pady=5, sticky="news")
        waveform_frame.columnconfigure(0, weight=1)
        waveform_frame.rowconfigure(0, weight=1)
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, master=waveform_frame)
        self.waveform_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # RMS vs Frequency Display Frame
        rms_frame = ttk.LabelFrame(self, text="RMS vs Frequency")
        rms_frame.grid(row=3, column=1, padx=10, pady=5, sticky="news")
        rms_frame.columnconfigure(0, weight=1)
        rms_frame.rowconfigure(0, weight=1)
        self.rms_canvas = FigureCanvasTkAgg(self.rms_fig, master=rms_frame)
        self.rms_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.enable_span_selector()

        # Trim Frame
        self.trim_frame = ttk.LabelFrame(self, text="Trim")
        self.trim_frame.grid(row=4, column=1, padx=10, pady=5, sticky="news")
        self.trim_frame.columnconfigure(0, weight=1)
        self.trim_frame.columnconfigure(1, weight=1)
        self.trim_frame.columnconfigure(2, weight=1)
        self.trim_frame.columnconfigure(3, weight=1)

        ttk.Label(self.trim_frame, text="Freq. Range (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.frequency_min = tk.DoubleVar(value=200)
        self.frequency_max = tk.DoubleVar(value=2000)
        self.frequency_min_entry = ttk.Entry(self.trim_frame, textvariable=self.frequency_min, width=8)
        self.frequency_max_entry = ttk.Entry(self.trim_frame, textvariable=self.frequency_max, width=8)
        self.frequency_min_entry.grid(row=0, column=1, padx=5, pady=0, sticky="w")
        ttk.Label(self.trim_frame, text="to").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.frequency_max_entry.grid(row=0, column=3, padx=5, pady=0, sticky="w")

        ttk.Label(self.trim_frame, text="Exclude RMS Percentile:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.percentile_value = tk.DoubleVar(value=0)
        self.percentile_entry = ttk.Entry(self.trim_frame, textvariable=self.percentile_value, width=8)
        self.percentile_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.percentile_slider = ttk.Scale(
            self.trim_frame, from_=0, to=100, orient="horizontal",
            command=self.slider_update_percentile, variable=self.percentile_value
        )
        self.percentile_slider.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        self.frequency_min_entry.bind("<KeyRelease>", lambda _: self.update_trimmed_data())
        self.frequency_max_entry.bind("<KeyRelease>", lambda _: self.update_trimmed_data())
        self.percentile_entry.bind("<KeyRelease>", self.update_percentile_range)
        self.percentile_slider.config(command=self.slider_update_percentile)

        self.rmse_value = tk.StringVar(value="RMSE: ---")
        self.rmse_label = ttk.Label(self.trim_frame, textvariable=self.rmse_value)
        self.rmse_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.rmse_label.configure(foreground="dark blue")

        self.reset_button = tk.Button(self.trim_frame, text="Reset", command=self.reset_plots)
        self.reset_button.grid(row=2, column=0, columnspan=4, padx=10, pady=5, ipadx=50, ipady=1, sticky="ens")
        self.reset_button.config(command=lambda: self.reset_plots(cancelled=False))

        # Results Frame
        rms_frame = ttk.LabelFrame(self, text="Result Display", width=200, height=100)
        rms_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        rms_frame.grid_propagate(False)

        self.rms_value = tk.StringVar(value="")
        self.rms_value_label = ttk.Label(rms_frame, textvariable=self.rms_value, font=("Arial", 14, "bold"))
        self.rms_value_label.pack(padx=5, pady=15)

        action_frame = ttk.Frame(self)
        action_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        action_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(action_frame, text="Run", command=self.start_measurement)
        self.run_button.grid(row=0, column=0, padx=10, pady=5, ipadx=10, ipady=10, sticky="news")

        self.cancel_button = ttk.Button(action_frame, text="Cancel", command=self.cancel_measurement, state="disabled")
        self.cancel_button.grid(row=0, column=1, padx=10, pady=5, ipadx=10, ipady=10, sticky="news")

        save_button = ttk.Button(action_frame, text="Save", command=self.save_results)
        save_button.grid(row=0, column=2, padx=10, pady=5, ipadx=10, ipady=10,
                         sticky="news")

        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=1)

        # Remaining Time Status Frame
        status_frame = tk.Frame(self, borderwidth=0, highlightthickness=0)
        status_frame.grid(row=7, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=1)

        self.remaining_time.set("")
        self.remaining_time_label = tk.Label(
            status_frame,
            textvariable=self.remaining_time,
            font=("Arial", 10),
            anchor="center"
        )
        self.remaining_time_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ews")

    def bind_shortcuts(self):
        """Bind keyboard shortcuts to GUI actions."""
        self.bind("<Control-Return>", lambda event: self.start_measurement())
        self.bind("<Control-s>", lambda event: self.save_results())
        self.bind("<Control-r>", lambda event: self.reset_plots())
        self.bind("<Escape>", lambda event: self.cancel_measurement())

    def safe_configure_widget(self, widget, **kwargs):
        """Safely configure a widget if it exists."""
        if widget and widget.winfo_exists():
            widget.config(**kwargs)

    @staticmethod
    def truncate_device_name(name, max_length=40):
        """Truncates a device name if it exceeds the maximum length."""
        if not isinstance(name, str):
            return "Unknown Device"
        return name if len(name) <= max_length else name[:max_length] + "..."

    def update_status_message(self, remaining_time, frequency=0, current_repeat=0, total_repeats=0):
        """Updates the status message with frequency and repetition details."""
        minutes, seconds = divmod(int(remaining_time), 60)
        time_text = f"{minutes:02}:{seconds:02}"
        frequency_text = f"Playing frequency: {frequency:.0f} Hz ({current_repeat}/{total_repeats})"
        self.remaining_time.set(f"{frequency_text} | Remaining time: {time_text}")

    def play_test_tone(self):
        """Play a short test tone to verify the audio device."""
        try:
            test_tone = self.generate_tone(frequency=1000, duration=0.3, level=0.005)  # 1 kHz tone for 1 second
            sd.play(test_tone, samplerate=44100, blocking=True)
        except Exception as e:
            messagebox.showerror("Device Error", "Failed to play test tone. Please check the selected device.")

    def change_input_device(self):
        devices = sd.query_devices()
        input_devices = [dev["name"] for dev in devices if dev["max_input_channels"] > 0]

        input_dropdown = ttk.Combobox(
            self.input_device_label.master,
            values=input_devices,
            state="readonly",
            style="TCombobox",
            width=30,
        )
        input_dropdown.grid(row=0, column=1, padx=0, pady=0, ipadx=10, sticky="w")
        input_dropdown.set("Select")

        def on_input_select(event):
            selected_device = input_dropdown.get()
            if selected_device:
                device_config["input_device"] = selected_device
                self.input_device_label.config(text=self.truncate_device_name(selected_device))
            input_dropdown.destroy()

        input_dropdown.bind("<<ComboboxSelected>>", on_input_select)

    def change_output_device(self):
        """Allows user to select an output device using a drop-down menu."""
        devices = sd.query_devices()
        output_devices = [dev["name"] for dev in devices if dev["max_output_channels"] > 0]

        output_dropdown = ttk.Combobox(
            self.output_device_label.master,
            values=output_devices,
            state="readonly",
            style="TCombobox",
            width=30,
        )
        output_dropdown.grid(row=1, column=1, padx=0, pady=0, ipadx=10, sticky="w")
        output_dropdown.set("Select")

        def on_output_select(event):
            selected_device = output_dropdown.get()
            if selected_device:
                device_config["output_device"] = selected_device
                self.output_device_label.config(text=self.truncate_device_name(selected_device))
            output_dropdown.destroy()

        output_dropdown.bind("<<ComboboxSelected>>", on_output_select)

    def select_device_popup(self, title, options):
        """Displays a popup for the user to select a device from a list."""
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("100x100")
        popup.transient(self)
        popup.grab_set()

        selected_device = tk.StringVar()

        listbox = tk.Listbox(popup, listvariable=tk.StringVar(value=options), height=15, selectmode="single")
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        self.wait_window(popup)

        return selected_device.get()

    def browse_model_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a JSON file",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as json_file:
                    model_data = json.load(json_file)

                required_models = ["Linear", "Quadratic", "Cubic", "Logarithmic", "Seg-Linear"]
                if not all(key in model_data for key in required_models):
                    raise ValueError("JSON file does not contain all required model definitions.")

                self.model_data = model_data
                self.reset_plots()
                messagebox.showinfo("Success", "Model file loaded successfully!")
                self.model_file_path.set(file_path)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON file: {e}")

    def estimate_volume(self, peak_freq):
        """
        Convert a resonance peak frequency into a volume (µL)
        using the loaded calibration model.
        """
        if not hasattr(self, "model_data") or not self.model_data:
            messagebox.showerror("Error", "No calibration data loaded.")
            return None

        model_name = self.model_selection.get()
        model_data = self.model_data.get(model_name)
        if not model_data:
            messagebox.showerror("Error", f"Selected model '{model_name}' not found in calibration data.")
            return None

        if model_name == "Linear":
            slope = model_data["Slope"]
            intercept = model_data["Intercept"]
            return slope * peak_freq + intercept

        elif model_name == "Quadratic":
            coeffs = model_data["Coefficients"]
            intercept = model_data["Intercept"]
            return intercept + sum(c * (peak_freq ** i) for i, c in enumerate(coeffs))

        elif model_name == "Cubic":
            coeffs = model_data["Coefficients"]
            intercept = model_data["Intercept"]
            return intercept + sum(c * (peak_freq ** i) for i, c in enumerate(coeffs))

        elif model_name == "Logarithmic":
            coefficient = model_data["Coefficient"]
            intercept = model_data["Intercept"]
            return intercept + coefficient * np.log(peak_freq)

        elif model_name == "Seg-Linear":
            for segment in model_data:
                if segment["Min Frequency"] <= peak_freq <= segment["Max Frequency"]:
                    slope = segment["Slope"]
                    intercept = segment["Intercept"]
                    return slope * peak_freq + intercept
            messagebox.showerror("Error", "Peak frequency out of range for 'Seg-Linear' model.")
            return None

        return None

    def update_results_display(self, peak_freq):
        """Update the Results Display with peak frequency and volume."""
        if not self.rms_value_label or not self.rms_value_label.winfo_exists():
            self.rms_value_label = ttk.Label(self.rms_value_label.master, textvariable=self.rms_value,
                                             font=("Arial", 14, "bold"))
            self.rms_value_label.pack(padx=5, pady=15)

        for widget in self.rms_value_label.master.winfo_children():
            widget.destroy()

        volume = None
        if not hasattr(self, "model_data") or not self.model_data:
            volume_info = "Volume: [Not available]"
        else:
            volume = self.estimate_volume(peak_freq)
            volume_info = f"Volume (µL): {volume:.5f}" if volume is not None else "Volume: [Not available]"

        model_name = self.est_model_selection.get()
        if model_name == "None":
            peak_text = f"Rec. Peak Freq.: {peak_freq:.2f} Hz"
        else:
            peak_text = f"Est. Peak Freq.: {peak_freq:.2f} Hz"

        volume_text = volume_info

        self.peak_freq = peak_freq
        self.volume = volume

        peak_label = tk.Label(
            self.rms_value_label.master,
            text=peak_text,
            font=("Arial", 14, "bold"),
            foreground="purple"
        )
        volume_label = tk.Label(
            self.rms_value_label.master,
            text=volume_text,
            font=("Arial", 14, "bold"),
            foreground="dark green"
        )

        peak_label.grid(row=0, column=0, padx=5, pady=10, sticky="e")
        volume_label.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        self.rms_value_label.master.grid_columnconfigure(0, weight=1)
        self.rms_value_label.master.grid_columnconfigure(1, weight=1)

    def bind_project_events(self):
        """Bind events to reset replicate counter on project/treatment change."""
        self.project_name.trace_add("write", self.reset_replicate)
        self.treatment_name.trace_add("write", self.reset_replicate)

    def reset_replicate(self, *args):
        """Reset replicate counter to 1 when project or treatment changes."""
        self.replicate_counter.set(1)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Save Folder")
        if folder:
            self.project_folder.set(folder)

    def initialize_project_folders(self):
        """Creates the main project folder and its subfolders."""
        project_name = self.project_name.get()
        if not project_name:
            messagebox.showerror("Error", "Project name cannot be empty.")
            return False

        project_path = os.path.join(self.project_folder.get(), project_name)
        plots_path = os.path.join(project_path, "Plots")
        wav_path = os.path.join(project_path, "wav")

        try:
            os.makedirs(plots_path, exist_ok=True)
            os.makedirs(wav_path, exist_ok=True)
            return project_path, plots_path, wav_path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create project folders: {e}")
            return False

    def generate_tone(self, frequency, duration=3.0, level=0.005, samplerate=44100):
        """Generates a sine wave tone."""
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        return level * np.sin(2 * np.pi * frequency * t)

    def validate_entries(self):
        """Validates user input entries."""
        errors = []

        try:
            freq_range = self.sound_frequency.get()
            if "-" not in freq_range:
                raise ValueError("Frequency range must be in the format 'min-max'.")

            freq_min, freq_max = map(float, freq_range.split("-"))
            if freq_min < 1 or freq_max <= freq_min:
                raise ValueError("Frequency range must have valid 'min-max' values with min >= 1 and max > min.")
        except ValueError as e:
            errors.append(f"Frequency range error: {e}")

        try:
            if not (0 <= self.playback_level.get() <= 1):
                errors.append("Playback level must be between 0 and 1.")
        except tk.TclError:
            errors.append("Playback level must be a valid number.")

        try:
            if self.sound_duration.get() <= 0:
                errors.append("Duration must be positive.")
        except tk.TclError:
            errors.append("Duration must be a valid number.")

        try:
            if self.lag_time.get() < 0:
                errors.append("Initial lag cannot be negative.")
        except tk.TclError:
            errors.append("Initial lag must be a valid number.")

        return errors

    def update_trimmed_data(self):
        """Updates the RMS data and plots based on the selected trim criteria."""
        if not hasattr(self, "rms_values") or not self.rms_values:
            return

        frequencies, rms_db_values = zip(*self.rms_values)

        freq_min = self.frequency_min.get()
        freq_max = self.frequency_max.get()

        percentile_threshold = self.percentile_value.get()
        rms_threshold = np.percentile(rms_db_values, percentile_threshold)

        self.trimmed_rms_values = [
            (f, rms) for f, rms in zip(frequencies, rms_db_values)
            if freq_min <= f <= freq_max and rms >= rms_threshold
        ]

        if len(self.trimmed_rms_values) < 3:
            self.rms_value.set("Insufficient data after trimming.")
            self.safe_configure_widget(self.rms_value_label, foreground="red")
            self.plot_rms_vs_frequency([])
            return

        self.plot_rms_vs_frequency(self.trimmed_rms_values)

        self.update_model_fit()

    def sync_frequency_entries(self, event=None):
        """Synchronizes the frequency range entries with the selected range."""
        if not hasattr(self, "trimmed_rms_values") or not self.trimmed_rms_values:
            return
        try:
            frequencies = [f for f, _ in self.trimmed_rms_values]
            self.frequency_min.set(min(frequencies))
            self.frequency_max.set(max(frequencies))
        except ValueError:
            pass

    def update_percentile_range(self, event=None):
        """Updates the RMS plot based on the percentile threshold."""
        self.percentile_slider.set(self.percentile_value.get())
        self.update_trimmed_data()

    def slider_update_percentile(self, value):
        """Updates the percentile entry and triggers trimming."""
        self.percentile_value.set(float(value))
        self.update_trimmed_data()

    def enable_span_selector(self):
        """Enables span selector for dynamic frequency trimming on the RMS plot."""

        def on_select(xmin, xmax):
            self.frequency_min.set(xmin)
            self.frequency_max.set(xmax)
            self.update_trimmed_data()

        self.span_selector = SpanSelector(
            self.rms_ax, on_select, "horizontal",
            useblit=True, interactive=True
        )

    def switch_to_percentile_method(self):
        """Switch to percentile-based trimming and reset the plot."""
        self.trimmed_rms_values = self.rms_values[:]
        self.plot_rms_vs_frequency(self.rms_values)
        self.percentile_slider.set(0)
        self.frequency_min.set(min(f for f, _ in self.rms_values))
        self.frequency_max.set(max(f for f, _ in self.rms_values))

    def switch_to_frequency_method(self):
        """Switch to frequency range-based trimming and reset the plot."""
        self.trimmed_rms_values = self.rms_values[:]
        self.plot_rms_vs_frequency(self.rms_values)
        self.frequency_min.set(min(f for f, _ in self.rms_values))
        self.frequency_max.set(max(f for f, _ in self.rms_values))
        self.percentile_slider.set(0)

    def reset_plots(self, cancelled=False):
        """Resets the RMS plot and restores the full dataset. Optionally marks waveform as cancelled."""
        if cancelled:
            self.waveform_ax.clear()
            self.waveform_ax.text(
                0.5, 0.5, "Canceled!",
                fontsize=30, color="grey", weight="bold",
                ha="center", va="center", transform=self.waveform_ax.transAxes
            )
            self.waveform_canvas.draw()

        if hasattr(self, "rms_values") and self.rms_values:
            self.trimmed_rms_values = self.rms_values[:]
            self.rms_ax.clear()
            self.plot_rms_vs_frequency(self.rms_values)
            self.update_model_fit()

        if hasattr(self, "span_selector") and self.span_selector:
            self.span_selector.disconnect_events()
        self.enable_span_selector()

        self.frequency_min.set(min(f for f, _ in self.rms_values))
        self.frequency_max.set(max(f for f, _ in self.rms_values))
        self.percentile_slider.set(0)
        self.percentile_value.set(0)

    def start_measurement(self, event=None):
        """
        Begin the scan: play tones, record audio, compute RMS
        dynamically, and update the plots in real time.
        """
        def measure():
            try:
                self._device_error_shown = False
                self.r2_est_model = None
                self.rmse_est_model = None
                self.rmse_value.set("RMSE: ---")
                self.safe_configure_widget(self.rmse_label, foreground="dark blue")
                self.clear_results_display()
                self.is_measuring = True
                self.run_button.config(text="Running...", state="disabled")
                self.cancel_button.config(state="normal")
                self.toggle_widgets_state("disabled", exclude=self.cancel_button)

                self.waveform_ax.clear()
                self.waveform_ax.text(
                    0.5, 0.5, "Please wait...",
                    fontsize=20, color="grey", weight='bold',
                    ha="center", va="center", transform=self.waveform_ax.transAxes
                )
                self.waveform_canvas.draw()

                errors = self.validate_entries()
                if errors:
                    messagebox.showerror("Input Validation Error", "\n".join(errors))
                    return

                freq_min, freq_max = map(float, self.sound_frequency.get().split("-"))
                resolution = self.frequency_resolution.get()
                if resolution <= 0:
                    raise ValueError("Frequency resolution must be greater than zero.")
                frequencies = np.arange(freq_min, freq_max + resolution, resolution)

                playback_duration = self.sound_duration.get()
                silence_duration = self.silence_duration.get()
                total_time = len(frequencies) * (playback_duration + silence_duration)
                remaining_time = total_time

                def update_remaining_time():
                    minutes, seconds = divmod(int(remaining_time), 60)
                    self.remaining_time.set(f"Remaining Time: {minutes:02}:{seconds:02}")

                update_remaining_time()

                recording_buffer = []

                samplerate = 44100
                playback_level = self.playback_level.get()
                lag_time = self.lag_time.get() if self.initial_lag.get() else 0
                self.rms_values = []

                if self._input_device_index is None or self._output_device_index is None:
                    def resolve_device(device_name):
                        devices = sd.query_devices()
                        matches = [idx for idx, dev in enumerate(devices) if dev["name"] == device_name]
                        return matches[0] if matches else None

                    self._input_device_index = resolve_device(device_config["input_device"])
                    self._output_device_index = resolve_device(device_config["output_device"])

                    if self._input_device_index is None or self._output_device_index is None:
                        messagebox.showerror("Device Error", "Input or Output device not found.")
                        return

                sd.default.device = (self._input_device_index, self._output_device_index)
                sd.default.samplerate = samplerate

                for i, frequency in enumerate(frequencies):
                    if not self.is_measuring:
                        self.reset_plots(cancelled=True)
                        return

                    try:
                        tone = self.generate_tone(frequency, playback_duration, playback_level, samplerate)
                        silence = np.zeros(int(silence_duration * samplerate), dtype=np.float32)

                        recorded = sd.playrec(tone, samplerate=samplerate, channels=1)
                        sd.wait()

                        recording_buffer.extend(recorded.flatten())

                        if i < len(frequencies) - 1:
                            recording_buffer.extend(silence)

                            sd.play(silence, samplerate=samplerate)
                            sd.wait()

                        lag_samples = int((lag_time / 1000) * samplerate)
                        recorded_for_rms = recorded[lag_samples:] if lag_time > 0 else recorded
                        rms_linear = np.sqrt(np.mean(recorded_for_rms ** 2))
                        rms_db = 20 * np.log10(rms_linear) if rms_linear > 0 else -np.inf
                        self.rms_values.append((frequency, rms_db))

                        num_samples = len(recorded)
                        samples_per_cycle = int(samplerate / frequency)
                        center_sample = num_samples // 2
                        start_sample = center_sample - (3 * samples_per_cycle // 2)
                        end_sample = center_sample + (3 * samples_per_cycle // 2)

                        if start_sample < 0 or end_sample > num_samples:
                            extracted_wave = recorded
                        else:
                            extracted_wave = recorded[start_sample:end_sample]

                        self.waveform_ax.clear()
                        self.waveform_ax.plot(extracted_wave, color="red")
                        self.waveform_canvas.draw()

                        self.plot_rms_vs_frequency(self.rms_values)

                        remaining_time -= (playback_duration + silence_duration)
                        update_remaining_time()

                    except Exception as e:
                        messagebox.showerror("Measurement Error", f"Error at {frequency} Hz: {e}")
                        break

                self.wave_data = np.array(recording_buffer, dtype=np.float32)

                if hasattr(self, "rms_values") and self.rms_values:
                    self.trimmed_rms_values = self.rms_values[:]
                    self.plot_rms_vs_frequency(self.rms_values)
                    self.update_model_fit()

                self.remaining_time.set("")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                self.after(0, self.complete_measurement)

                if self.is_measuring:
                    self.waveform_ax.clear()
                    self.waveform_ax.text(
                        0.5, 0.5, "Completed!",
                        fontsize=30, color="grey", weight='bold',
                        ha="center", va="center", transform=self.waveform_ax.transAxes
                    )
                    self.waveform_canvas.draw()

        threading.Thread(target=measure).start()


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

    def cancel_measurement(self):
        """Handles cancellation of the measurement process."""
        self.is_measuring = False
        self.reset_plots(cancelled=True)
        self.clear_results_display()
        self.cancel_button.config(state="disabled")

        playback_duration = self.sound_duration.get() * 1000
        self.after(int(playback_duration), lambda: self.run_button.config(state="normal"))

    def complete_measurement(self):
        """Resets the GUI after measurement is complete."""
        self.is_measuring = False
        self.run_button.config(text="Run", state="normal")
        self.cancel_button.config(state="disabled")
        self.toggle_widgets_state("normal", exclude=self.run_button)

        if self.is_measuring:
            self.waveform_ax.clear()
            self.waveform_ax.text(
                0.5, 0.5, "Completed!",
                fontsize=30, color="grey", weight="bold",
                ha="center", va="center", transform=self.waveform_ax.transAxes
            )
            self.waveform_canvas.draw()

    def toggle_widgets_state(self, state, exclude=None):
        """Recursively enables/disables all widgets in the application.

        Args:
            state (str): The state to set, e.g., "disabled" or "normal".
            exclude (Widget): Widget to exclude from disabling/enabling.
        """

        def recursive_toggle(widget):
            for child in widget.winfo_children():
                if child == exclude:
                    continue
                try:
                    child.configure(state=state)
                except (tk.TclError, AttributeError):
                    pass
                recursive_toggle(child)

        recursive_toggle(self)

        if state == "disabled":
            self.run_button.config(state="disabled")
            self.cancel_button.config(state="normal" if self.is_measuring else "disabled")
        elif state == "normal" and not self.is_measuring:
            self.run_button.config(state="normal")
            self.cancel_button.config(state="disabled")

    def clear_results_display(self):
        """Clears the Results Display frame entirely."""
        for widget in self.rms_value_label.master.winfo_children():
            widget.destroy()

        self.rms_value.set("")
        self.rms_value_label = ttk.Label(self.rms_value_label.master, textvariable=self.rms_value,
                                         font=("Arial", 14, "bold"))
        self.rms_value_label.pack(padx=5, pady=15)

    def fit_model_to_rms(self, frequencies, rms_values):
        """Fit the selected EST model to RMS vs Frequency data."""
        models = {
            "Gaussian": (self.gaussian, [max(rms_values), np.mean(frequencies), np.std(frequencies), min(rms_values)]),
            "Lorentzian": (
                self.lorentzian, [max(rms_values), np.mean(frequencies), np.std(frequencies), min(rms_values)]),
            "Asymmetric Lorentzian": (
                self.asymmetric_lorentzian,
                [max(rms_values), np.mean(frequencies), np.std(frequencies), min(rms_values), 0.1],
            ),
            "Voigt": (
                self.voigt,
                [
                    max(rms_values),
                    np.mean(frequencies),
                    (max(frequencies) - min(frequencies)) / 4,
                    (max(frequencies) - min(frequencies)) / 8,
                    min(rms_values)
                ]
            ),
            "Quadratic": (self.quadratic, [1, 1, np.mean(rms_values)]),
        }

        model_name = self.est_model_selection.get()
        if model_name not in models:
            messagebox.showerror("Error", f"Unknown model: {model_name}")
            return None

        model_func, initial_guess = models[model_name]

        maxfev = min(5000, len(frequencies) * 500)

        try:
            popt, _ = curve_fit(model_func, frequencies, rms_values, p0=initial_guess, maxfev=5000)
            if model_name in ["Gaussian", "Lorentzian", "Asymmetric Lorentzian", "Voigt"]:
                peak_freq = popt[1]
                peak_amp = model_func(peak_freq, *popt)
            elif model_name == "Quadratic":
                peak_freq = -popt[1] / (2 * popt[0]) if popt[0] != 0 else None
                peak_amp = model_func(peak_freq, *popt) if peak_freq else None

            return peak_freq, peak_amp, popt
        except Exception as e:
            messagebox.showerror("Fit Error", f"Failed to fit the model: {e}")
            return None

    def update_model_fit(self, event=None):
        """Update the model fit and RMS plot dynamically when the model selection changes."""
        if self.trimmed_rms_values:
            frequencies, rms_db_values = zip(*self.trimmed_rms_values)
        elif hasattr(self, "rms_values") and self.rms_values:
            frequencies, rms_db_values = zip(*self.rms_values)
        else:
            self.rms_value.set("No data available for fitting.")
            self.safe_configure_widget(self.rms_value_label, foreground="red")
            self.rmse_value.set("RMSE: ---")
            return

        model_name = self.est_model_selection.get()
        if model_name == "None":
            if frequencies and rms_db_values:
                max_rms_index = np.argmax(rms_db_values)
                recorded_peak_freq = frequencies[max_rms_index]
                self.update_results_display(recorded_peak_freq)
            else:
                self.rms_value.set("No valid data for peak calculation.")
                self.safe_configure_widget(self.rms_value_label, foreground="red")
            self.rmse_value.set("RMSE: ---")
            return

        results = self.fit_model_to_rms(frequencies, rms_db_values)
        if results:
            peak_freq, peak_amp, params = results

            models = {
                "Gaussian": self.gaussian,
                "Lorentzian": self.lorentzian,
                "Asymmetric Lorentzian": self.asymmetric_lorentzian,
                "Voigt": self.voigt,
                "Quadratic": self.quadratic,
            }
            model_func = models.get(model_name)
            if model_func:
                try:
                    predicted = model_func(np.array(frequencies), *params)
                    residuals = np.array(rms_db_values) - predicted
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((rms_db_values - np.mean(rms_db_values)) ** 2)
                    self.r2_est_model = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    self.rmse_est_model = np.sqrt(np.mean(residuals ** 2))

                    self.rmse_value.set(f"RMSE: {self.rmse_est_model:.7f}")
                except Exception as e:
                    self.r2_est_model = None
                    self.rmse_est_model = None
                    self.rmse_value.set("RMSE: ---")

                fitted_model = lambda x: model_func(x, *params)
                self.plot_rms_vs_frequency(
                    self.trimmed_rms_values, model_func=fitted_model, peak_freq=peak_freq, peak_amp=peak_amp
                )

            self.update_results_display(peak_freq)
        else:
            self.r2_est_model = None
            self.rmse_est_model = None
            self.rmse_value.set("RMSE: ---")
            self.rms_value.set("Model fitting failed.")
            self.safe_configure_widget(self.rms_value_label, foreground="red")

    def plot_rms_vs_frequency(self, rms_values, model_func=None, peak_freq=None, peak_amp=None):
        """Plots RMS vs Frequency with optional model trend and estimated peak."""
        try:
            self.rms_ax.clear()

            if not rms_values:
                self.rms_ax.text(
                    0.5, 0.5, "No data available.",
                    fontsize=12, color="red", weight="bold",
                    ha="center", va="center", transform=self.rms_ax.transAxes
                )
                self.rms_canvas.draw()
                return

            frequencies, rms_db = zip(*rms_values)

            self.rms_ax.scatter(frequencies, rms_db, color="blue")
            self.rms_ax.tick_params(axis='both', labelcolor="grey", color="grey", labelsize=8)

            for spine in self.rms_ax.spines.values():
                spine.set_color('white')

            if model_func is not None:
                fine_freqs = np.linspace(min(frequencies), max(frequencies), 1000)
                model_values = model_func(fine_freqs)
                self.rms_ax.plot(fine_freqs, model_values, color="red", linewidth=2)

            if peak_freq is not None and peak_amp is not None:
                self.rms_ax.scatter([peak_freq], [peak_amp], color="green", s=100, zorder=5)

            self.rms_canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", f"An error occurred while updating RMS plot: {e}")

    def plot_waveform(self, waveform_data):
        """Plots the waveform."""
        try:
            self.waveform_ax.clear()
            self.waveform_ax.plot(waveform_data)
            self.waveform_ax.set_title("Waveform")
            self.waveform_ax.set_xlabel("Time")
            self.waveform_ax.set_ylabel("Amplitude")
            self.waveform_ax.grid(True)
            self.waveform_canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot the waveform: {e}")

    def save_results(self):
        """
        Validate inputs and results, then save plots, audio,
        and CSV/JSON logs. Increments the replicate counter.
        """
        if not self.project_folder.get():
            messagebox.showerror("Error", "Please select a Save Folder in the Project frame.")
            return

        if not hasattr(self, "rms_values") or not self.rms_values:
            messagebox.showerror("Error", "No results to save. Please complete a measurement first.")
            return

        if not isinstance(self.replicate_counter.get(), int) or self.replicate_counter.get() < 1:
            messagebox.showerror("Error", "Replicate must be an integer greater than or equal to 1.")
            return

        paths = self.initialize_project_folders()
        if not paths:
            return

        project_path, plots_path, wav_path = paths
        project_name = self.project_name.get()
        treatment = self.treatment_name.get() or "Default"
        replicate = self.replicate_counter.get()

        self.span_selector.disconnect_events()

        try:
            rms_filename = f"Plot_{project_name}_{treatment}_{replicate}.png"
            rms_filepath = os.path.join(plots_path, rms_filename)
            self.rms_fig.savefig(rms_filepath, dpi=600)

            wav_filename = f"{project_name}_{treatment}_{replicate}.wav"
            wav_filepath = os.path.join(wav_path, wav_filename)
            with wave.open(wav_filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(np.array(self.wave_data * 32767, dtype=np.int16).tobytes())

            calib_file = self.model_file_path.get() or "Not Available"

            est_peak_freq = f"{self.peak_freq:.2f}" if hasattr(self,
                                                               "peak_freq") and self.peak_freq is not None else "Not Available"
            est_volume = f"{self.volume:.5f}" if hasattr(self,
                                                         "volume") and self.volume is not None else "Not Available"

            r2_est_model = f"{self.r2_est_model:.5f}" if hasattr(self, "r2_est_model") else "Not Available"
            rmse_est_model = f"{self.rmse_est_model:.5f}" if hasattr(self, "rmse_est_model") else "Not Available"

            csv_filename = f"{project_name}.csv"
            csv_filepath = os.path.join(project_path, csv_filename)
            is_new_file = not os.path.exists(csv_filepath)

            with open(csv_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if is_new_file:
                    writer.writerow([
                        "Project", "Treatment", "Replicate", "Date and time", "Output device", "Input device",
                        "Calib. File", "Calib. EST model", "R2 Calib. EST model", "RMSE Calib. EST model",
                        "Calib. Overall model", "Frequency range (Hz)",
                        "Playback level (0-1)", "Frequency resolution (Hz)", "Playback duration (s)",
                        "Silence duration (s)", "Excluded initial lag (ms)", "Trim. freq. range (Hz)",
                        "Trim. Exc. RMS percentile", "EST Peak frequency (Hz)", "EST Volume (µL)"
                    ])
                writer.writerow([
                    project_name, treatment, replicate, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    device_config.get('output_device'), device_config.get('input_device'), calib_file,
                    self.est_model_selection.get(), r2_est_model, rmse_est_model,
                    self.model_selection.get(), self.sound_frequency.get(), self.playback_level.get(),
                    self.frequency_resolution.get(), self.sound_duration.get(),
                    self.silence_duration.get(), self.lag_time.get() if self.initial_lag.get() else 0,
                    f"{self.frequency_min.get()}-{self.frequency_max.get()}",
                    self.percentile_value.get(), est_peak_freq, est_volume
                ])

            self.replicate_counter.set(self.replicate_counter.get() + 1)

            messagebox.showinfo("Success", f"Results saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")
        finally:
            self.enable_span_selector()
            self.rms_canvas.draw_idle()

    def save_last_settings(self):
        """Saves all the settings in a JSON file."""
        config = {
            "project_name": self.project_name.get(),
            "treatment_name": self.treatment_name.get(),
            "replicate_counter": self.replicate_counter.get(),
            "project_folder": self.project_folder.get(),
            "sound_frequency": self.sound_frequency.get(),
            "playback_level": self.playback_level.get(),
            "frequency_resolution": self.frequency_resolution.get(),
            "sound_duration": self.sound_duration.get(),
            "silence_duration": self.silence_duration.get(),
            "initial_lag": self.initial_lag.get(),
            "lag_time": self.lag_time.get(),
            "input_device": device_config.get("input_device", "Unknown Device"),
            "output_device": device_config.get("output_device", "Unknown Device"),
            "model_file_path": self.model_file_path.get(),
            "est_model_selection": self.est_model_selection.get(),
            "model_selection": self.model_selection.get(),
        }

        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            pass

    def load_last_settings(self):
        """Loads settings from the JSON file, if available."""
        if not os.path.exists(CONFIG_FILE):
            return

        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)

            self.project_name.set(config.get("project_name", ""))
            self.treatment_name.set(config.get("treatment_name", ""))
            self.replicate_counter.set(config.get("replicate_counter", 1))
            self.project_folder.set(config.get("project_folder", ""))
            self.sound_frequency.set(config.get("sound_frequency", "600-1200"))
            self.playback_level.set(config.get("playback_level", 0.005))
            self.frequency_resolution.set(config.get("frequency_resolution", 100))
            self.sound_duration.set(config.get("sound_duration", 3.0))
            self.silence_duration.set(config.get("silence_duration", 0.5))
            self.initial_lag.set(config.get("initial_lag", True))
            self.lag_time.set(config.get("lag_time", 50))
            self.model_file_path.set(config.get("model_file_path", ""))
            self.est_model_selection.set(config.get("est_model_selection", "Lorentzian"))
            self.model_selection.set(config.get("model_selection", "Seg-Linear"))

            self._input_device_index = config.get("input_device", None)
            self._output_device_index = config.get("output_device", None)

            devices = sd.query_devices()
            input_name = devices[self._input_device_index][
                "name"] if self._input_device_index is not None else "Unknown Device"
            output_name = devices[self._output_device_index][
                "name"] if self._output_device_index is not None else "Unknown Device"
            self.input_device_label.config(text=input_name)
            self.output_device_label.config(text=output_name)

        except Exception as e:
            pass

    def on_exit(self):
        """Confirm exit before closing the application."""
        confirm = messagebox.askyesno(
            "Exit",
            "Are you sure you want to quit?",
            icon="warning"
        )
        if confirm:
            self.save_last_settings()
            self.destroy()
            os._exit(0)

if __name__ == "__main__":
    app = Measurement()

    device_config = load_device_configuration()

    if config_missing:
        app.after(200, lambda: show_missing_config_warning(app))

    app.mainloop()
