#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Single Tone Sampler Module
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
single_tone.py

GUI for generating and recording a single-frequency test tone,
displaying the waveform and RMS (dB), and saving results.

Usage:
    python single_tone.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import csv
import os
import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys, os

def resource(rel_path):
    """
    Return the absolute path to a resource, whether running from
    source or from a PyInstaller one‐file bundle.
    """
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

DEVICE_CONFIG_FILE    = resource("device_config.json")
SINGLETON_CONFIG_FILE = resource("singletone_config.json")

def load_device_configuration():
    """
    Attempt to read singleton_config → device_config, validate
    against actual sound devices, and return a dict:
      {'input_device': ..., 'output_device': ...}
    """
    global config_missing
    config_missing = True
    config = {"input_device": "Unknown Device", "output_device": "Unknown Device"}

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

    return validate_device_config(config)

def validate_device_config(config):
    """
    Ensure the requested input/output names exist on this machine,
    replacing invalid entries with "Unknown Device".
    """
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
        print(f"Could not load icon: {e}")

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


CONFIG_FILE = resource("singletone_config.json")


class SingleToneSampler(tk.Tk):
    """
    Main window for:
      - Project metadata (name, treatment, replicate, folder)
      - Device selection / test-tone playback
      - Single-frequency tone playback + recording
      - Display of 3-cycle waveform & RMS value
      - Saving results and incrementing replicate
    """

    def __init__(self):
        """
        Initialize the Tk window, load last settings, prepare Matplotlib
        canvases, and call create_widgets().
        """
        super().__init__()
        self.withdraw()
        self.title("Single Tone Sampler - Acoustic Volumeter v. 1.0")
        self.geometry("670x630")
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

        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.bind_shortcuts()
        self.sound_frequency = tk.DoubleVar(value=500)
        self.playback_level = tk.DoubleVar(value=0.005)
        self.sound_duration = tk.DoubleVar(value=5.0)
        self.initial_lag = tk.BooleanVar(value=True)
        self.lag_time = tk.IntVar(value=50)
        self.project_name = tk.StringVar()
        self.treatment_name = tk.StringVar()
        self.replicate_counter = tk.IntVar(value=1)
        self.project_folder = tk.StringVar()
        self.is_measuring = False
        self.rms_context_menu = tk.Menu(self, tearoff=0)
        self.rms_context_menu.add_command(label="Clear plot", command=self.clear_rms_plot)

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

        self.rms_fig, self.rms_ax = plt.subplots(figsize=(2, 3))
        self.rms_fig.set_layout_engine('constrained')
        self.rms_ax.axis("off")

        self.create_widgets()

    def create_widgets(self):
        """
        Build all UI sections: Project info, Device frame,
        Playback settings, Waveform and RMS frames, Action buttons.
        """
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

        output_device_name = device_config['output_device'][:45] + "..." if len(
            device_config['output_device']) > 45 else device_config['output_device']
        input_device_name = device_config['input_device'][:45] + "..." if len(device_config['input_device']) > 45 else \
            device_config['input_device']

        right_column_frame = ttk.Frame(self)
        right_column_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="news")
        right_column_frame.columnconfigure(0, weight=1)

        device_frame = ttk.LabelFrame(right_column_frame, text="Detected Devices", width=300, height=100)
        device_frame.grid(row=0, column=1, columnspan=1, padx=10, pady=5, sticky="new")
        device_frame.grid_propagate(False)
        device_frame.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.configure("Custom.TButton", font=("TkDefaultFont", 8))

        input_device_name = str(device_config.get('input_device', "Unknown Device"))
        output_device_name = str(device_config.get('output_device', "Unknown Device"))

        input_device_name = input_device_name[:45] + "..." if len(input_device_name) > 45 else input_device_name
        output_device_name = output_device_name[:45] + "..." if len(output_device_name) > 45 else output_device_name

        self.input_device_label = ttk.Label(device_frame, text=f"{input_device_name}", width=35, anchor="w")
        self.input_device_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.output_device_label = ttk.Label(device_frame, text=f"{output_device_name}", width=35, anchor="w")
        self.output_device_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        input_change_button = ttk.Button(device_frame, text="In", command=self.change_input_device,
                                         style="Custom.TButton")
        input_change_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        output_change_button = ttk.Button(device_frame, text="Out", command=self.change_output_device,
                                          style="Custom.TButton")
        output_change_button.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        playback_waveform_frame = ttk.Frame(self)
        playback_waveform_frame.grid(row=1, column=0, padx=10, pady=5, sticky="news")
        playback_waveform_frame.columnconfigure(0, weight=1)

        playback_frame = ttk.LabelFrame(playback_waveform_frame, text="Playback Settings")
        playback_frame.grid(row=1, column=0, padx=5, pady=5, sticky="new")
        playback_frame.columnconfigure(1, weight=1)
        ttk.Label(playback_frame, text="Playback Frequency (Hz):").grid(row=0, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.sound_frequency, width=15).grid(row=0, column=1, padx=5, pady=5, sticky="ns")
        ttk.Label(playback_frame, text="Playback Level (0-1):").grid(row=1, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.playback_level, width=15).grid(row=1, column=1, padx=5, pady=5, sticky="ns")
        ttk.Label(playback_frame, text="Playback Duration (s):").grid(row=3, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(playback_frame, textvariable=self.sound_duration, width=15).grid(row=3, column=1, padx=5, pady=5, sticky="ns")
        ttk.Checkbutton(playback_frame, text="Exclude Initial Lag (ms):", variable=self.initial_lag).grid(row=5,
                                                                                                          column=0,
                                                                                                          padx=15,
                                                                                                          pady=5,
                                                                                                          sticky="w")
        ttk.Entry(playback_frame, textvariable=self.lag_time, width=15).grid(row=5, column=1, padx=5, pady=5, sticky="ns")

        waveform_frame = ttk.LabelFrame(playback_waveform_frame, text="Waveform")
        waveform_frame.grid(row=2, column=0, padx=10, pady=5, sticky="news")
        waveform_frame.columnconfigure(0, weight=1)
        waveform_frame.rowconfigure(0, weight=1)
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, master=waveform_frame)
        self.waveform_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        rms_frame = ttk.LabelFrame(right_column_frame, text="RMS vs Frequency")
        rms_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ns")

        self.rms_canvas = FigureCanvasTkAgg(self.rms_fig, master=rms_frame)
        self.rms_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="ns")
        self.rms_canvas.get_tk_widget().bind("<Button-3>", self.show_rms_context_menu)

        rms_frame = ttk.LabelFrame(self, text="Result Display", width=180, height=100)
        rms_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        rms_frame.grid_propagate(False)

        self.rms_value = tk.StringVar(value="")  # Initialize variable
        tk.Label(rms_frame, textvariable=self.rms_value, font=("Arial", 14, "bold"), fg="darkblue").pack(padx=5,
                                                                                                         pady=15)

        action_frame = ttk.Frame(self)
        action_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

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


    def bind_shortcuts(self):
        """Bind keyboard shortcuts to GUI actions."""
        self.bind("<Control-Return>", lambda event: self.start_measurement())
        self.bind("<Control-s>", lambda event: self.save_results())
        self.bind("<Control-r>", lambda event: self.clear_rms_plot())
        self.bind("<Escape>", lambda event: self.cancel_measurement())

    @staticmethod
    def truncate_device_name(name, max_length=40):
        """Truncates a device name if it exceeds the maximum length."""
        if not isinstance(name, str):
            return "Unknown Device"
        return name if len(name) <= max_length else name[:max_length] + "..."


    def play_test_tone(self):
        """
        Play a short 1 kHz tone so the user can verify the output device.
        """
        try:
            test_tone = self.generate_tone(frequency=1000, duration=0.3, level=0.005)
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
            frequency = self.sound_frequency.get()
            if frequency < 1:
                errors.append("Frequency must be at least 1 Hz.")
        except tk.TclError:
            errors.append("Frequency must be a valid number.")

        try:
            if not (0 <= self.playback_level.get() <= 1):
                errors.append("Playback level must be between 0 and 1.")
        except tk.TclError:
            errors.append("Playback level must be a valid number.")

        try:
            if self.sound_duration.get() <= 0:
                errors.append("Duration must be greater than zero.")
        except tk.TclError:
            errors.append("Duration must be a valid number.")

        try:
            if self.lag_time.get() < 0:
                errors.append("Initial lag cannot be negative.")
        except tk.TclError:
            errors.append("Initial lag must be a valid number.")

        return errors

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

    def start_measurement(self):
        """
        Generate, play & record the single test tone, compute RMS (dB),
        plot the 3-cycle waveform, and display the numeric RMS.
        """
        try:
            errors = self.validate_entries()
            if errors:
                messagebox.showerror("Input Validation Error", "\n".join(errors))
                return

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
            self.update_idletasks()

            frequency = self.sound_frequency.get()
            duration = self.sound_duration.get()
            playback_level = self.playback_level.get()
            samplerate = 44100
            lag_time = self.lag_time.get() if self.initial_lag.get() else 0

            if not self._device_error_shown:
                def resolve_device(device_name):
                    devices = sd.query_devices()
                    matches = [idx for idx, dev in enumerate(devices) if dev["name"] == device_name]
                    return matches[0] if matches else None

                self._input_device_index = resolve_device(device_config["input_device"])
                self._output_device_index = resolve_device(device_config["output_device"])

                device_errors = []
                if self._input_device_index is None:
                    device_errors.append(f"Input device '{device_config['input_device']}' not found.")
                if self._output_device_index is None:
                    device_errors.append(f"Output device '{device_config['output_device']}' not found.")

                if device_errors:
                    messagebox.showerror("Device Error", "\n".join(device_errors))
                    self._device_error_shown = True
                    return

            sd.default.device = (self._input_device_index, self._output_device_index)
            sd.default.samplerate = samplerate

            tone = self.generate_tone(frequency, duration, playback_level, samplerate)
            recorded = sd.playrec(tone, samplerate=samplerate, channels=1)
            sd.wait()

            lag_samples = int((lag_time / 1000) * samplerate)
            recorded_for_rms = recorded[lag_samples:] if lag_time > 0 else recorded
            rms_linear = np.sqrt(np.mean(recorded_for_rms ** 2))
            rms_db = 20 * np.log10(rms_linear) if rms_linear > 0 else -np.inf
            self.rms_value.set(f"RMS (dB): {rms_db:.12f}")
            self.numeric_rms_value = rms_db
            self.wave_data = recorded.flatten()
            current_replicate = self.replicate_counter.get()

            self.plot_waveform(recorded.flatten(), frequency, samplerate)

            self.update_rms_plot(frequency, rms_db)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.complete_measurement()

    def plot_waveform(self, waveform_data, frequency, samplerate=44100):
        """
        Extract exactly three cycles from the center of `waveform_data`
        and plot it in red with all axes/decorations turned off.
        """
        try:
            self.waveform_ax.clear()

            samples_per_cycle = int(samplerate / frequency)
            total_samples = len(waveform_data)

            center_sample = total_samples // 2
            start_sample = max(0, center_sample - (3 * samples_per_cycle // 2))
            end_sample = min(total_samples, center_sample + (3 * samples_per_cycle // 2))
            extracted_wave = waveform_data[start_sample:end_sample]

            self.waveform_ax.plot(extracted_wave, color="red")

            self.waveform_ax.axis("off")

            self.waveform_canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot the waveform: {e}")

    def cancel_measurement(self):
        """
        Abort an in‐progress tone measurement, clear the waveform,
        and roll back the most recent RMS point.
        """
        self.is_measuring = False
        self.reset_plots(cancelled=True)

        if hasattr(self, 'rms_history') and self.rms_history:
            self.rms_history.pop()

        self.update_rms_plot()

        self.cancel_button.config(state="disabled")
        self.run_button.config(state="normal")
        self.toggle_widgets_state("normal", exclude=self.run_button)

    def complete_measurement(self):
        """
        Reset Run/Cancel buttons and re‐enable widgets after a tone
        has finished or been canceled.
        """
        self.is_measuring = False
        self.run_button.config(text="Run", state="normal")
        self.cancel_button.config(state="disabled")
        self.toggle_widgets_state("normal", exclude=self.run_button)

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

        if exclude:
            exclude.config(state="normal")
        self.update_idletasks()

    def show_rms_context_menu(self, event):
        """Displays the context menu on right-click."""
        self.rms_context_menu.post(event.x_root, event.y_root)

    def clear_rms_plot(self):
        """
        Confirm with the user, then clear the RMS plot and history.
        """
        response = messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the RMS plot?")
        if response:
            self.rms_ax.clear()
            self.rms_ax.set_xticks([])
            self.rms_ax.set_xticklabels([])
            self.rms_canvas.draw()

            if hasattr(self, 'rms_history'):
                self.rms_history = []
            self.rms_value.set("")

    def clear_results_display(self):
        """
        Clear all widgets in the results frame and recreate the RMS label.
        """
        for widget in self.rms_value_label.master.winfo_children():
            widget.destroy()

        self.rms_value.set("")
        self.rms_value_label = ttk.Label(self.rms_value_label.master, textvariable=self.rms_value,
                                         font=("Arial", 14, "bold"))
        self.rms_value_label.pack(padx=5, pady=15)

    def update_rms_plot(self, frequency=None, rms_db=None):
        """
        Append [frequency, rms_db] to history (if given), then redraw the
        RMS-vs-frequency plot showing previous (grey) and current (blue) points.
        """
        try:
            if not hasattr(self, 'rms_history'):
                self.rms_history = []

            if frequency is not None and rms_db is not None:
                self.rms_history.append((frequency, rms_db))

            self.rms_ax.clear()

            if len(self.rms_history) > 1:
                previous_data = self.rms_history[:-1]
                prev_frequencies, prev_rms_values = zip(*previous_data)
                self.rms_ax.scatter(prev_frequencies, prev_rms_values, color="grey", alpha=0.6,
                                    label="Previous Measurements")

            if self.rms_history:
                current_frequency, current_rms_db = self.rms_history[-1]
                self.rms_ax.scatter([current_frequency], [current_rms_db], color="blue", label="Current Measurement")

            if self.rms_history:
                self.rms_ax.set_xticks([self.rms_history[-1][0]])
                self.rms_ax.set_xticklabels([f"{self.rms_history[-1][0]:.0f} Hz"], fontsize=8,
                                            color="grey")

            self.rms_ax.tick_params(axis='both', labelcolor="grey", color="grey", labelsize=8)
            for spine in self.rms_ax.spines.values():
                spine.set_color('white')  # Set borders to white

            self.rms_canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to update RMS plot: {e}")

    def plot_waveform(self, waveform_data, frequency, samplerate=44100):
        """
        Extract exactly three cycles from the center of `waveform_data` and plot
        them in red with axes turned off.
        """
        try:
            self.waveform_ax.clear()

            samples_per_cycle = int(samplerate / frequency)
            total_samples = len(waveform_data)

            center_sample = total_samples // 2
            start_sample = max(0, center_sample - (3 * samples_per_cycle // 2))
            end_sample = min(total_samples, center_sample + (3 * samples_per_cycle // 2))
            extracted_wave = waveform_data[start_sample:end_sample]

            self.waveform_ax.plot(extracted_wave, color="red")

            self.waveform_ax.axis("off")

            self.waveform_canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot the waveform: {e}")

    def save_results(self):
        """
        Validate project and folder settings, then export:
          - RMS plot as PNG
          - Waveform as WAV
          - Summary CSV
        and increment the replicate counter.
        """
        if not self.project_folder.get():
            messagebox.showerror("Error", "Please select a Save Folder in the Project frame.")
            return

        if not self.project_name.get() or not self.treatment_name.get():
            messagebox.showerror("Error", "Project name and treatment name cannot be empty.")
            return

        if not isinstance(self.replicate_counter.get(), int) or self.replicate_counter.get() < 1:
            messagebox.showerror("Error", "Replicate must be an integer greater than or equal to 1.")
            return

        project_name = self.project_name.get()
        treatment = self.treatment_name.get()
        replicate = self.replicate_counter.get()

        project_path = os.path.join(self.project_folder.get(), f"{project_name}")
        plots_path = os.path.join(project_path, "Plots")
        wav_path = os.path.join(project_path, "Wav")

        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(wav_path, exist_ok=True)

        try:
            rms_filename = f"{project_name}_{treatment}_{replicate}.png"
            rms_filepath = os.path.join(plots_path, rms_filename)
            self.rms_fig.savefig(rms_filepath, dpi=600)

            wav_filename = f"{project_name}_{treatment}_{replicate}.wav"
            wav_filepath = os.path.join(wav_path, wav_filename)

            if not hasattr(self, 'wave_data') or self.wave_data is None:
                messagebox.showerror(
                    "Error",
                    "Wave data is not available. This might occur if the measurement process didn't complete successfully.\n\n" \
                    "Please ensure that the measurement process is run completely before saving."
                )
                return

            with wave.open(wav_filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(np.array(self.wave_data * 32767, dtype=np.int16).tobytes())

            csv_filename = f"{project_name}.csv"
            csv_filepath = os.path.join(project_path, csv_filename)
            is_new_file = not os.path.exists(csv_filepath)

            with open(csv_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if is_new_file:
                    writer.writerow([
                        "Treatment", "Replicate", "Date and time", "Output device", "Input device",
                        "Playback Frequency (Hz)", "Playback level (0-1)", "Playback duration (s)",
                        "Excluded initial lag (ms)", "RMS (dB)"
                    ])

                rms_db = getattr(self, 'numeric_rms_value', None)
                if rms_db is None:
                    rms_db = "N/A"
                writer.writerow([
                    treatment, replicate, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    device_config.get('output_device', 'Unknown'), device_config.get('input_device', 'Unknown'),
                    self.sound_frequency.get(), self.playback_level.get(), self.sound_duration.get(),
                    self.lag_time.get() if self.initial_lag.get() else 0, rms_db
                ])

            self.replicate_counter.set(self.replicate_counter.get() + 1)

            messagebox.showinfo("Success",
                                f"Results saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def save_last_settings(self):
        """
        Dump current UI settings (project, tone, lag, devices) to JSON.
        """
        config = {
            "project_name": self.project_name.get(),
            "treatment_name": self.treatment_name.get(),
            "replicate_counter": self.replicate_counter.get(),
            "project_folder": self.project_folder.get(),
            "sound_frequency": self.sound_frequency.get(),
            "playback_level": self.playback_level.get(),
            "sound_duration": self.sound_duration.get(),
            "initial_lag": self.initial_lag.get(),
            "lag_time": self.lag_time.get(),
            "input_device": device_config.get("input_device", "Unknown Device"),
            "output_device": device_config.get("output_device", "Unknown Device"),

        }

        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            pass

    def load_last_settings(self):
        """
        Restore saved UI settings from JSON, if present.
        """
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
            self.sound_duration.set(config.get("sound_duration", 3.0))
            self.initial_lag.set(config.get("initial_lag", True))
            self.lag_time.set(config.get("lag_time", 50))
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
        """Save settings and close the application."""
        if messagebox.askyesno("Exit", "Are you sure you want to quit?"):
           self.save_last_settings()
           self.destroy()
           os._exit(0)

if __name__ == "__main__":
    app = SingleToneSampler()

    device_config = load_device_configuration()

    if config_missing:
        app.after(200, lambda: show_missing_config_warning(app))

    app.mainloop()