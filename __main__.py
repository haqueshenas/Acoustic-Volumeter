#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Main Launcher
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
__main__.py

Constructs the main Tkinter GUI for:
  - Animated logo
  - “About” dialog
  - Buttons to launch each tool module

Usage:
    python __main__.py
"""

try:
    import pyi_splash
except ImportError:
    pyi_splash = None

# ── Core Python libs ───────────────────────────────────────────────────────────
import sys
import os
import subprocess
import json
import csv
import io
import threading
import queue
import shutil
import time
import tempfile
import warnings
from datetime import datetime

# ── Tkinter & friends ──────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Scale, TclError
from tkinterdnd2 import TkinterDnD, DND_FILES    # for wave_viewer drag-and-drop

# ── Pillow (images) ─────────────────────────────────────────────────────────────
from PIL import Image, ImageTk

# ── NumPy / Pandas ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── SciPy ──────────────────────────────────────────────────────────────────────
import scipy
import scipy.optimize
import scipy.optimize._minpack
import scipy.signal
import scipy.io
import scipy.io.wavfile

# ── Sound I/O ──────────────────────────────────────────────────────────────────
import sounddevice as sd
import _sounddevice_data             # PortAudio binaries for sounddevice
import soundfile as sf
import wave

# ── scikit-learn ────────────────────────────────────────────────────────────────
import sklearn
import sklearn.preprocessing         # PolynomialFeatures
import sklearn.linear_model          # LinearRegression
import sklearn.metrics               # mean_squared_error

# ── Matplotlib ─────────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

# ── Ensure runpy is seen ────────────────────────────────────────────────────────
import runpy

# ─── MODULE DISPATCH ─────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    script_name = sys.argv[1]
    import runpy, os, sys

    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(base, script_name)

    runpy.run_path(script_path, run_name="__main__")
    sys.exit(0)
# ─────────────────────────────────────────────────────────────────────────────

def resource(rel_path):
    """
    Return the absolute path to a resource (image, icon, or sub‐script),
    whether running from source or from a PyInstaller one‐file bundle.
    """
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

subprocess_data = {"active": False, "process": None}

def show_about():
    """
    Display a modal “About” dialog with application name, version,
    description, VPL link, copyright, license, and contact email.
    """
    about_text = (
        "Acoustic Volumeter v1.0\n\n"
        "An open-source, DIY, microphone-free acoustic volumetry platform "
        "for rapid, precise volumetric measurements\n —designed for global "
        "phenotyping applications.\n\n"
        "This software is part of the Easy Phenotyping Lab (EPL)—\n"
        "a non-profit initiative aimed at sharing open, affordable,\n"
        "and reliable computational tools for plant and crop phenotyping "
        "research.\n"
        "Visit https://haqueshenas.github.io/EPL\n\n"
        "© 2025 Abbas Haghshenas\n"
        "Licensed under the MIT License\n\n"
        "Contact: haqueshenas@gmail.com"
    )
    messagebox.showinfo("About Acoustic Volumeter", about_text)

def set_button_states(state):
    """
    Enable or disable all toolkit buttons.

    Parameters:
        state (str): Either 'normal' or 'disabled', passed
                     to each button’s .config(state=…).
    """
    for button in buttons.values():
        button.config(state=state)

def terminate_subprocess():
    """
    If a module subprocess is running, terminate it cleanly (or kill
    after timeout), reset subprocess_data, re-enable buttons, and
    restart the logo animation if appropriate.
    """
    if subprocess_data["active"] and subprocess_data["process"].poll() is None:
        subprocess_data["process"].terminate()
        try:
            subprocess_data["process"].wait(timeout=2)
        except subprocess.TimeoutExpired:
            subprocess_data["process"].kill()
    subprocess_data["active"] = False
    subprocess_data["process"] = None
    set_button_states(tk.NORMAL)

    if all(button.cget('state') == 'normal' for button in buttons.values()):
        header_frame.after(50, animate_logo)

def open_module(module_filename):
    """
    Launch the given module (e.g. "measurement.py") by
    re-invoking this same EXE with the module’s .py path.
    """
    def monitor_process():
        if subprocess_data["process"].poll() is not None:
            subprocess_data["active"] = False
            subprocess_data["process"] = None
            set_button_states(tk.NORMAL)
            if all(b.cget('state') == 'normal' for b in buttons.values()):
                header_frame.after(50, animate_logo)
        else:
            root.after(100, monitor_process)

    set_button_states(tk.DISABLED)

    mod_path = resource(module_filename)

    try:
        p = subprocess.Popen(
            [sys.executable, mod_path],
            close_fds=True,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt" else 0
            ),
        )
        subprocess_data["active"] = True
        subprocess_data["process"] = p
        root.after(100, monitor_process)

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Could not launch module {module_filename!r}: {e}"
        )
        set_button_states(tk.NORMAL)

root = tk.Tk()
root.withdraw()
root.title("Acoustic Volumeter v.1")
root.geometry("800x620")
root.resizable(False, False)
root.iconbitmap(resource("AV.ico"))

root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 800) // 2
y = (screen_height - 620) // 2
root.geometry(f"800x620+{x}+{y}")

root.attributes('-topmost', True)
root.after(0, lambda: root.attributes('-topmost', False))  # Reset topmost after bringing it to the front

root.deiconify()
if pyi_splash:
    pyi_splash.close()

def close_main_window():
    """
    Handle the window-close event: if a module subprocess is active,
    terminate it, then destroy the main Tk window.
    """
    if subprocess_data["active"]:
        terminate_subprocess()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", close_main_window)

header_frame = tk.Frame(root, bg="#57863C", height=100)
header_frame.pack(side=tk.TOP, fill=tk.X)

content_frame = tk.Frame(root, bg="white")
content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)

footer_frame = tk.Frame(root, bg="#57863C")
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(0, 0))

def rotate_logo(image_path, angle=0):
    """
    Load the given image file, resize it to 50×50, rotate it by the
    specified angle, and return a PhotoImage for Tkinter.

    Parameters:
        image_path (str): Path to the logo image (e.g. "AV.png").
        angle (int):      Degrees to rotate clockwise (default 0).

    Returns:
        ImageTk.PhotoImage: Rotated, resized logo.
    """
    img = Image.open(image_path).resize((50, 50), Image.Resampling.LANCZOS)
    rotated_image = img.rotate(angle, resample=Image.Resampling.BICUBIC)
    return ImageTk.PhotoImage(rotated_image)

current_angle = 0
animation_running = False

def animate_logo():
    """
    Continuously rotate the header logo while no module is running.
    If any button is disabled, reset the logo to 0° and pause rotation.
    """
    global current_angle, about_image_tk
    if all(button.cget('state') == 'normal' for button in buttons.values()):
        current_angle = (current_angle - 33) % 360
        about_image_tk = rotate_logo(resource("AV.png"), current_angle)
        about_button.config(image=about_image_tk)
        header_frame.after(70, animate_logo)
    else:
        current_angle = 0
        about_image_tk = rotate_logo(resource("AV.png"), current_angle)
        about_button.config(image=about_image_tk)

try:
    about_image_tk = rotate_logo(resource("AV.png"))
    about_button = tk.Button(
        header_frame, image=about_image_tk, command=show_about, bg="#57863C", relief="flat", borderwidth=0
    )
    about_button.pack(side=tk.LEFT, padx=20)
    header_frame.after(50, animate_logo)
except FileNotFoundError:
    about_button = tk.Button(header_frame, text="About", command=show_about, bg="#57863C", relief="flat", fg="white")
    about_button.pack(side=tk.LEFT, padx=20)

title_label = tk.Label(
    header_frame, text="Acoustic Volumeter v1.0", font=("Helvetica", 24, "bold"), bg="#57863C", fg="white"
)
title_label.pack(side=tk.LEFT, padx=10)

left_frame = tk.Frame(content_frame, bg="#f2f2f2", width=200)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

toolkit_label = tk.Label(left_frame, text="Toolkit", font=("Arial", 16, "bold"), bg="#f2f2f2", fg="#57863C")
toolkit_label.pack(pady=10)

buttons = {}

buttons["measurement"] = tk.Button(
    left_frame, text="Measurement", width=20, height=2, command=lambda: open_module("measurement.py")
)
buttons["measurement"].pack(pady=5)

calibration_frame = tk.LabelFrame(left_frame, text="Calibration", bg="#f2f2f2", fg="dim gray", padx=10, pady=10)
calibration_frame.pack(fill=tk.X, pady=10)

calibration_buttons = [
    ("Acoustic Profiling", "profiling.py"),
    ("Resonance Modeling", "r_modeling.py"),
    ("Overall Trend Modeling", "overall_trend.py"),
]

for text, module in calibration_buttons:
    buttons[text.lower()] = tk.Button(
        calibration_frame, text=text, width=20, height=2, command=lambda m=module: open_module(m)
    )
    buttons[text.lower()].pack(pady=5)

utilities_frame = tk.LabelFrame(left_frame, text="Utilities", bg="#f2f2f2", fg="dim gray", padx=10, pady=10)
utilities_frame.pack(fill=tk.X, pady=10)

utility_buttons = [
    ("Single Tone Sampler", "single_tone.py"),
    ("Wave Viewer", "wave_viewer.py"),
]

for text, module in utility_buttons:
    buttons[text.lower()] = tk.Button(
        utilities_frame, text=text, width=20, height=2, command=lambda m=module: open_module(m)
    )
    buttons[text.lower()].pack(pady=5)

try:
    main_image = Image.open(resource("main.png"))

    max_width, max_height = 400, 300  # Maximum dimensions
    image_ratio = main_image.width / main_image.height
    if main_image.width > max_width or main_image.height > max_height:
        if image_ratio > 1:
            new_width = max_width
            new_height = int(max_width / image_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * image_ratio)
        main_image = main_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    main_image_tk = ImageTk.PhotoImage(main_image)

    center_label = tk.Label(content_frame, bg="white")
    center_label.pack(expand=True)

    text_label = tk.Label(
        center_label,
        text="Welcome to Acoustic Volumeter\n\nUse the toolkit on the left to get started.\n",
        font=("Helvetica", 16),
        bg="white",
        fg="#57863C",
        justify="center",
    )
    text_label.pack()

    image_label = tk.Label(center_label, image=main_image_tk, bg="white")
    image_label.image = main_image_tk  # Keep a reference to avoid garbage collection
    image_label.pack()
except FileNotFoundError:
    center_label = tk.Label(
        content_frame,
        text="Welcome to Acoustic Volumeter\n\nUse the toolkit on the left to get started.",
        font=("Helvetica", 16),
        bg="white",
        fg="#57863C",
        justify="center",
    )
    center_label.pack(expand=True)


footer_label = tk.Label(
    footer_frame, text="© 2025 Abbas Haghshenas", font=("Arial", 10), bg="#57863C", fg="white"
)
footer_label.pack(side=tk.RIGHT, padx=10)

root.mainloop()