#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Wave Viewer Module
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
wave_viewer.py

Drag-&-drop or load WAV files, pan/zoom their waveforms on synchronized
timelines, toggle linear/dB scale and export high-res PNGs.

Usage:
    python wave_viewer.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, Scale
from tkinterdnd2 import TkinterDnD, DND_FILES
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from scipy.io import wavfile
import soundfile as sf
import warnings
from scipy.io.wavfile import WavFileWarning

import sys, os

def resource(rel_path):
    """
    Return the absolute path to a resource (image, icon, etc.),
    whether running from source or from a PyInstaller one-file bundle.
    """
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

class WaveViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Wave Viewer - Acoustic Volumeter v.1.0")
        self.root.iconbitmap(resource("AV.ico"))
        self.root.geometry("1000x600")
        self.root.drop_target_register(DND_FILES)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.dnd_bind("<<Drop>>", self.drag_and_drop)

        self.files = []
        self.fig, self.axs = plt.subplots(1, 1)
        self.zoom_level = 1.0
        self.zoom_step = 1.1
        self.min_zoom = 1
        self.max_zoom = None

        self.is_dragging = False
        self.last_press_x = None

        self.setup_frames()
        self.setup_controls()
        self.update_plot_layout()

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_frames(self):
        """
        Build and pack the left control panel and right plotting area (with
        mouse & scroll zoom bindings).
        """
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.timeline_frame = tk.Frame(self.plot_frame)
        self.timeline_frame.pack(fill="x", pady=(0, 10))

        self.canvas.mpl_connect("scroll_event", self.on_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def setup_controls(self):
        """
        Create buttons, sliders, radio-buttons and checkboxes but keep them
        disabled until files are loaded.
        """
        self.load_button = tk.Button(self.control_frame, text="Load WAV Files", height=2, width=14,
                                     command=self.load_files)
        self.load_button.pack(pady=10)

        self.export_button = tk.Button(self.control_frame, text="Export PNG", height=2, width=14, state="disabled",
                                       command=self.save_plot_with_confirmation)
        self.export_button.pack(pady=10)

        self.scale_radio_buttons = []
        tk.Label(self.control_frame, text="Amplitude Scale:", state="normal").pack(pady=10)
        self.scale_var = tk.StringVar(value="Linear")
        for scale in ["Linear", "dB"]:
            rb = tk.Radiobutton(
                self.control_frame,
                text=scale,
                variable=self.scale_var,
                value=scale,
                state="disabled",
                command=self.update_plot
            )
            rb.pack(anchor="w", padx=25)
            self.scale_radio_buttons.append(rb)

        zoom_controls_frame = tk.Frame(self.control_frame)
        zoom_buttons_frame = tk.Frame(self.control_frame)

        zoom_label = tk.Label(zoom_controls_frame, text="Zoom Level:", state="normal")
        zoom_label.pack(pady=15)

        self.zoom_slider = Scale(
            zoom_controls_frame,
            from_=self.min_zoom,
            to=self.max_zoom,
            orient="horizontal",
            command=self.on_slider_zoom,
            length=100,
            state="disabled",
            showvalue=0
        )
        self.zoom_slider.set(self.zoom_level)

        self.zoom_in_button = tk.Button(zoom_buttons_frame, text="+", width=2, height=1, state="disabled",
                                   command=lambda: self.adjust_zoom_slider(1))
        self.zoom_out_button = tk.Button(zoom_buttons_frame, text="-", width=2, height=1, state="disabled",
                                    command=lambda: self.adjust_zoom_slider(-1))

        zoom_controls_frame.pack(pady=10)
        zoom_buttons_frame.pack(pady=2)

        self.zoom_slider.pack(side="top", fill="x", expand=True, padx=0)
        self.zoom_in_button.pack(side="left", padx=5)
        self.zoom_out_button.pack(side="left", padx=5)

        self.root.bind("<Up>", lambda e: self.adjust_zoom_slider(1))
        self.root.bind("<Down>", lambda e: self.adjust_zoom_slider(-1))

        self.timeline_slider = Scale(
            self.timeline_frame,
            from_=0,
            to=1000,
            orient="horizontal",
            state="disabled",
            command=self.on_timeline_slide,
            showvalue=0

        )

        self.timeline_slider.pack(fill="x")

        self.root.bind("<Left>", lambda event: self.adjust_timeline_slider(-1))
        self.root.bind("<Right>", lambda event: self.adjust_timeline_slider(1))

        self.show_labels = tk.BooleanVar(value=True)
        self.show_values = tk.BooleanVar(value=True)

        self.reset_button = tk.Button(self.control_frame, text="Reset View", height=2, width=14, state="disabled",
                                      command=self.reset_plot)
        self.reset_button.pack(anchor="s", pady=20)

        self.clear_button = tk.Button(self.control_frame, text="Clear Files", height=2, width=14, state="disabled",
                                      command=self.clear_files)
        self.clear_button.pack(pady=5)

        self.show_labels = tk.BooleanVar(value=True)
        self.show_labels_check = tk.Checkbutton(
            self.control_frame,
            text="Show File Labels",
            variable=self.show_labels,
            state="disabled",
            command=self.update_plot
        )
        self.show_labels_check.pack(pady=15, anchor='w')

        self.show_values_check = tk.Checkbutton(
            self.control_frame,
            text="Show Values",
            variable=self.show_values,
            state="disabled",
            command=self.update_plot
        )
        self.show_values_check.pack(pady=0, anchor='w')

    def load_files(self):
        """
        Prompt for up to 8 WAVs, read via soundfile, normalize to float32,
        then enable controls and redraw.
        """
        MAX_FILES = 8
        file_paths = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])

        if len(file_paths) + len(self.files) > MAX_FILES:
            messagebox.showwarning("File Limit Exceeded", f"Only {MAX_FILES} files are allowed at a time.")
            return

        if not file_paths:
            return

        self.files = []
        for path in file_paths:
            try:
                data, sample_rate = sf.read(path)
                data = data.astype(np.float32)
                self.files.append((os.path.basename(path), sample_rate, data))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {path}: {e}")

        if self.files:
            self.enable_widgets()
            max_sample_count = max(len(data) for _, _, data in self.files)
            self.max_zoom = int(max_sample_count / 10)
            self.zoom_slider.config(to=self.max_zoom)

        self.update_plot_layout()
        self.update_plot()
        self.set_dynamic_max_zoom()

    def drag_and_drop(self, event):
        """
        Handle files dropped onto the window: load, normalize, then redraw.
        """
        MAX_FILES = 8
        file_paths = list(self.root.tk.splitlist(event.data))
        current_file_count = len(self.files)

        if current_file_count + len(file_paths) > MAX_FILES:
            messagebox.showwarning("File Limit Exceeded", f"Only {MAX_FILES} files are allowed at a time.")
            return

        for path in file_paths:
            if path.lower().endswith('.wav'):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", WavFileWarning)
                        sample_rate, data = wavfile.read(path)

                    if data.dtype.kind == 'i':
                        max_val = np.iinfo(data.dtype).max
                        data = data.astype(np.float32) / max_val
                    elif data.dtype.kind == 'f':
                        data = data.astype(np.float32)

                    self.files.append((os.path.basename(path), sample_rate, data))
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {path}: {e}")

        self.update_plot_layout()
        self.update_plot()
        self.set_dynamic_max_zoom()
        self.enable_widgets()

    def enable_widgets(self):
        """
        Activate all UI controls once at least one file is in memory.
        """
        self.export_button.config(state="normal")
        self.zoom_in_button.config(state="normal")
        self.zoom_out_button.config(state="normal")
        self.zoom_slider.config(state="normal")
        self.timeline_slider.config(state="normal")
        self.reset_button.config(state="normal")
        self.clear_button.config(state="normal")
        for button in self.scale_radio_buttons:
            button.config(state="normal")
        self.show_labels_check.config(state="normal")
        self.show_values_check.config(state="normal")

    def disable_widgets(self):
        """
        Deactivate all UI controls when no files are present.
        """
        self.export_button.config(state="disabled")
        self.zoom_in_button.config(state="disabled")
        self.zoom_out_button.config(state="disabled")
        self.zoom_slider.config(state="disabled")
        self.timeline_slider.config(state="disabled")
        self.reset_button.config(state="disabled")
        self.clear_button.config(state="disabled")
        for button in self.scale_radio_buttons:
            button.config(state="disabled")
        self.show_labels_check.config(state="disabled")
        self.show_values_check.config(state="disabled")

    def update_plot_layout(self):
        """
        Updates the plot layout based on the number of files loaded.
        Creates a subplot for each file or a single blank axis if no files are present.
        """
        num_files = len(self.files)

        self.fig.clf()

        if num_files == 0:
            self.axs = self.fig.subplots(1, 1)
            self.axs = [self.axs]
            for ax in self.axs:
                ax.axis('off')
        else:
            self.axs = self.fig.subplots(num_files, 1, sharex=True)
            if num_files == 1:
                self.axs = [self.axs]

        self.canvas.draw()

    def format_time(self, x, pos):
        """
        Convert x in seconds to MM:SS[.ms] string based on current zoom.
        """
        minutes = int(x // 60)
        seconds = int(x % 60)

        if self.zoom_level < 10:
            return f"{minutes:02}:{seconds:02}"
        elif self.zoom_level < 100:
            milliseconds = int((x * 1000) % 1000)
            return f"{minutes:02}:{seconds:02}.{milliseconds:03}"
        elif self.zoom_level < 1000:
            microseconds = int((x * 1e6) % 1e6)
            return f"{minutes:02}:{seconds:02}.{microseconds:06}"[:9]
        else:
            microseconds = int((x * 1e6) % 1e6)
            return f"{minutes:02}:{seconds:02}.{microseconds:06}"

    def update_plot(self):
        """
        Updates the plot with data from loaded files. Adjusts the layout, scales, and labels dynamically.
        """
        if not self.files:
            self.clear_files()
            return

        self.fig.clf()

        num_files = len(self.files)
        self.axs = self.fig.subplots(num_files, 1, sharex=True)
        if num_files == 1:
            self.axs = [self.axs]

        self.max_time = 0

        for i, (file_name, sample_rate, data) in enumerate(self.files):
            ax = self.axs[i]
            time = np.linspace(0, len(data) / sample_rate, num=len(data))
            self.max_time = max(self.max_time, time[-1])

            scale = self.scale_var.get()
            if scale == "dB":
                data = 20 * np.log10(np.abs(data) + 1e-6)
                ax.set_ylim(-120, 0)
            elif scale == "Linear":
                ax.set_ylim(-1, 1)

            # Plot data
            ax.plot(time, data, label=file_name, zorder=1)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)

            if self.show_labels.get():
                ax.legend([file_name], loc='upper right', framealpha=0.5)
            else:
                ax.legend().set_visible(False)

            if i == len(self.files) - 1:
                ax.set_xlabel("Time (MM:SS.sss)", color='#595959')
                ax.set_ylabel("Amplitude", loc='center', color='#595959')

            ax.xaxis.set_major_formatter(FuncFormatter(self.format_time))
            ax.set_xlim(0, self.max_time)  # Set X-axis range

        self.fig.tight_layout()

        self.apply_zoom(initial=True)
        self.update_timeline_slider()

        self.canvas.draw()

    def clear_files(self):
        """
        Unload all files, reset zoom/timeline sliders, and show blank canvas.
        """
        self.files.clear()

        self.fig.clf()

        self.axs = self.fig.subplots(1, 1)
        self.axs = [self.axs]

        for ax in self.axs:
            ax.axis('off')

        self.zoom_level = 1
        self.zoom_slider.set(self.zoom_level)
        self.timeline_slider.set(0)

        self.canvas.draw()
        self.disable_widgets()

    def save_plot_with_confirmation(self):
        """
        Ask for a PNG path, save current figure at 600 dpi, then notify.
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.fig.savefig(file_path, dpi=600)
            messagebox.showinfo("Save Successful", f"PNG file saved successfully at: {file_path}")

    def on_zoom(self, event):
        """
        Mouse-wheel zoom handler: adjust zoom_level logarithmically.
        """
        if not self.files:
            return

        scale_factor = self.zoom_step if event.button == 'up' else 1 / self.zoom_step
        new_zoom_level = self.zoom_level * scale_factor

        if self.min_zoom <= new_zoom_level <= self.max_zoom:
            self.zoom_level = new_zoom_level

            slider_value = 1000 * np.log(self.zoom_level / self.min_zoom) / np.log(self.max_zoom / self.min_zoom)
            self.zoom_slider.set(slider_value)

            self.apply_zoom()
            self.update_timeline_slider()

    def on_slider_zoom(self, value):
        """
        Map zoom slider position back to zoom_level and reapply zoom.
        """
        if not self.files:
            return

        slider_value = float(value)
        self.zoom_level = self.min_zoom * (self.max_zoom / self.min_zoom) ** (slider_value / 1000)

        self.apply_zoom()
        self.update_timeline_slider()

    def set_dynamic_max_zoom(self):
        """
        After loading, compute max_zoom from file durations and reset slider.
        """
        self.max_time = max(len(data) / sample_rate for _, sample_rate, data in self.files)
        target_time_precision = 1e-6
        self.max_zoom = int(self.max_time / target_time_precision)

        self.zoom_step = self.max_zoom ** (1 / 50)
        self.zoom_slider.config(from_=0, to=1000)
        self.zoom_level = self.min_zoom
        self.zoom_slider.set(0)

    def apply_zoom(self, initial=False):
        """
        Apply zoom_level to all axes by adjusting their x-limits.
        """
        if not self.files:
            return

        window_width = self.max_time / self.zoom_level

        if window_width >= self.max_time:
            window_width = self.max_time
            x_min = 0
            x_max = self.max_time
        else:
            current_xlim = self.axs[0].get_xlim()
            center = (current_xlim[0] + current_xlim[1]) / 2
            x_min = center - window_width / 2
            x_max = center + window_width / 2

            if x_min < 0:
                x_min = 0
                x_max = window_width
            elif x_max > self.max_time:
                x_max = self.max_time
                x_min = self.max_time - window_width

        for ax in self.axs:
            ax.set_xlim(x_min, x_max)

        if not initial:
            self.canvas.draw()

    def on_press(self, event):
        """
        Begin a pan (drag) operation by recording initial mouse-x position.
        """
        if event.button == 1:
            self.is_dragging = True
            self.last_press_x = event.xdata

    def on_release(self, event):
        """
        End any ongoing pan (drag) operation.
        """
        self.is_dragging = False

    def on_drag(self, event):
        """
        Continue panning: shift x-limits according to mouse movement.
        """
        if not self.is_dragging or not self.files:
            return
        if event.xdata is None or self.last_press_x is None:
            return

        dx = self.last_press_x - event.xdata
        current_xlim = self.axs[0].get_xlim()
        new_xlim = (current_xlim[0] + dx, current_xlim[1] + dx)

        if new_xlim[0] < 0:
            new_xlim = (0, current_xlim[1] - current_xlim[0])
        elif new_xlim[1] > self.max_time:
            new_xlim = (self.max_time - (current_xlim[1] - current_xlim[0]), self.max_time)

        for ax in self.axs:
            ax.set_xlim(new_xlim)
        self.canvas.draw()

    def on_timeline_slide(self, value):
        """
        Center the zoom window based on timeline slider’s fractional value.
        """
        if not self.files:
            return

        center_pos = float(value) / 1000 * self.max_time
        window_width = self.max_time / self.zoom_level
        x_min = max(0, center_pos - window_width / 2)
        x_max = min(self.max_time, center_pos + window_width / 2)

        for ax in self.axs:
            ax.set_xlim(x_min, x_max)

        self.canvas.draw()

    def adjust_zoom_slider(self, direction):
        """
        Incrementally change zoom_level via +/– buttons or Up/Down keys.
        """
        if not self.files:
            return

        new_zoom = self.zoom_level * (self.zoom_step ** direction)
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_level = new_zoom

            slider_value = 1000 * np.log(self.zoom_level / self.min_zoom) / np.log(self.max_zoom / self.min_zoom)
            self.zoom_slider.set(slider_value)

            self.apply_zoom()
            self.update_timeline_slider()

    def adjust_timeline_slider(self, step):
        """
        Nudge the timeline slider left or right by one unit.
        """
        current_value = self.timeline_slider.get()
        new_value = max(self.timeline_slider.cget('from'),
                        min(self.timeline_slider.cget('to'), current_value + step))
        self.timeline_slider.set(new_value)

    def update_timeline_slider(self):
        """
        Sync the timeline slider to the current x-axis center.
        """
        if not self.files:
            return
        current_xlim = self.axs[0].get_xlim()
        slider_position = int((sum(current_xlim) / 2 / self.max_time) * 1000)
        self.timeline_slider.set(slider_position)

    def on_mouse_move(self, event):
        """
        If “Show Values” is checked, highlight & annotate the nearest sample.
        """
        if not self.files:
            return

        if not self.show_values.get():
            return

        if event.inaxes not in self.axs:
            return

        mouse_x = event.xdata
        if mouse_x is None:
            return

        for i, ax in enumerate(self.axs):
            for label in ax.texts:
                label.remove()
            for line in ax.lines:
                if hasattr(line, '_is_marker') and line._is_marker:
                    line.remove()
                elif line.get_linestyle() == '--' and line.get_color() == 'red':
                    line.remove()

            _, sample_rate, data = self.files[i]
            time = np.linspace(0, len(data) / sample_rate, num=len(data))

            closest_idx = np.argmin(np.abs(time - mouse_x))
            closest_time = time[closest_idx]

            if self.scale_var.get() == "dB":
                original_value = np.abs(data[closest_idx])
                closest_value = 20 * np.log10(original_value + 1e-6)
            else:
                closest_value = data[closest_idx]

            marker_line, = ax.plot(
                closest_time, closest_value, 'ro', markersize=5, zorder=5
            )
            marker_line._is_marker = True

            ax.text(
                closest_time, closest_value,
                f"{closest_value:.3f}",
                color='red', fontsize=9, ha='left',
                va='bottom',
                transform=ax.transData,
                zorder=6
            )

            ax.axvline(
                x=closest_time,
                color='red',
                linestyle='--',
                linewidth=0.8,
                zorder=4
            )

        self.canvas.draw()

    def reset_plot(self):
        """
        Restore zoom_level=1, timeline=0 and full-view x-limits.
        """
        if not self.files:
            return
        self.zoom_level = 1
        self.zoom_slider.set(self.zoom_level)
        self.timeline_slider.set(0)
        for ax in self.axs:
            ax.set_xlim(0, self.max_time)
        self.canvas.draw()

    def save_plot(self):
        """
        (Unused) prompt for a PNG path and save at 600 dpi.
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.fig.savefig(file_path, dpi=600)

    def on_exit(self):
        """
        Confirm and then cleanly destroy the window & exit process.
        """
        if messagebox.askyesno("Exit", "Are you sure you want to quit?"):
            self.root.destroy()
            os._exit(0)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = WaveViewer(root)
    root.mainloop()