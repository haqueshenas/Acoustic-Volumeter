#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Acoustic Volumeter — Acoustic Profiling Module
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
acoustic_profiling.py

Interactive GUI to generate and record an RMS vs frequency sweep:
  - Select & test input/output devices
  - Configure sweep parameters
  - Run, plot, and trim the sweep
  - Save CSV, PNG, WAV, and log

Usage:
    python acoustic_profiling.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import TclError
import sounddevice as sd
import numpy as np
import pandas as pd
import threading
import shutil
import time
import queue
import json
import os
from scipy.signal import find_peaks
from scipy.io import wavfile
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys, os, tempfile

def resource(rel_path):
    """
    Get absolute path to resource (works before and after bundling).
    """
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)


CONFIG_FILE = resource("device_config.json")

root = tk.Tk()
root.withdraw()
root.title("Acoustic Profiling - Acoustic Volumeter version 1.0")
root.geometry("740x760")
root.iconbitmap(resource("AV.ico"))
root.resizable(width=False, height=False)
update_gui_after_id = None

root.lift()
root.attributes('-topmost', True)
root.after(1000, lambda: root.attributes('-topmost', False))

root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f"{width}x{height}+{x}+{y}")

root.deiconify()

frequency_range = tk.StringVar(value="100-3000")
level = tk.DoubleVar(value=0.005)
step_size = tk.DoubleVar(value=100)
duration = tk.DoubleVar(value=3)
silence_duration = tk.DoubleVar(value=0)
is_running = False
selected_output_device = tk.StringVar()
selected_input_device = tk.StringVar()
unique_devices = {}
canceled_text = None
please_wait_text = None

dim_gray = 'dimgray'

fig, ax = plt.subplots(figsize=(7, 5))
line, = ax.plot([], [], 'b-', marker='o')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("RMS Amplitude (dB)")
ax.set_title("RMS vs Frequency")
plt.tight_layout()

ax.spines['top'].set_color(dim_gray)
ax.spines['bottom'].set_color(dim_gray)
ax.spines['left'].set_color(dim_gray)
ax.spines['right'].set_color(dim_gray)

ax.xaxis.label.set_color(dim_gray)
ax.yaxis.label.set_color(dim_gray)

ax.tick_params(axis='x', colors=dim_gray)
ax.tick_params(axis='y', colors=dim_gray)

ax.title.set_color(dim_gray)

status_var = tk.StringVar()
status_label = tk.Label(root, textvariable=status_var, fg="blue")
status_label.grid(row=10, column=0, columnspan=2)

volume_queue = queue.Queue(maxsize=1)
current_volume_level2 = tk.DoubleVar(value=0.05)

def setup_gui():
    """
    Build and arrange all widgets in the main window:
      - Preparation panel (device selection & test)
      - Waveform preview
      - Playback settings
      - Control buttons (Run, Reset, Cancel)
      - Main RMS vs Frequency plot area
    """
    root.configure(padx=20, pady=0)

    def check_config_file():
        if not os.path.exists(CONFIG_FILE):
            messagebox.showwarning(
                "Setup Required",
                "No device configuration found. Please select input and output devices in the Preparation section."
            )

    preparation_frame = tk.LabelFrame(root, text="Preparation", padx=10, pady=10)
    preparation_frame.grid(row=0, column=0, padx=(0, 10), pady=(5, 5), sticky="nw")

    tk.Label(preparation_frame, text="Select Output Device:").grid(row=0, column=0, padx=5, pady=5)
    output_menu = ttk.Combobox(preparation_frame, textvariable=selected_output_device, width=30)
    output_menu.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(preparation_frame, text="Select Input Device:").grid(row=1, column=0, padx=5, pady=5)
    input_menu = ttk.Combobox(preparation_frame, textvariable=selected_input_device, width=30)
    input_menu.grid(row=1, column=1, padx=5, pady=5)

    global start_experiment_button, cancel_experiment_button, test_input_button, test_output_button, volume_slider, waveform_fig, \
        waveform_canvas, waveform_ax, trim_lag_enabled, lag_duration, reset_button

    test_output_button = tk.Button(preparation_frame, text="Test Selected Output", width=17, command=test_output_device)
    test_output_button.grid(row=2, column=0, padx=0, pady=10)

    tk.Label(preparation_frame, text="Volume:").grid(row=2, column=1, padx=30, pady=5, sticky="w")
    volume_slider = tk.Scale(preparation_frame, from_=0, to=1, resolution=0.01, orient="horizontal", length=120,
                             variable=current_volume_level2)
    volume_slider.set(current_volume_level2.get())
    volume_slider.grid(row=2, column=1, padx=5, pady=5, sticky="e")

    sound_level_indicator = ttk.Progressbar(preparation_frame, orient="horizontal", length=175, mode="determinate")
    sound_level_indicator.grid(row=3, column=1, padx=5, pady=5, sticky="e")

    test_input_button = tk.Button(preparation_frame, text=" Test Selected Input ", width=17,
                                  command=lambda: test_input_device(sound_level_indicator))
    test_input_button.grid(row=3, column=0, padx=0, pady=10)

    waveform_frame = tk.LabelFrame(root, text="Waveform Preview", padx=5, pady=5)
    waveform_frame.grid(row=0, column=0, padx=(25, 10), pady=(210, 10), sticky="nw")

    waveform_fig = Figure(figsize=(3, 1), dpi=100)
    waveform_ax = waveform_fig.add_subplot(111)
    waveform_ax.set_xlabel("")
    waveform_ax.set_ylabel("")

    waveform_ax.spines['top'].set_color('white')
    waveform_ax.spines['bottom'].set_color('white')
    waveform_ax.spines['left'].set_color('white')
    waveform_ax.spines['right'].set_color('white')

    waveform_ax.xaxis.label.set_color('white')
    waveform_ax.yaxis.label.set_color('white')
    waveform_ax.tick_params(axis='x', colors='white')
    waveform_ax.tick_params(axis='y', colors='white')

    waveform_canvas = FigureCanvasTkAgg(waveform_fig, master=waveform_frame)
    waveform_canvas.get_tk_widget().pack()

    experiment_frame = tk.LabelFrame(root, text="Playback Settings", padx=10, pady=10)
    experiment_frame.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="ne")

    tk.Label(experiment_frame, text="Frequency Range (Hz):").grid(row=0, column=0, padx=5, pady=5)
    tk.Entry(experiment_frame, textvariable=frequency_range).grid(row=0, column=1, padx=5, pady=5)

    tk.Label(experiment_frame, text="Playback Level (0-1):").grid(row=1, column=0, padx=5, pady=5)
    level_entry = tk.Entry(experiment_frame, textvariable=level)
    level_entry.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(experiment_frame, text="Freq. Resolution (Hz):").grid(row=2, column=0, padx=5, pady=5)
    tk.Entry(experiment_frame, textvariable=step_size).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(experiment_frame, text="Playback Duration (s):").grid(row=3, column=0, padx=5, pady=5)
    tk.Entry(experiment_frame, textvariable=duration).grid(row=3, column=1, padx=5, pady=5)

    tk.Label(experiment_frame, text="Silence Duration (s):").grid(row=4, column=0, padx=5, pady=5)
    tk.Entry(experiment_frame, textvariable=silence_duration).grid(row=4, column=1, padx=5, pady=5)

    trim_lag_enabled = tk.BooleanVar(value=True)
    tk.Checkbutton(experiment_frame, text="Exclude Initial Lag (ms):", variable=trim_lag_enabled).grid(row=5, column=0, padx=0, pady=5)

    lag_duration = tk.DoubleVar(value=50)
    lag_entry = tk.Entry(experiment_frame, textvariable=lag_duration)
    lag_entry.grid(row=5, column=1, padx=5, pady=5)
    lag_entry.config(state="normal")

    def toggle_lag_entry(*args):
        lag_entry.config(state="normal" if trim_lag_enabled.get() else "disabled")
    trim_lag_enabled.trace_add("write", toggle_lag_entry)

    reset_button = tk.Button(experiment_frame, text="Reset", command=reset_experiment_parameters, width=15)
    reset_button.grid(row=7, column=0, columnspan=2, padx=5, pady=8)

    start_experiment_button = tk.Button(experiment_frame, text="Run", command=start_experiment, width=15)
    start_experiment_button.grid(row=8, column=0, columnspan=2, padx=5, pady=15, sticky="w")

    cancel_experiment_button = tk.Button(experiment_frame, text=" Cancel ", command=cancel_experiment, state=tk.DISABLED,
                                         width=15)
    cancel_experiment_button.grid(row=8, column=1, columnspan=2, padx=5, pady=8)

    def bind_keys():
        root.bind_all("<Control-Return>", lambda event: start_experiment())
        root.bind_all("<Control-r>", lambda event: reset_experiment_parameters())
        root.bind_all("<Escape>", lambda event: cancel_experiment())

    bind_keys()

    canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=5, pady=(5, 10), sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)

    populate_device_lists(input_menu, output_menu)
    level.trace("w", lambda *args: limit_playback_level(level, level_entry))

    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (root.winfo_width() // 2)
    y = (screen_height // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    load_device_config()

def populate_device_lists(input_menu, output_menu):
    """
    Populate the input/output dropdowns with available sounddevice devices.
    """
    for device in sd.query_devices():
        if device['max_input_channels'] > 0 and device['name'] not in unique_devices:
            unique_devices[device['name']] = device['index']
        if device['max_output_channels'] > 0 and device['name'] not in unique_devices:
            unique_devices[device['name']] = device['index']

    input_menu['values'] = [name for name in unique_devices.keys() if
                            sd.query_devices(unique_devices[name])['max_input_channels'] > 0]
    output_menu['values'] = [name for name in unique_devices.keys() if
                             sd.query_devices(unique_devices[name])['max_output_channels'] > 0]


def limit_playback_level(level_var, level_entry):
    """
    Ensure playback level stays within [0,1], ringing the bell on invalid entry.
    """
    value = level_var.get()
    if value < 0 or value > 1:
        level_var.set(max(0, min(1, value)))
        level_entry.bell()


def test_output_device():
    """
    Play a brief 1 kHz tone on the selected output device to verify routing.
    """
    if not validate_output_device():
        return
    try:
        sample_rate = 44100
        duration = 1.0
        frequency = 1000
        test_level = current_volume_level2.get()
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = (test_level * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        sd.play(tone, samplerate=sample_rate, device=unique_devices[selected_output_device.get()])
        sd.wait()
        messagebox.showinfo("Sound Test", "Testing sound played.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to test output device: {e}")


def test_input_device(sound_level_indicator):
    """
    Record audio for 5 s from the selected microphone, updating a progress bar
    with the live RMS level.
    """
    global update_gui_after_id

    try:
        start_experiment_button.config(state=tk.DISABLED)
        test_output_button.config(state=tk.DISABLED)
        reset_button.config(state=tk.DISABLED)
        volume_slider.config(state=tk.DISABLED)

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            volume_level = np.max(np.abs(indata))
            if not volume_queue.full():
                volume_queue.put(volume_level)
            else:
                volume_queue.queue.clear()
                volume_queue.put(volume_level)

        def update_gui():
            try:
                volume_level = volume_queue.get_nowait() * 100
                sound_level_indicator[
                    'value'] = volume_level if volume_level > 0 else 0
            except queue.Empty:
                sound_level_indicator['value'] = 0

            if stream.active:
                global update_gui_after_id
                update_gui_after_id = root.after(25, update_gui)

        messagebox.showinfo("Input Test", "Clap near the microphone to test for 5 seconds.")

        input_device = unique_devices[selected_input_device.get()]
        stream = sd.InputStream(device=input_device, channels=1, callback=audio_callback,
                                samplerate=48000, blocksize=1024)
        stream.start()

        update_gui()
        root.after(5000, lambda: stop_test(stream, sound_level_indicator))

    except Exception as e:
        messagebox.showerror("Error", f"Failed to test input device: {e}")
        if update_gui_after_id:
            root.after_cancel(update_gui_after_id)
            update_gui_after_id = None
        sound_level_indicator['value'] = 0


def stop_test(stream, sound_level_indicator):
    """
    Stop and close the input stream, reset the progress bar, and re-enable buttons.
    """
    global update_gui_after_id

    try:
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"Error stopping the stream: {e}")

    sound_level_indicator['value'] = 0

    if update_gui_after_id is not None:
        root.after_cancel(update_gui_after_id)
        update_gui_after_id = None

    start_experiment_button.config(state=tk.NORMAL)
    test_output_button.config(state=tk.NORMAL)
    reset_button.config(state=tk.NORMAL)
    volume_slider.config(state=tk.NORMAL)


def save_device_config():
    """
    Write current device and sweep settings to device_config.json.
    """
    config = {
        "input_device": selected_input_device.get(),
        "output_device": selected_output_device.get(),
        "frequency_range": frequency_range.get(),
        "playback_level": level.get(),
        "frequency_resolution": step_size.get(),
        "Playback_duration": duration.get(),
        "silence_duration": silence_duration.get(),
        "trim_lag_enabled": trim_lag_enabled.get(),
        "lag_duration": lag_duration.get(),
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def load_device_config():
    """
    Read device_config.json if it exists and restore GUI variables accordingly.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

        if config.get("input_device") in unique_devices:
            selected_input_device.set(config["input_device"])
        if config.get("output_device") in unique_devices:
            selected_output_device.set(config["output_device"])

        frequency_range.set(config.get("frequency_range", frequency_range.get()))
        level.set(config.get("playback_level", level.get()))
        step_size.set(config.get("frequency_resolution", step_size.get()))
        duration.set(config.get("Playback_duration", duration.get()))
        silence_duration.set(config.get("silence_duration", silence_duration.get()))
        trim_lag_enabled.set(config.get("trim_lag_enabled", trim_lag_enabled.get()))
        lag_duration.set(config.get("lag_duration", lag_duration.get()))

def reset_experiment_parameters():
    """
    Restore all sweep parameter fields to their default values.
    """
    frequency_range.set("100-3000")
    level.set("0.005")
    step_size.set("100")
    duration.set("2.5")
    silence_duration.set("0.5")
    trim_lag_enabled.set(True)
    lag_duration.set(50)

def estimate_resonance(results):
    """
    From a list of [frequency, RMS] pairs, detect the highest RMS peak
    and return its frequency, or None if no peak found.
    """
    freqs = [row[0] for row in results]
    rms_dB_values = [row[1] for row in results]
    peaks, _ = find_peaks(rms_dB_values)
    if peaks.size > 0:
        resonance_freq = freqs[peaks[np.argmax([rms_dB_values[p] for p in peaks])]]
        return resonance_freq
    else:
        return None

def run_experiment():
    """
    Execute the full sweep in a background thread:
      - Play-and-record tones with optional lag exclusion
      - Compute live RMS, update waveform & RMS plot
      - Append silence segments, track remaining time
      - On completion, call save_data()
    """
    global is_running, combined_recording, temp_wav_path, canceled_text
    is_running = True
    combined_recording = []

    if canceled_text:
        canceled_text.remove()
        canceled_text = None

    start_experiment_button.config(state=tk.DISABLED)
    reset_button.config(state=tk.DISABLED)
    cancel_experiment_button.config(state=tk.NORMAL)

    results = []
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    temp_wav_path = os.path.join(tempfile.gettempdir(), "AcousticVolumeter_temp.wav")

    try:
        sample_rate = 44100
        input_device = unique_devices[selected_input_device.get()]
        output_device = unique_devices[selected_output_device.get()]
        sd.default.device = (input_device, output_device)
        sd.default.samplerate = sample_rate

        freqs = np.linspace(
            int(frequency_range.get().split("-")[0]),
            int(frequency_range.get().split("-")[1]),
            int((int(frequency_range.get().split("-")[1]) - int(frequency_range.get().split("-")[0])) / step_size.get()) + 1
        )
        duration_val = duration.get()
        silence_val = silence_duration.get()
        trim_samples = int((lag_duration.get() / 1000) * sample_rate) if trim_lag_enabled.get() else 0
        total_time = len(freqs) * (duration_val + silence_val)

        please_wait_text = ax.text(
            0.5, 0.5, "Please wait...",
            fontsize=30, color='grey', weight='bold', ha='center', va='center',
            transform=ax.transAxes
        )
        canvas.draw()

        def remove_message_after_delay():
            time.sleep(duration_val)
            if please_wait_text:
                please_wait_text.remove()
                canvas.draw()

        threading.Thread(target=remove_message_after_delay, daemon=True).start()

        for i, freq in enumerate(freqs):
            if not is_running:
                break

            remaining_time = total_time - (i * (duration_val + silence_val))
            minutes, seconds = divmod(remaining_time, 60)
            root.after(0, lambda freq=freq, i=i, minutes=minutes, seconds=seconds:
            status_var.set(
                f"Playing frequency: {freq} Hz ({i + 1}/{len(freqs)}) | Remaining time: {int(minutes):02d}:{int(seconds):02d}"))
            root.update_idletasks()

            t = np.linspace(0, duration_val, int(sample_rate * duration_val), False)
            tone = (level.get() * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            recording = sd.playrec(tone, samplerate=sample_rate, channels=1)
            sd.wait()

            trimmed_recording = recording[trim_samples:] if trim_samples > 0 else recording
            combined_recording.append(trimmed_recording)

            wavfile.write(temp_wav_path, sample_rate, np.concatenate(combined_recording))

            absolute_rms = np.sqrt(np.mean(trimmed_recording ** 2))
            absolute_rms_dB = 20 * np.log10(absolute_rms) if absolute_rms > 0 else -np.inf
            results.append([freq, absolute_rms_dB])

            center_index = len(trimmed_recording) // 2
            wave_cycles = int(sample_rate / freq * 3)
            start_index = max(center_index - wave_cycles // 2, 0)
            end_index = min(center_index + wave_cycles // 2, len(trimmed_recording))

            waveform_ax.clear()
            waveform_ax.plot(trimmed_recording[start_index:end_index], color="red")
            root.after(0, waveform_canvas.draw)

            line.set_xdata(freqs[:len(results)])
            line.set_ydata([row[1] for row in results])
            ax.relim()
            ax.autoscale_view()
            root.after(0, canvas.draw)

            if i < len(freqs) - 1:
                silence_segment = np.zeros((int(sample_rate * silence_val), 1), dtype=np.float32)
                combined_recording.append(silence_segment)
                sd.sleep(int(silence_val * 1000))

        root.after(0, lambda: status_var.set("Experiment complete."))
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        resonance_freq = estimate_resonance(results)

        if is_running:
            save_data(results, start_time, end_time, resonance_freq, combined_recording, trim_lag_enabled.get(), lag_duration.get())

            if resonance_freq is not None:
                root.after(0, lambda: messagebox.showinfo("Recorded Peak",
                                                          f"Recorded Peak Frequency: {resonance_freq} Hz"))

        root.after(0, lambda: status_var.set(""))

    except Exception as e:
        if is_running:
            root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {e}"))

    finally:
        is_running = False
        try:
            if root.winfo_exists():
                root.after(0, lambda: start_experiment_button.config(state=tk.NORMAL))
                root.after(0, lambda: test_input_button.config(state=tk.NORMAL))
                root.after(0, lambda: test_output_button.config(state=tk.NORMAL))
                root.after(0, lambda: reset_button.config(state=tk.NORMAL))
                root.after(0, lambda: volume_slider.config(state=tk.NORMAL))
                root.after(0, lambda: cancel_experiment_button.config(state=tk.DISABLED))
        except TclError:
            pass

def open_wav_file_stream(filename, sample_rate):
    """Open a .wav file for streaming audio data."""
    wavfile.write(filename, sample_rate, np.array([], dtype=np.float32))
    return filename


def append_audio_segment(filename, segment):
    """Append audio data to an open .wav file in streaming mode."""
    wavfile.write(filename, 44100, segment)


def cancel_experiment():
    """
    Cancel any ongoing sweep, mark the plot “Canceled!”, and reset UI state.
    """
    global is_running, please_wait_text, canceled_text
    is_running = False

    if please_wait_text:
        please_wait_text.remove()
        please_wait_text = None

    canceled_text = ax.text(
        0.5, 0.5, "Canceled!",
        fontsize=30, color='grey', weight='bold', ha='center', va='center',
        transform=ax.transAxes
    )
    canvas.draw()

    status_var.set("Experiment canceled.")
    reset_button.config(state=tk.NORMAL)
    start_experiment_button.config(state=tk.NORMAL)
    cancel_experiment_button.config(state=tk.DISABLED)

def validate_output_device():
    if not selected_output_device.get():
        messagebox.showerror(
            "Output Device Required",
            "Please select an output device in the Preparation section."
        )
        return False
    return True

def validate_input_device():
    if not selected_input_device.get():
        messagebox.showerror(
            "Input Device Required",
            "Please select an input device in the Preparation section."
        )
        return False
    return True

def validate_inputs():
    try:
        freq_range = frequency_range.get()
        freq_min, freq_max = map(int, freq_range.split('-'))
        if freq_min <= 0 or freq_max <= 0 or freq_min >= freq_max:
            raise ValueError("Frequency range must have positive values separated by '-'. Minimum should be less than maximum.")
    except ValueError as e:
        messagebox.showerror("Invalid Frequency Range", str(e))
        return False

    if not (0 <= level.get() <= 1):
        messagebox.showerror("Invalid Playback Level", "Playback level must be between 0 and 1.")
        return False

    if not step_size.get() > 0 or not float(step_size.get()).is_integer():
        messagebox.showerror("Invalid Frequency Resolution", "Frequency resolution must be a positive integer.")
        return False

    if duration.get() <= 0:
        messagebox.showerror("Invalid Playback Duration", "Playback duration must be a positive number.")
        return False

    if silence_duration.get() < 0:
        messagebox.showerror("Invalid Silence Duration", "Silence duration must be a positive number.")
        return False

    if lag_duration.get() < 0:
        messagebox.showerror("Invalid Lag Duration", "Lag duration must be a positive number or zero.")
        return False

    return True

def start_experiment():
    """
    Validate all settings, disable controls, save config, and spawn
    the sweep thread via run_experiment().
    """
    if update_gui_after_id:
        messagebox.showerror(
            "Operation in Progress",
            "Please wait until the microphone test is complete before starting the experiment."
        )
        return

    if not validate_output_device():
        return
    if not validate_input_device():
        return

    if validate_inputs():
        global is_running
        is_running = True
        save_device_config()

        test_input_button.config(state=tk.DISABLED)
        test_output_button.config(state=tk.DISABLED)
        reset_button.config(state=tk.DISABLED)
        volume_slider.config(state=tk.DISABLED)

        experiment_thread = threading.Thread(target=run_experiment)
        experiment_thread.start()
    else:
        status_var.set("Please correct input errors.")

def on_close():
    global is_running, update_gui_after_id
    if is_running:
        if messagebox.askokcancel("Exit", "The experiment is still running. Do you want to terminate it?"):
            is_running = False
            if update_gui_after_id:
                root.after_cancel(update_gui_after_id)
            root.destroy()
            os._exit(0)
    else:
        if update_gui_after_id:
            root.after_cancel(update_gui_after_id)
        root.destroy()
        os._exit(0)

def check_microphone():
    if not sd.query_devices(kind='input'):
        messagebox.showerror("Error", "No microphone detected.")
        return False
    return True


def save_data(results, start_time, end_time, peak_freq, recording_combined, trim_lag_enabled, lag_duration):
    """
    Prompt for a CSV file path, save:
      - RMS data (CSV)
      - Plot (PNG)
      - WAV (moving temp file)
      - Log file with sweep parameters & peak
    """
    filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

    if filename:
        df = pd.DataFrame(results, columns=["Frequency (Hz)", "RMS (dB)"])
        df.to_csv(filename, index=False)

        fig.savefig(f"{filename.rsplit('.', 1)[0]}.png", dpi=300, format='png')

        log_experiment(results, start_time, end_time, peak_freq, filename.rsplit('.', 1)[0], trim_lag_enabled, lag_duration)

        final_wav_path = f"{filename.rsplit('.', 1)[0]}.wav"
        shutil.move(temp_wav_path, final_wav_path)

        messagebox.showinfo("Save Complete", "Experiment data, plot, log, and sound saved.")

def estimate_peak(results):
    """
    Alias of estimate_resonance; detect and return the frequency of the
    maximum RMS peak from the results list.
    """
    freqs = [row[0] for row in results]
    rms_dB_values = [row[1] for row in results]
    peaks, _ = find_peaks(rms_dB_values)
    if peaks.size > 0:
        peak_freq = freqs[peaks[np.argmax([rms_dB_values[p] for p in peaks])]]
        return peak_freq
    else:
        return None

def log_experiment(results, start_time, end_time, peak_freq, base_filename, trim_lag_enabled, lag_duration):
    log_filename = f"{base_filename}_log.txt"

    log_content = (
        f"Output Device: {selected_output_device.get()}\n"
        f"Input Device: {selected_input_device.get()}\n"
        f"Experiment Start Time: {start_time}\n"
        f"Experiment End Time: {end_time}\n"
        f"Frequency Range (Hz): {frequency_range.get()}\n"
        f"Playback Level: {level.get()}\n"
        f"Frequency Resolution (Hz): {step_size.get()}\n"
        f"Playback Duration (s): {duration.get()}\n"
        f"Silence Duration (s): {silence_duration.get()}\n"
        f"Lag Exclusion Enabled: {'Yes' if trim_lag_enabled else 'No'}\n"
        f"Lag Duration (ms): {lag_duration if trim_lag_enabled else 'N/A'}\n"
        f"Peak Frequency (Hz): {peak_freq if peak_freq is not None else 'Not detected'}\n"
        "-----------------------------------------\n"
    )

    # Write to the log file
    with open(log_filename, "a") as log_file:
        log_file.write(log_content)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=9, column=0, columnspan=2, sticky="nsew")

setup_gui()
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()