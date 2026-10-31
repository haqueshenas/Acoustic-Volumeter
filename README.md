 <div align="center">

<img src="AV.png" alt="Acoustic Volumeter logo" width="155">

# Acoustic Volumeter

**Microphone-free acoustic volumetry for rapid and precise volume measurement.**

Release v1.0.0 · Free and open-source research software

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15280747.svg)](https://doi.org/10.5281/zenodo.15280747) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Download Windows application (v1.0.0)](https://github.com/haqueshenas/Acoustic-Volumeter/releases/tag/v1.0.0) · [PDF user guide](https://github.com/haqueshenas/Acoustic-Volumeter/releases/download/v1.0.0/README.pdf) · [Published article](https://doi.org/10.1016/j.atech.2025.101430) · [Source code](https://github.com/haqueshenas/Acoustic-Volumeter/tree/v1.0.0) · [Easy Phenotyping Lab (EPL)](https://haqueshenas.github.io/EPL/)

</div>

---

## Overview

Acoustic Volumeter is the software component of an open-source, do-it-yourself (DIY) platform for microphone-free acoustic volumetry, developed for rapid and precise volume measurement in crop phenotyping and seed science.

The method estimates sample volume by tracking shifts in the resonance peak of a sealed acoustic chamber. Instead of requiring a conventional measurement microphone, the sensor uses a dynamic microphone cartridge mounted on the chamber as an acoustic–electrical transducer. A computer sound card provides the excitation and signal-acquisition interface.

The software integrates acoustic profiling, resonance analysis, calibration, volume measurement, and signal inspection in a modular graphical interface.

In the published ten-grain wheat assay, the method achieved an average error of less than 0.2 µL per grain under the reported experimental conditions. This result should not be interpreted as a universal accuracy guarantee; performance depends on the sensor, calibration, measurement settings, and experimental conditions.

For the scientific background, hardware design, experimental validation, and detailed software instructions, see the [peer-reviewed article](https://doi.org/10.1016/j.atech.2025.101430) and the accompanying [PDF user guide](https://github.com/haqueshenas/Acoustic-Volumeter/releases/download/v1.0.0/README.pdf).

## How the platform works

The recommended workflow consists of the following steps:

1. **Prepare the sensor:** Assemble the acoustic chamber and connect it to a compatible computer sound card, following the published hardware instructions.
2. **Profile the sensor:** Use Acoustic Profiling to record its RMS response across a frequency sweep.
3. **Model the resonance:** Use Resonance Modeling to fit the recorded profile and estimate its resonance peak frequency.
4. **Build the calibration:** Use Overall Trend Modeling to relate resonance frequency to known sample volumes.
5. **Measure unknown samples:** Use Measurement with the resulting calibration models to estimate sample volumes.

The Single Tone Sampler and Wave Viewer provide additional utilities for audio-device testing and signal inspection.

## Quick start — Windows application

The easiest way to use Acoustic Volumeter is the pre-built Windows x64 application.

1. Download `AcousticVolumeter-v1.0.0-Windows-x64.zip` from the [GitHub Releases page](https://github.com/haqueshenas/Acoustic-Volumeter/releases/tag/v1.0.0).
2. Extract the complete ZIP archive to a folder on your computer.
3. Run `AcousticVolumeter.exe` from the extracted folder. A separate Python installation is not required.
4. Connect and configure the sensor and audio devices according to the published hardware instructions.
5. Start with Acoustic Profiling to examine the sensor response and prepare the calibration data.
6. Complete the calibration workflow before measuring unknown samples.

For detailed instructions, parameter descriptions, examples, and practical considerations, consult the [PDF user guide](https://github.com/haqueshenas/Acoustic-Volumeter/releases/download/v1.0.0/README.pdf).

## Tools at a glance

Acoustic Volumeter provides six complementary tools organized into three functional groups.

### Measurement

**Measurement**

The main volumetry module. It acquires the acoustic response, estimates the resonance peak using a selected EST (peak-estimation) model, and uses an overall frequency-to-volume calibration model to estimate unknown sample volumes. Measurement results and associated data can be saved for subsequent analysis.

### Calibration

**Acoustic Profiling**

Generates a frequency sweep and records the sensor's RMS response. It supports audio-device selection and testing, real-time signal visualization, and export of profiling data for subsequent modeling.

**Resonance Modeling**

Fits analytical models to RMS-versus-frequency profiles to estimate resonance peak parameters, particularly peak frequency. Supported models include Quadratic, Gaussian, Lorentzian, Asymmetric Lorentzian, and Voigt.

**Overall Trend Modeling**

Builds calibration relationships between estimated resonance frequencies and known sample volumes. Available regression approaches include linear, quadratic, cubic, logarithmic, and piecewise-linear models. The resulting calibration model can be exported as a JSON file for use in Measurement.

### Utilities

**Single Tone Sampler**

Generates and records a single sine-wave tone to test audio devices and inspect RMS response at a selected frequency. It can also support other RMS-based experimental investigations.

**Wave Viewer**

An interactive browser for WAV recordings, with multi-file waveform visualization, zooming, panning, timeline navigation, and amplitude inspection.

## Running from source

Acoustic Volumeter is a Python-based application. To run the source code, Python 3.8 or newer and the required dependencies are needed.

### Installation

```bash
git clone https://github.com/haqueshenas/Acoustic-Volumeter.git
cd Acoustic-Volumeter
git checkout v1.0.0
python -m pip install -r requirements.txt
```

### Launch the main application

```bash
python __main__.py
```

### Run individual modules

Each tool can also be launched independently:

```bash
python measurement.py       # Measurement
python profiling.py         # Acoustic Profiling
python r_modeling.py        # Resonance Modeling
python overall_trend.py     # Overall Trend Modeling
python single_tone.py       # Single Tone Sampler
python wave_viewer.py       # Wave Viewer
```

All source-code execution options require the dependencies listed in `requirements.txt`.

**Platform note:** The pre-built application provided with this release is for Windows x64. Running the source code on other operating systems may require compatible Python dependencies, Tk/Tkinter support, and audio input/output drivers. A pre-built macOS or Linux application is not provided with this release.

## Hardware requirements and safety

Acoustic Volumeter is not a standalone software-only measurement system. Volumetric measurements require a compatible DIY acoustic sensor, an appropriate audio input/output interface, and a correctly assembled hardware connection.

The sensor construction, circuit connections, and experimental setup are described in the published article. Please consult its Materials and Methods section before assembling or operating the hardware.

Observe the following precautions during operation:

- Always begin with very low playback levels (e.g., 0.005–0.01 on the application's 0–1 amplitude scale) and increase gradually only when necessary.
- Do not exceed the normal operating ranges of the sound card, amplifier, or custom circuitry.
- Check the selected input and output devices before running a frequency sweep.
- Disable microphone monitoring or audio loopback (such as "Listen to this device") and suppress system notification sounds to prevent unwanted feedback.
- Use the Single Tone Sampler or Acoustic Profiling tools to check signal levels before conducting full sweeps.
- Calibration is specific to the sensor assembly, audio interface, and measurement conditions. Recalibration is recommended after hardware changes. Significant changes in ambient conditions, including temperature and humidity, should also be evaluated.

Always inspect calibration and measurement results before using them in scientific analyses.

## Citation

If you use Acoustic Volumeter or its underlying methodology in scientific research, please cite the following peer-reviewed publication:

Haghshenas, A., & Emam, Y. (2025). Reimagined microphone-free acoustic volumetry: An open, DIY platform for global phenotyping. *Smart Agricultural Technology, 12*, 101430.

https://doi.org/10.1016/j.atech.2025.101430

### Software citation

For the specific software release used in your research, please also cite its Zenodo record:

Haghshenas, A. (2025). *haqueshenas/Acoustic-Volumeter: Acoustic Volumeter v1.0.0 (v1.0.0).* Zenodo.

https://doi.org/10.5281/zenodo.15280747

## License

The Acoustic Volumeter software is released under the [MIT License](LICENSE).

Copyright (C) 2025 Abbas Haghshenas.

Please refer to the `LICENSE` file in the repository for the complete license terms.

## About and contact

<div align="center">

**Acoustic Volumeter**

An open-source research tool developed as part of the **Easy Phenotyping Lab (EPL)** initiative, which aims to share open, accessible, and reliable computational tools for plant and crop phenotyping research.

[Easy Phenotyping Lab](https://haqueshenas.github.io/EPL/) · [GitHub repository](https://github.com/haqueshenas/Acoustic-Volumeter) · [GitHub Releases](https://github.com/haqueshenas/Acoustic-Volumeter/releases)

**Developed by Abbas Haghshenas**

[haqueshenas@gmail.com](mailto:haqueshenas@gmail.com)

</div>
