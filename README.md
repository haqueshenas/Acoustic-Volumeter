# Acoustic Volumeter v1.0

An open-source, DIY, microphone-free acoustic volumetry platform for rapid, precise volume measurements—ideal for global phenotyping applications.

**❗ For detailed instructions and background, please see [README.pdf](README.pdf).**

---

## Overview

Acoustic Volumeter implements the microphone-free method of Haghshenas & Emam (2025). It measures shifts in a chamber’s resonance peak to estimate volume, with accuracy down to single-grain scales.

---

## Installation

### Option A: Pre-built Executable (Windows)
1. Download `AcousticVolumeter.zip` from this repo.  
2. Unzip.  
3. Run **AcousticVolumeter.exe**.

### Option B: From Source
```bash
git clone https://github.com/haqueshenas/Acoustic-Volumeter.git
cd Acoustic-Volumeter
pip install -r requirements.txt
```

### Option C: Run Individual Modules
```bash
python measurement.py       # Direct volumetry
python profiling.py        # Acoustic profiling
python r_modeling.py       # Resonance modeling
python overall_trend.py    # Trend modeling
python single_tone.py      # Single-tone sampler
python wave_viewer.py      # WAV browser
```

> **Note:** All source options require Python 3.8+ and the dependencies listed in `requirements.txt`.

---

## Quick Tools at a Glance

- **Measurement**: Peak-fit EST & trend models  
- **Calibration**:  
  - Acoustic Profiling (frequency-sweep RMS)  
  - Resonance Modeling (Gaussian, Lorentzian, Voigt)  
  - Overall Trend Modeling (linear, polynomial, log, piecewise)  
- **Utilities**:  
  - Single Tone Sampler  
  - Wave Viewer (drag-and-drop WAV browser)

---

## Disclaimer

Use Acoustic Volumeter at your own risk. The developers make no warranty of any kind, express or implied, including but not limited to merchantability, fitness for a particular purpose, or non infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability arising from the use of this software.
Warning:
Please take special care not to exceed the standard input/output ranges of your sound card, amplifier, or custom circuitry. High playback amplitudes, improper amplifier settings, or mis‐wiring can damage your hardware or computer. Always start with low levels (e.g. 0.005–0.01) and verify each component under the Single Tone Sampler or Acoustic Profiling tool before running full sweeps.

---

## Citation

If you use this tool, please cite:  
Haghshenas A. & Emam Y. (2025). *Reimagined Microphone-Free Acoustic Volumetry: An Open, DIY Platform for Global Phenotyping.* bioRxiv.
Haghshenas, A. (2025). haqueshenas/Acoustic-Volumeter: Acoustic Volumeter v1.0.0 (v1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.15280747](https://doi.org/10.5281/zenodo.15280747)

---

## License

Released under the [MIT License](LICENSE).

---

## About This Tool

This software is part of the Easy Phenotyping Lab (EPL)—a non-profit initiative aimed at sharing open, affordable, and reliable computational tools for plant and crop phenotyping research. Visit https://haqueshenas.github.io/EPL for more tools and modules developed under VPL.

---

## Contact

**Abbas Haghshenas**  
haqueshenas@gmail.com  
