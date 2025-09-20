# Real-Time Spectrum Anomaly Detection

This project provides a real-time Power Spectral Density (PSD) visualizer with anomaly detection from raw IQ data (e.g., HackRF, RTL-SDR, NOAA satellite captures).  
It uses Z-score based anomaly detection with an adaptive noise floor to highlight unexpected signals in the spectrum.

---

## Features
- Real-time PSD streaming from raw IQ files (`.raw`)  
- Hann windowing + normalization for accurate FFTs  
- Adaptive noise floor estimation (tracks changing backgrounds)  
- Z-score anomaly detection per block:
  - Detects unexpected narrowband spikes  
  - Detects wideband/jamming-type anomalies  
- DC offset removal (avoids false spikes at DC)  
- Continuous Z-score trace:
  - Shows how anomaly strength evolves over time  
- Live Matplotlib plots:
  - Top: PSD with anomalies highlighted in red  
  - Bottom: Z-score time trace with threshold  

---

## Requirements
Install dependencies with:
```bash
pip install numpy scipy matplotlib
