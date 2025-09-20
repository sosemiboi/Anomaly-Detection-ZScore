import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from collections import deque

def stream_psd_with_zbins(filename, dtype, sample_rate,
                          fft_bins=4096, block_size=4096,
                          history=100, threshold=3,
                          std_floor=2.0):
    """
    Stream PSD with adaptive noise floor, per-bin Z-score spectrum,
    block-level anomaly trace, and anomaly percentage display.
    DC offset excluded.
    """
    bytes_per_sample = np.dtype(dtype).itemsize * 2  # I + Q
    window = get_window("hann", block_size)
    U = (window**2).sum()

    psd_history = deque(maxlen=history)
    zscore_trace = deque(maxlen=500)

    plt.ion()
    fig, (ax_psd, ax_zbin, ax_ztrace) = plt.subplots(3, 1, figsize=(10, 12))

    dc_bin = fft_bins // 2  # index for DC

    with open(filename, "rb") as f:
        block_idx = 0
        while True:
            raw = f.read(block_size * bytes_per_sample)
            if not raw:
                break

            iq_array = np.frombuffer(raw, dtype=dtype)
            if len(iq_array) < block_size * 2:
                break

            i_data = iq_array[::2]
            q_data = iq_array[1::2]
            iq_complex = i_data + 1j * q_data

            iq_windowed = iq_complex * window
            fft_array = np.fft.fftshift(np.fft.fft(iq_windowed, fft_bins))

            psd = (np.abs(fft_array) ** 2) / (U * sample_rate)
            psd_db = 10 * np.log10(psd + 1e-12)

            # Exclude DC offset
            psd_db[dc_bin] = np.nan  

            # Adaptive noise floor (median of current block, ignoring NaN)
            noise_floor_est = np.nanmedian(psd_db)
            psd_db = np.maximum(psd_db, noise_floor_est - 3)

            psd_history.append(psd_db)

            anomalies = np.zeros_like(psd_db, dtype=bool)
            block_score = 0
            z_scores = np.zeros_like(psd_db)

            if len(psd_history) >= 20:
                mean_psd = np.nanmean(psd_history, axis=0)
                std_psd = np.nanstd(psd_history, axis=0)

                std_psd = np.maximum(std_psd, std_floor)

                z_scores = (psd_db - mean_psd) / std_psd
                z_scores[dc_bin] = np.nan  # DC excluded

                anomalies = np.abs(z_scores) > threshold

                block_score = np.percentile(np.abs(z_scores[~np.isnan(z_scores)]), 95)

            # Percentage of anomalous bins in this block
            anomaly_percent = 100 * np.sum(anomalies) / anomalies.size

            zscore_trace.append(block_score)
            freqs = np.linspace(-sample_rate/2, sample_rate/2, fft_bins)

            # ---- PSD plot ----
            ax_psd.clear()
            ax_psd.plot(freqs, psd_db, lw=1, color="blue", label="PSD")
            ax_psd.axhline(noise_floor_est, color="green", linestyle="--", lw=1,
                           label=f"Adaptive Noise Floor ({noise_floor_est:.1f} dB)")
            if anomalies.any():
                ax_psd.scatter(freqs[anomalies], psd_db[anomalies],
                               color="red", s=15, label="Anomaly")
            ax_psd.set_ylabel("PSD (dB/Hz)")
            ax_psd.set_title("Real-Time PSD with Anomalies (DC removed)")
            ax_psd.grid(True)
            ax_psd.legend(loc="upper right")

            # Show anomaly percentage
            ax_psd.text(0.02, 0.95, f"Anomaly %: {anomaly_percent:.2f}%",
                        transform=ax_psd.transAxes, fontsize=10,
                        verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

            # ---- Per-bin Z-score spectrum ----
            ax_zbin.clear()
            ax_zbin.plot(freqs, z_scores, lw=1, color="purple", label="Z-score per bin")
            ax_zbin.axhline(threshold, color="orange", linestyle="--", lw=1, label="Threshold")
            ax_zbin.axhline(-threshold, color="orange", linestyle="--", lw=1)
            ax_zbin.set_ylabel("Z-score")
            ax_zbin.set_title("Per-Bin Z-score Spectrum (Anomaly Detection)")
            ax_zbin.grid(True)
            ax_zbin.legend(loc="upper right")

            # ---- Block-level Z-score trace ----
            ax_ztrace.clear()
            ax_ztrace.plot(zscore_trace, lw=1, color="blue")
            ax_ztrace.axhline(threshold, color="orange", linestyle="--", lw=1, label="Threshold")
            ax_ztrace.set_xlabel("Block Index (time)")
            ax_ztrace.set_ylabel("95th Percentile |Z|")
            ax_ztrace.set_title("Block-Level Anomaly Score Over Time")
            ax_ztrace.grid(True)
            ax_ztrace.legend(loc="upper right")

            plt.pause(0.01)
            block_idx += 1

    plt.ioff()
    plt.show()


# Example usage
filename = r"C:\Users\saich\OneDrive\Desktop\noaa15-2.raw"
dtype = np.int8   # HackRF default
sample_rate = 2e6
stream_psd_with_zbins(filename, dtype, sample_rate)

