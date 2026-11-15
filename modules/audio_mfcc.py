import os
import tempfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr


def analyze_audio_mfcc(audio_path, fig_mfcc, n_mfcc=13):
    """
    Alur preprocessing & ekstraksi fitur:
    1. Load audio asli (y_asli, sr) -> Audio BEFORE
    2. Preprocessing langsung dengan noisereduce (y_bersih -> Audio AFTER)
    3. Simpan file sementara WAV After untuk diputar di UI
    4. Ekstraksi 13 koefisien MFCC dari y_bersih & plot Heatmap
    """
    nama_file = os.path.basename(audio_path)

    # 1. Load Audio Asli (Audio BEFORE preprocessing)
    y_asli, sr = librosa.load(audio_path)
    durasi = librosa.get_duration(y=y_asli, sr=sr)

    # 2. Preprocessing Noise Reduction langsung pada audio asli (Audio AFTER)
    y_bersih = nr.reduce_noise(
        y=y_asli,
        sr=sr,
        n_std_thresh_stationary=1.0,
        stationary=True
    )
    y_bersih = np.clip(y_bersih, -1.0, 1.0)

    # 3. Simpan audio After ke folder temp sistem (format PCM_16)
    temp_dir = tempfile.gettempdir()
    after_path = os.path.join(temp_dir, "uts_audio_after_clean.wav")
    sf.write(after_path, y_bersih, sr, subtype="PCM_16")

    # 4. Ekstraksi Fitur MFCC (Sidik Jari Suara BERSIH)
    mfcc = librosa.feature.mfcc(y=y_bersih, sr=sr, n_mfcc=n_mfcc)

    df_stats = pd.DataFrame({
        "Koefisien": [f"MFCC_{i+1}" for i in range(n_mfcc)],
        "Mean": np.round(np.mean(mfcc, axis=1), 4),
        "Std Dev": np.round(np.std(mfcc, axis=1), 4),
        "Min": np.round(np.min(mfcc, axis=1), 4),
        "Max": np.round(np.max(mfcc, axis=1), 4),
    })

    # 5. Visualisasi MFCC Heatmap
    fig_mfcc.clear()
    ax = fig_mfcc.add_subplot(1, 1, 1)
    img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, ax=ax)
    fig_mfcc.colorbar(img, ax=ax)

    ax.set_yticks(np.arange(n_mfcc) + 0.5)
    ax.set_yticklabels([f"MFCC {i+1}" for i in range(n_mfcc)], fontsize=8.5)
    ax.set_title(f"MFCC (Sidik Jari Suara BERSIH) - {nama_file}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Koefisien MFCC", fontsize=9)
    ax.set_xlabel("Waktu (detik)", fontsize=9)
    ax.tick_params(labelsize=8.5)

    fig_mfcc.tight_layout()

    metadata = {
        "sample_rate": sr,
        "durasi": round(durasi, 2),
        "shape_mfcc": mfcc.shape,
        "before_path": os.path.abspath(audio_path),
        "after_path": after_path
    }

    return df_stats, metadata