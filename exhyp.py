import numpy as np
import wave
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# SYNTHETIC METRONOME → EXTRATONE / HYPERTONE EXPERIMENT V2
#
# Erzeugt:
#   1. WAV-Datei für jede BPM-Stufe
#   2. FFT-Spektrum für jede Stufe
#   3. Übersicht der theoretischen Alias-Frequenzen
#
# Keine externe Audio-Library notwendig.
# ============================================================


# ============================================================
# 1. EXPERIMENT-PARAMETER
# ============================================================

# ---------- Audio ----------
SAMPLE_RATE = 44100       # 44100 / 48000 / 96000 / 192000 Hz
DURATION = 3.0            # Sekunden pro Testdatei

# ---------- BPM-Testreihe ----------
BPM_VALUES = [
    3600,
    10000,
    50000,
    100000,
    250000,
    500000,
    750000,
    1000000,
    1200000,
    1500000,
    2000000,
    3000000,
    5000000,
]

# ---------- Synthetischer Metronom-Klick ----------
CLICK_LENGTH = 0.003      # Sekunden
CLICK_FREQ = 3000         # Hz
CLICK_FREQ_2 = 5200       # Hz

# ---------- Mischung ----------
CLICK_MIX = 0.65           # tonaler Anteil
NOISE_MIX = 0.35           # Rauschanteil
HARMONIC_2_MIX = 0.30      # zweiter Resonanzanteil

# ---------- Hüllkurven ----------
DECAY = 0.0018             # Ton-Abklingzeit
NOISE_DECAY = 0.0009       # Noise-Abklingzeit

# ---------- Rauschen ----------
NOISE_SEED = 42

# ---------- Pegel ----------
OUTPUT_PEAK = 0.95

# ---------- FFT ----------
FFT_SIZE = 65536
FFT_WINDOW_START = 0.5    # Analyse beginnt nach dieser Zeit
FFT_MAX_FREQUENCY = None   # z.B. 22050; None = komplette Nyquist-Bandbreite

# ---------- Ausgabeordner ----------
OUTPUT_DIR = Path("hypertone_v2_output")


# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def bpm_to_hz(bpm):
    """BPM -> Wiederholungsfrequenz."""
    return bpm / 60.0


def hz_to_bpm(hz):
    """Wiederholungsfrequenz -> BPM."""
    return hz * 60.0


def calculate_alias_frequency(frequency, sample_rate):
    """
    Berechnet die auf den hörbaren Bereich gefaltete Frequenz.

    Für ein digitales System mit Sample Rate Fs werden Frequenzen
    über Nyquist gespiegelt.

    Beispiel bei 44.1 kHz:

        25 kHz -> 19.1 kHz
        30 kHz -> 14.1 kHz
        40 kHz -> 4.1 kHz
    """

    nyquist = sample_rate / 2.0

    # Frequenz zunächst auf einen Bereich 0 ... Fs falten
    frequency = frequency % sample_rate

    # Spiegelung oberhalb Nyquist
    if frequency > nyquist:
        frequency = sample_rate - frequency

    return abs(frequency)


def create_metronome_click():
    """
    Erzeugt einen synthetischen Metronom-Klick.

    Der Klang besteht aus:
      - kurzer Sinusresonanz
      - zweiter Resonanz
      - kurzer Noise-Komponente
    """

    samples = max(1, int(SAMPLE_RATE * CLICK_LENGTH))
    t = np.arange(samples) / SAMPLE_RATE

    # Tonale Hüllkurve
    envelope = np.exp(-t / DECAY)

    tone_1 = (
        np.sin(2 * np.pi * CLICK_FREQ * t)
        * envelope
    )

    tone_2 = (
        np.sin(2 * np.pi * CLICK_FREQ_2 * t)
        * envelope
        * HARMONIC_2_MIX
    )

    # Noise-Komponente
    rng = np.random.default_rng(NOISE_SEED)

    noise = rng.standard_normal(samples)

    noise_envelope = np.exp(-t / NOISE_DECAY)

    noise *= noise_envelope

    # Komponenten mischen
    click = (
        CLICK_MIX * (tone_1 + tone_2)
        + NOISE_MIX * noise
    )

    # Kleiner Fade-Out gegen harte Samplegrenze
    fade_samples = min(
        samples,
        max(1, int(SAMPLE_RATE * 0.0002))
    )

    click[-fade_samples:] *= np.linspace(
        1,
        0,
        fade_samples
    )

    # Einzelnen Klick normalisieren
    peak = np.max(np.abs(click))

    if peak > 0:
        click /= peak

    return click


def create_click_train(bpm):
    """
    Erzeugt eine periodische Folge des Metronom-Klicks.

    Wichtig:
    Die Klickfrequenz wird direkt aus BPM berechnet.

        frequency = BPM / 60
    """

    click_rate = bpm_to_hz(bpm)

    total_samples = int(
        SAMPLE_RATE * DURATION
    )

    output = np.zeros(
        total_samples,
        dtype=np.float64
    )

    click = create_metronome_click()

    # Abstand zwischen Klicks in Samples
    interval = SAMPLE_RATE / click_rate

    position = 0.0

    while position < total_samples:

        sample_position = int(
            round(position)
        )

        if sample_position < total_samples:

            end = min(
                total_samples,
                sample_position + len(click)
            )

            length = end - sample_position

            if length > 0:
                output[
                    sample_position:end
                ] += click[:length]

        position += interval

    # Normalisieren
    peak = np.max(np.abs(output))

    if peak > 0:
        output *= OUTPUT_PEAK / peak

    return output


def save_wav(filename, audio):
    """Speichert Mono-16-bit-WAV."""

    pcm = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    with wave.open(str(filename), "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(
            pcm.tobytes()
        )


def calculate_fft(audio):
    """
    FFT-Analyse eines Ausschnitts des Signals.
    """

    start_sample = int(
        FFT_WINDOW_START * SAMPLE_RATE
    )

    segment = audio[start_sample:]

    # Nur FFT_SIZE Samples verwenden
    if len(segment) < FFT_SIZE:

        padded = np.zeros(FFT_SIZE)

        padded[:len(segment)] = segment

        segment = padded

    else:

        segment = segment[:FFT_SIZE]

    # Hann-Fenster
    window = np.hanning(len(segment))

    windowed = segment * window

    spectrum = np.fft.rfft(windowed)

    magnitude = np.abs(spectrum)

    # Frequenzachse
    frequencies = np.fft.rfftfreq(
        len(segment),
        1 / SAMPLE_RATE
    )

    # dB
    magnitude_db = 20 * np.log10(
        np.maximum(magnitude, 1e-12)
    )

    return frequencies, magnitude_db


def save_fft_plot(
    frequencies,
    magnitude_db,
    bpm,
    filename
):
    """Speichert FFT-Spektrum als PNG."""

    plt.figure(figsize=(12, 6))

    plt.plot(
        frequencies,
        magnitude_db
    )

    plt.title(
        f"FFT Spectrum — {bpm:,} BPM "
        f"({bpm_to_hz(bpm):,.3f} Hz)"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")

    if FFT_MAX_FREQUENCY is not None:

        plt.xlim(
            0,
            min(
                FFT_MAX_FREQUENCY,
                SAMPLE_RATE / 2
            )
        )

    else:

        plt.xlim(
            0,
            SAMPLE_RATE / 2
        )

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()


# ============================================================
# 3. HAUPTPROGRAMM
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("SYNTHETIC METRONOME / EXTRATONE / HYPERTONE V2")
    print("=" * 70)

    print()
    print(f"Sample Rate : {SAMPLE_RATE:,} Hz")
    print(
        f"Nyquist     : "
        f"{SAMPLE_RATE / 2:,.1f} Hz"
    )
    print(
        f"Duration    : "
        f"{DURATION:.2f} s"
    )

    print()
    print("-" * 70)
    print(
        f"{'BPM':>12} "
        f"{'Frequency':>15} "
        f"{'Alias':>15} "
        f"{'Status':>15}"
    )
    print("-" * 70)

    results = []

    for bpm in BPM_VALUES:

        frequency = bpm_to_hz(bpm)

        alias = calculate_alias_frequency(
            frequency,
            SAMPLE_RATE
        )

        nyquist = SAMPLE_RATE / 2

        if frequency <= nyquist:

            status = "DIRECT"

        else:

            status = "ALIASED"

        print(
            f"{bpm:>12,} "
            f"{frequency:>15,.3f} "
            f"{alias:>15,.3f} "
            f"{status:>15}"
        )

        # ----------------------------------------------------
        # Audio erzeugen
        # ----------------------------------------------------

        audio = create_click_train(bpm)

        wav_filename = (
            OUTPUT_DIR
            / f"metronome_{bpm}BPM.wav"
        )

        save_wav(
            wav_filename,
            audio
        )

        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        frequencies, magnitude_db = calculate_fft(
            audio
        )

        fft_filename = (
            OUTPUT_DIR
            / f"spectrum_{bpm}BPM.png"
        )

        save_fft_plot(
            frequencies,
            magnitude_db,
            bpm,
            fft_filename
        )

        results.append({
            "bpm": bpm,
            "frequency": frequency,
            "alias": alias,
            "status": status
        })

    # ========================================================
    # 4. GESAMTÜBERSICHT
    # ========================================================

    print()
    print("=" * 70)
    print("DETAILS")
    print("=" * 70)

    for result in results:

        print()

        print(
            f"BPM: "
            f"{result['bpm']:,}"
        )

        print(
            f"Original frequency: "
            f"{result['frequency']:,.3f} Hz"
        )

        print(
            f"Theoretical alias: "
            f"{result['alias']:,.3f} Hz"
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        if result["frequency"] > SAMPLE_RATE / 2:

            print(
                "  -> Die ursprüngliche "
                "Wiederholungsfrequenz liegt "
                "oberhalb von Nyquist."
            )

            print(
                "  -> Der hörbare Anteil entsteht "
                "durch digitale Frequenzfaltung."
            )

    # ========================================================
    # 5. GESAMT-FFT MIT ALIAS-FREQUENZEN
    # ========================================================

    plt.figure(figsize=(14, 7))

    for result in results:

        bpm = result["bpm"]

        audio = create_click_train(bpm)

        frequencies, magnitude_db = calculate_fft(
            audio
        )

        # Relative Darstellung:
        # jedes Spektrum wird auf seinen Peak normiert
        magnitude_db -= np.max(magnitude_db)

        plt.plot(
            frequencies,
            magnitude_db,
            label=f"{bpm:,} BPM"
        )

    plt.title(
        "Comparison of Extratone / Hypertone Spectra"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative Magnitude (dB)")

    plt.xlim(
        0,
        FFT_MAX_FREQUENCY
        if FFT_MAX_FREQUENCY is not None
        else SAMPLE_RATE / 2
    )

    plt.ylim(-100, 5)

    plt.grid(True, alpha=0.25)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "ALL_SPECTRA_COMPARISON.png",
        dpi=180
    )

    plt.close()

    print()
    print("=" * 70)
    print("FERTIG")
    print("=" * 70)

    print()
    print(
        f"Alle Dateien befinden sich in:"
    )

    print(
        f"    {OUTPUT_DIR.resolve()}"
    )

    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
