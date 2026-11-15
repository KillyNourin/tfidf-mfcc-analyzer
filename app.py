import os
import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit, QTabWidget,
    QFrame, QFileDialog, QTableWidget, QTableWidgetItem, QScrollArea,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def resource_path(relative_path):
    """Mendapatkan path absolut untuk mode script maupun PyInstaller .exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class MplCanvas(FigureCanvas):
    """Widget kanvas Matplotlib agar grafik tampil langsung di dalam jendela desktop"""
    def __init__(self, width=6, height=3.2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#FFFFFF")
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class FeatureExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UTS Pengolahan Data Tidak Terstruktur - TF-IDF & MFCC Analyzer")
        self.resize(1360, 840)

        # Pastikan folder dataset ada
        os.makedirs("datasets/text", exist_ok=True)
        os.makedirs("datasets/audio", exist_ok=True)

        self.current_csv_path = None
        self.current_audio_path = None

        # Pemutar Audio Desktop
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.init_ui()
        self.load_stylesheet()
        self.refresh_local_files()

    def load_stylesheet(self):
        qss_path = resource_path(os.path.join("assets", "style.qss"))
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # ================= HEADER =================
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(22, 16, 22, 16)
        header_layout.setSpacing(6)

        badge_row = QHBoxLayout()
        badge = QLabel("UTS Pengolahan Data Tidak Terstruktur • Desktop App")
        badge.setObjectName("HeaderBadge")
        badge_row.addWidget(badge)
        badge_row.addStretch()

        title = QLabel("Ekstraksi Fitur Data Teks (TF-IDF) & Data Audio (MFCC)")
        title.setObjectName("HeaderTitle")

        subtitle = QLabel(
            "Aplikasi desktop mandiri untuk mengeksplorasi tahapan preprocessing (NLTK & Librosa) "
            "serta ekstraksi fitur pada data tidak terstruktur."
        )
        subtitle.setObjectName("HeaderSubtitle")

        header_layout.addLayout(badge_row)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header_card)

        # ================= TABS =================
        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_text_tab(), "1. Analisis Data Teks (TF-IDF)")
        self.tabs.addTab(self.build_audio_tab(), "2. Analisis Data Audio (MFCC)")
        main_layout.addWidget(self.tabs)

    # ---------------------------------------------------------
    # TAB 1: DATA TEKS (TF-IDF)
    # ---------------------------------------------------------
    def build_text_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        # Panel Kiri: Kontrol Dataset Teks
        left_card = QFrame()
        left_card.setObjectName("PanelCard")
        left_card.setFixedWidth(360)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(10)

        lbl_title = QLabel("Konfigurasi Dataset Teks")
        lbl_title.setObjectName("SectionTitle")
        lbl_desc = QLabel("Pilih file CSV dari folder datasets/text atau cari file dari komputer.")
        lbl_desc.setObjectName("SectionDesc")
        lbl_desc.setWordWrap(True)

        lbl_combo = QLabel("Dataset Lokal (datasets/text/):")
        lbl_combo.setObjectName("FieldLabel")
        self.combo_csv = QComboBox()
        self.combo_csv.currentTextChanged.connect(self.on_select_local_csv)

        btn_browse_csv = QPushButton("Browse File CSV Lain...")
        btn_browse_csv.setObjectName("SecondaryButton")
        btn_browse_csv.clicked.connect(self.browse_csv_file)

        lbl_idx = QLabel("Indeks Baris Dokumen yang Diinspeksi:")
        lbl_idx.setObjectName("FieldLabel")
        self.spin_doc_idx = QSpinBox()
        self.spin_doc_idx.setMinimum(0)
        self.spin_doc_idx.setMaximum(9999)

        btn_run_text = QPushButton("Jalankan Preprocessing & TF-IDF")
        btn_run_text.setObjectName("PrimaryButton")
        btn_run_text.clicked.connect(self.run_text_preview)

        self.lbl_text_status = QLabel("Status: Siap memproses data teks.")
        self.lbl_text_status.setObjectName("SectionDesc")
        self.lbl_text_status.setWordWrap(True)

        left_layout.addWidget(lbl_title)
        left_layout.addWidget(lbl_desc)
        left_layout.addWidget(lbl_combo)
        left_layout.addWidget(self.combo_csv)
        left_layout.addWidget(btn_browse_csv)
        left_layout.addWidget(lbl_idx)
        left_layout.addWidget(self.spin_doc_idx)
        left_layout.addSpacing(8)
        left_layout.addWidget(btn_run_text)
        left_layout.addWidget(self.lbl_text_status)
        left_layout.addStretch()

        # Panel Kanan: Scroll Area Hasil Preprocessing & TF-IDF
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        right_card = QFrame()
        right_card.setObjectName("PanelCard")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(10)

        sec1 = QLabel("Hasil Tahapan Preprocessing Teks (NLTK)")
        sec1.setObjectName("SectionTitle")
        right_layout.addWidget(sec1)

        self.txt_raw = self.create_readonly_box("1. Raw Text (Teks Asli):", 55)
        self.txt_case = self.create_readonly_box("2. Case Folding:", 45)
        self.txt_clean = self.create_readonly_box("3. Cleaning (Regex):", 45)
        self.txt_token = self.create_readonly_box("4. Tokenizing:", 45)
        self.txt_lemma = self.create_readonly_box("5. Stopword Removal & Lemmatization:", 45)

        for lbl, box in [self.txt_raw, self.txt_case, self.txt_clean, self.txt_token, self.txt_lemma]:
            right_layout.addWidget(lbl)
            right_layout.addWidget(box)

        sec2 = QLabel("Hasil Ekstraksi Fitur (Matriks & Grafik TF-IDF)")
        sec2.setObjectName("SectionTitle")
        right_layout.addSpacing(10)
        right_layout.addWidget(sec2)

        self.table_tfidf = QTableWidget()
        self.table_tfidf.setMinimumHeight(200)
        self.table_tfidf.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.table_tfidf)

        self.canvas_tfidf = MplCanvas(width=7, height=3.2)
        self.canvas_tfidf.setMinimumHeight(260)
        right_layout.addWidget(self.canvas_tfidf)

        scroll.setWidget(right_card)

        layout.addWidget(left_card)
        layout.addWidget(scroll)
        return tab

    # TAB 2: DATA AUDIO (MFCC)
    def build_audio_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        # Panel Kiri: Kontrol Dataset Audio
        left_card = QFrame()
        left_card.setObjectName("PanelCard")
        left_card.setFixedWidth(360)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(10)

        lbl_title = QLabel("Konfigurasi Dataset Audio")
        lbl_title.setObjectName("SectionTitle")
        lbl_desc = QLabel("Pilih file .wav/.mp3 dari datasets/audio atau cari dari komputer.")
        lbl_desc.setObjectName("SectionDesc")
        lbl_desc.setWordWrap(True)

        lbl_combo = QLabel("Dataset Lokal (datasets/audio/):")
        lbl_combo.setObjectName("FieldLabel")
        self.combo_wav = QComboBox()
        self.combo_wav.currentTextChanged.connect(self.on_select_local_wav)

        btn_browse_wav = QPushButton("Browse File Audio Lain...")
        btn_browse_wav.setObjectName("SecondaryButton")
        btn_browse_wav.clicked.connect(self.browse_wav_file)

        # Tombol Play / Stop Audio
        player_row = QHBoxLayout()
        btn_play = QPushButton("▶ Putar Audio")
        btn_play.setObjectName("SecondaryButton")
        btn_play.clicked.connect(self.play_audio)

        btn_stop = QPushButton("■ Stop")
        btn_stop.setObjectName("SecondaryButton")
        btn_stop.clicked.connect(self.player.stop)

        player_row.addWidget(btn_play)
        player_row.addWidget(btn_stop)

        btn_run_audio = QPushButton("Jalankan Preprocessing & MFCC")
        btn_run_audio.setObjectName("PrimaryButton")
        btn_run_audio.clicked.connect(self.run_audio_preview)

        self.lbl_audio_status = QLabel("Status: Siap memproses data audio.")
        self.lbl_audio_status.setObjectName("SectionDesc")
        self.lbl_audio_status.setWordWrap(True)

        left_layout.addWidget(lbl_title)
        left_layout.addWidget(lbl_desc)
        left_layout.addWidget(lbl_combo)
        left_layout.addWidget(self.combo_wav)
        left_layout.addWidget(btn_browse_wav)
        left_layout.addLayout(player_row)
        left_layout.addSpacing(8)
        left_layout.addWidget(btn_run_audio)
        left_layout.addWidget(self.lbl_audio_status)
        left_layout.addStretch()

        # Panel Kanan: Grafik Waveform & MFCC
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        right_card = QFrame()
        right_card.setObjectName("PanelCard")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(12)

        sec1 = QLabel("Visualisasi Preprocessing (Waveform Sebelum vs Sesudah Trimming)")
        sec1.setObjectName("SectionTitle")
        right_layout.addWidget(sec1)

        self.canvas_waveform = MplCanvas(width=7, height=3.0)
        self.canvas_waveform.setMinimumHeight(250)
        right_layout.addWidget(self.canvas_waveform)

        sec2 = QLabel("Visualisasi Ekstraksi Fitur (MFCC Heatmap & Statistik 13 Koefisien)")
        sec2.setObjectName("SectionTitle")
        right_layout.addWidget(sec2)

        self.canvas_mfcc = MplCanvas(width=7, height=3.2)
        self.canvas_mfcc.setMinimumHeight(260)
        right_layout.addWidget(self.canvas_mfcc)

        self.table_mfcc = QTableWidget()
        self.table_mfcc.setMinimumHeight(200)
        self.table_mfcc.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.table_mfcc)

        scroll.setWidget(right_card)

        layout.addWidget(left_card)
        layout.addWidget(scroll)
        return tab

    # FUNGSI PEMBANTU UI & EVENT HANDLER SEMENTARA
    def create_readonly_box(self, label_text, max_height):
        lbl = QLabel(label_text)
        lbl.setObjectName("FieldLabel")
        box = QTextEdit()
        box.setReadOnly(True)
        box.setMaximumHeight(max_height)
        return lbl, box

    def refresh_local_files(self):
        csv_files = [f for f in os.listdir("datasets/text") if f.lower().endswith(".csv")]
        self.combo_csv.clear()
        self.combo_csv.addItems(csv_files)

        wav_files = [f for f in os.listdir("datasets/audio") if f.lower().endswith((".wav", ".mp3"))]
        self.combo_wav.clear()
        self.combo_wav.addItems(wav_files)

    def on_select_local_csv(self, filename):
        if filename:
            self.current_csv_path = os.path.join("datasets", "text", filename)

    def browse_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Dataset CSV", "", "CSV Files (*.csv)")
        if path:
            self.current_csv_path = path
            self.lbl_text_status.setText(f"File eksternal dipilih: {os.path.basename(path)}")

    def on_select_local_wav(self, filename):
        if filename:
            self.current_audio_path = os.path.join("datasets", "audio", filename)

    def browse_wav_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Dataset Audio", "", "Audio Files (*.wav *.mp3)")
        if path:
            self.current_audio_path = path
            self.lbl_audio_status.setText(f"File eksternal dipilih: {os.path.basename(path)}")

    def play_audio(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "Peringatan", "Pilih file audio terlebih dahulu!")
            return
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.current_audio_path)))
        self.player.play()

    def populate_table(self, table_widget, df):
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        for r in range(len(df)):
            for c in range(len(df.columns)):
                table_widget.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))

    def run_text_preview(self):
        if not self.current_csv_path or not os.path.exists(self.current_csv_path):
            QMessageBox.warning(self, "Peringatan", "Tidak ada file CSV yang dipilih!")
            return

        # Preview UI sementara sebelum dihubungkan ke modules/text_tfidf.py
        idx = self.spin_doc_idx.value()
        self.txt_raw[1].setText(f"[Preview baris #{idx}] The acting and cinematography were breathtaking!")
        self.txt_case[1].setText("the acting and cinematography were breathtaking!")
        self.txt_clean[1].setText("the acting and cinematography were breathtaking")
        self.txt_token[1].setText("['the', 'acting', 'and', 'cinematography', 'were', 'breathtaking']")
        self.txt_lemma[1].setText("['acting', 'cinematography', 'breathtaking']")

        df_dummy = pd.DataFrame({
            "Term": ["breathtaking", "cinematography", "acting"],
            "TF-IDF Score": [0.642, 0.581, 0.498]
        })
        self.populate_table(self.table_tfidf, df_dummy)

        self.canvas_tfidf.axes.clear()
        self.canvas_tfidf.axes.barh(df_dummy["Term"], df_dummy["TF-IDF Score"], color="#2563EB")
        self.canvas_tfidf.axes.set_title("Top TF-IDF Terms (Preview UI)")
        self.canvas_tfidf.fig.tight_layout()
        self.canvas_tfidf.draw()

        self.lbl_text_status.setText(f"UI Sukses! Siap hubungkan modules/text_tfidf.py ({os.path.basename(self.current_csv_path)})")

    def run_audio_preview(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "Peringatan", "Tidak ada file audio yang dipilih!")
            return

        df_dummy = pd.DataFrame({
            "Koefisien": [f"MFCC_{i}" for i in range(1, 6)],
            "Mean": [-108.5, 79.4, -12.1, 18.7, -6.2],
            "Std Dev": [41.2, 19.8, 13.5, 10.4, 8.9]
        })
        self.populate_table(self.table_mfcc, df_dummy)
        self.lbl_audio_status.setText(f"UI Sukses! Siap hubungkan modules/audio_mfcc.py ({os.path.basename(self.current_audio_path)})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureExtractionApp()
    window.show()
    sys.exit(app.exec())