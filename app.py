import os
import sys
import shutil
from pathlib import Path

import matplotlib
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit, QTabWidget,
    QFrame, QFileDialog, QTableWidget, QTableWidgetItem, QScrollArea,
    QHeaderView, QMessageBox, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from modules.text_tfidf import analyze_text_csv
from modules.audio_mfcc import analyze_audio_mfcc


# Konfigurasi gaya grafik Matplotlib agar menyatu dengan latar hitam Alphamark
matplotlib.rcParams.update({
    "figure.facecolor": "#121212",
    "axes.facecolor": "#0D0D0D",
    "axes.edgecolor": "#2A2A2A",
    "axes.labelcolor": "#F5F5F5",
    "text.color": "#FFFFFF",
    "xtick.color": "#A3A3A3",
    "ytick.color": "#A3A3A3",
})


BASE_DIR = Path(__file__).resolve().parent
TEXT_DATASET_DIR = BASE_DIR / "datasets" / "text"
AUDIO_DATASET_DIR = BASE_DIR / "datasets" / "audio"


def resource_path(relative_path):
    """Mendapatkan path absolut untuk mode script maupun PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        base_path = BASE_DIR
    return str(base_path / relative_path)


def apply_glass_shadow(widget):
    """Memberikan efek elevasi halus pada kartu UI."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(24)
    shadow.setXOffset(0)
    shadow.setYOffset(6)
    shadow.setColor(QColor(0, 0, 0, 140))
    widget.setGraphicsEffect(shadow)


class MplCanvas(FigureCanvas):
    """Kanvas Matplotlib dengan warna dasar hitam matte yang menyatu dengan panel."""
    def __init__(self, width=6, height=3.2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#121212")
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("#0D0D0D")
        super().__init__(self.fig)


class FeatureExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scrapping Val Audio")
        self.resize(1360, 840)

        TEXT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        self.current_csv_path = None
        self.current_text_export_df = None
        self.current_audio_path = None
        self.audio_before_path = None
        self.audio_after_path = None

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.init_ui()
        self.load_stylesheet()
        self.refresh_local_files()

    def load_stylesheet(self):
        qss_path = Path(resource_path(Path("assets") / "style.qss"))
        if qss_path.exists():
            with qss_path.open("r", encoding="utf-8") as file:
                self.setStyleSheet(file.read())

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header Studio Editorial
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        apply_glass_shadow(header_card)
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(24, 18, 24, 18)
        header_layout.setSpacing(6)

        badge_row = QHBoxLayout()
        badge = QLabel("PENGOLAHAN DATA TIDAK TERSTRUKTUR")
        badge.setObjectName("HeaderBadge")
        badge_row.addWidget(badge)
        badge_row.addStretch()

        title = QLabel("Ekstraksi Fitur Data Teks (TF-IDF) dan Data Audio (MFCC)")
        title.setObjectName("HeaderTitle")

        header_layout.addLayout(badge_row)
        header_layout.addWidget(title)
        main_layout.addWidget(header_card)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_text_tab(), "01  ▪  Analisis Data Teks (TF-IDF)")
        self.tabs.addTab(self.build_audio_tab(), "02  ▪  Analisis Data Audio (MFCC)")
        main_layout.addWidget(self.tabs)

    # TAB 1: DATA TEKS (TF-IDF)
    def build_text_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        left_card = QFrame()
        left_card.setObjectName("PanelCard")
        left_card.setFixedWidth(360)
        apply_glass_shadow(left_card)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)

        lbl_title = QLabel("▪  Konfigurasi Dataset Teks")
        lbl_title.setObjectName("SectionTitle")
        lbl_desc = QLabel("Pilih file CSV dari folder datasets/text atau unggah file baru.")
        lbl_desc.setObjectName("SectionDesc")
        lbl_desc.setWordWrap(True)

        lbl_combo = QLabel("Dataset Lokal (datasets/text/):")
        lbl_combo.setObjectName("FieldLabel")
        self.combo_csv = QComboBox()
        self.combo_csv.currentTextChanged.connect(self.on_select_local_csv)

        btn_browse_csv = QPushButton("+ Browse & Simpan CSV")
        btn_browse_csv.setObjectName("SecondaryButton")
        btn_browse_csv.clicked.connect(self.browse_csv_file)

        lbl_idx = QLabel("Indeks Baris Dokumen:")
        lbl_idx.setObjectName("FieldLabel")
        self.spin_doc_idx = QSpinBox()
        self.spin_doc_idx.setMinimum(0)
        self.spin_doc_idx.setMaximum(9999)

        lbl_language = QLabel("Bahasa Dataset:")
        lbl_language.setObjectName("FieldLabel")
        self.combo_language = QComboBox()
        self.combo_language.addItem("Bahasa Inggris", "english")
        self.combo_language.addItem("Bahasa Indonesia", "indonesian")

        btn_run_text = QPushButton("Jalankan Preprocessing & TF-IDF  →")
        btn_run_text.setObjectName("PrimaryButton")
        btn_run_text.clicked.connect(self.run_text_preview)

        self.btn_download_text = QPushButton("↓ Download Hasil Preprocessing & TF-IDF")
        self.btn_download_text.setObjectName("SecondaryButton")
        self.btn_download_text.setEnabled(False)
        self.btn_download_text.clicked.connect(self.download_text_results)

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
        left_layout.addWidget(lbl_language)
        left_layout.addWidget(self.combo_language)
        left_layout.addSpacing(8)
        left_layout.addWidget(btn_run_text)
        left_layout.addWidget(self.btn_download_text)
        left_layout.addWidget(self.lbl_text_status)
        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        right_card = QFrame()
        right_card.setObjectName("PanelCard")
        apply_glass_shadow(right_card)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(22, 22, 22, 22)
        right_layout.setSpacing(10)

        sec1 = QLabel("▪  Hasil Tahapan Preprocessing Teks (NLTK)")
        sec1.setObjectName("SectionTitle")
        right_layout.addWidget(sec1)

        self.txt_raw = self.create_readonly_box("1. Raw Text (Teks Asli):", 55)
        self.txt_case = self.create_readonly_box("2. Case Folding:", 45)
        self.txt_clean = self.create_readonly_box("3. Cleaning (Regex):", 45)
        self.txt_token = self.create_readonly_box("4. Tokenizing:", 45)
        self.txt_lemma = self.create_readonly_box("5. Stopword Removal & Lemmatization:", 45)

        for label, box in [
            self.txt_raw, self.txt_case, self.txt_clean,
            self.txt_token, self.txt_lemma
        ]:
            right_layout.addWidget(label)
            right_layout.addWidget(box)

        sec2 = QLabel("▪  Hasil Ekstraksi Fitur (Matriks & Grafik TF-IDF)")
        sec2.setObjectName("SectionTitle")
        right_layout.addSpacing(10)
        right_layout.addWidget(sec2)

        self.table_tfidf = QTableWidget()
        self.table_tfidf.setMinimumHeight(200)
        self.table_tfidf.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
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

        left_card = QFrame()
        left_card.setObjectName("PanelCard")
        left_card.setFixedWidth(360)
        apply_glass_shadow(left_card)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)

        lbl_title = QLabel("▪  Konfigurasi Dataset Audio")
        lbl_title.setObjectName("SectionTitle")
        lbl_desc = QLabel("Pilih file .wav/.mp3 dari folder datasets/audio atau unggah file baru.")
        lbl_desc.setObjectName("SectionDesc")
        lbl_desc.setWordWrap(True)

        lbl_combo = QLabel("Dataset Lokal (datasets/audio/):")
        lbl_combo.setObjectName("FieldLabel")
        self.combo_wav = QComboBox()
        self.combo_wav.currentTextChanged.connect(self.on_select_local_wav)

        btn_browse_wav = QPushButton("+ Browse & Simpan Audio")
        btn_browse_wav.setObjectName("SecondaryButton")
        btn_browse_wav.clicked.connect(self.browse_wav_file)

        btn_run_audio = QPushButton("Jalankan Preprocessing & MFCC  →")
        btn_run_audio.setObjectName("PrimaryButton")
        btn_run_audio.clicked.connect(self.run_audio_preview)

        lbl_player = QLabel("Pemutar Audio (Before vs After):")
        lbl_player.setObjectName("FieldLabel")
        row_ba = QHBoxLayout()

        self.btn_play_before = QPushButton("▶ Before (Asli)")
        self.btn_play_before.setObjectName("SecondaryButton")
        self.btn_play_before.setEnabled(False)
        self.btn_play_before.clicked.connect(
            lambda: self.play_audio_file(self.audio_before_path, "Before (Audio Asli)")
        )

        self.btn_play_after = QPushButton("▶ After (Clean)")
        self.btn_play_after.setObjectName("SecondaryButton")
        self.btn_play_after.setEnabled(False)
        self.btn_play_after.clicked.connect(
            lambda: self.play_audio_file(self.audio_after_path, "After (Noise Reduced)")
        )

        row_ba.addWidget(self.btn_play_before)
        row_ba.addWidget(self.btn_play_after)

        btn_stop = QPushButton("■ Stop Audio")
        btn_stop.setObjectName("SecondaryButton")
        btn_stop.clicked.connect(self.stop_audio)

        self.lbl_audio_status = QLabel("Status: Siap memproses data audio.")
        self.lbl_audio_status.setObjectName("SectionDesc")
        self.lbl_audio_status.setWordWrap(True)

        left_layout.addWidget(lbl_title)
        left_layout.addWidget(lbl_desc)
        left_layout.addWidget(lbl_combo)
        left_layout.addWidget(self.combo_wav)
        left_layout.addWidget(btn_browse_wav)
        left_layout.addSpacing(4)
        left_layout.addWidget(btn_run_audio)
        left_layout.addSpacing(6)
        left_layout.addWidget(lbl_player)
        left_layout.addLayout(row_ba)
        left_layout.addWidget(btn_stop)
        left_layout.addSpacing(4)
        left_layout.addWidget(self.lbl_audio_status)
        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        right_card = QFrame()
        right_card.setObjectName("PanelCard")
        apply_glass_shadow(right_card)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(22, 22, 22, 22)
        right_layout.setSpacing(12)

        sec_mfcc = QLabel(
            "▪  Visualisasi Ekstraksi Fitur (MFCC Heatmap & Statistik 13 Koefisien)"
        )
        sec_mfcc.setObjectName("SectionTitle")
        right_layout.addWidget(sec_mfcc)

        self.canvas_mfcc = MplCanvas(width=7, height=3.8)
        self.canvas_mfcc.setMinimumHeight(320)
        right_layout.addWidget(self.canvas_mfcc)

        self.table_mfcc = QTableWidget()
        self.table_mfcc.setMinimumHeight(240)
        self.table_mfcc.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        right_layout.addWidget(self.table_mfcc)

        scroll.setWidget(right_card)
        layout.addWidget(left_card)
        layout.addWidget(scroll)
        return tab

    # EVENT HANDLER & PEMBANTU UI
    def create_readonly_box(self, label_text, max_height):
        label = QLabel(label_text)
        label.setObjectName("FieldLabel")
        box = QTextEdit()
        box.setReadOnly(True)
        box.setMaximumHeight(max_height)
        return label, box

    def clear_text_results(self):
        """Bersihkan hasil lama agar tidak tertukar dengan dataset yang baru dipilih."""
        for label, box in [
            self.txt_raw, self.txt_case, self.txt_clean,
            self.txt_token, self.txt_lemma
        ]:
            box.clear()

        self.current_text_export_df = None
        if hasattr(self, "btn_download_text"):
            self.btn_download_text.setEnabled(False)

        self.table_tfidf.clear()
        self.table_tfidf.setRowCount(0)
        self.table_tfidf.setColumnCount(0)
        self.canvas_tfidf.axes.clear()
        self.canvas_tfidf.axes.set_facecolor("#0D0D0D")
        self.canvas_tfidf.draw()

    def refresh_local_files(self):
        csv_files = sorted(
            f.name for f in TEXT_DATASET_DIR.iterdir()
            if f.is_file() and f.suffix.lower() == ".csv"
        )
        self.combo_csv.clear()
        self.combo_csv.addItems(csv_files)

        wav_files = sorted(
            f.name for f in AUDIO_DATASET_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3")
        )
        self.combo_wav.clear()
        self.combo_wav.addItems(wav_files)

    def on_select_local_csv(self, filename):
        if not filename:
            self.current_csv_path = None
            return

        self.current_csv_path = str(TEXT_DATASET_DIR / filename)
        self.clear_text_results()
        self.lbl_text_status.setText(f"Dataset terpilih: {filename}")

    def browse_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Dataset CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        source_path = Path(path)
        dest_path = TEXT_DATASET_DIR / source_path.name
        try:
            if source_path.resolve() != dest_path.resolve():
                shutil.copy2(source_path, dest_path)

            self.refresh_local_files()
            self.combo_csv.setCurrentText(source_path.name)
            self.current_csv_path = str(dest_path)
            self.clear_text_results()
            self.lbl_text_status.setText(
                f"Tersimpan di datasets/text/: {source_path.name}"
            )
        except Exception as error:
            QMessageBox.critical(self, "Gagal Menyimpan CSV", str(error))

    def download_text_results(self):
        """Simpan tabel hasil preprocessing + skor TF-IDF ke file CSV."""
        if self.current_text_export_df is None or self.current_text_export_df.empty:
            QMessageBox.warning(
                self, "Peringatan",
                "Belum ada hasil preprocessing & TF-IDF yang bisa diunduh. Jalankan proses terlebih dahulu!"
            )
            return

        stem_name = Path(self.current_csv_path).stem if self.current_csv_path else "dataset"
        default_filename = f"hasil_preprocessing_tfidf_{stem_name}.csv"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Simpan Hasil Preprocessing & TF-IDF",
            default_filename,
            "CSV Files (*.csv)"
        )
        if not save_path:
            return

        if not save_path.lower().endswith(".csv"):
            save_path += ".csv"

        try:
            self.current_text_export_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            self.lbl_text_status.setText(f"File tersimpan: {os.path.basename(save_path)}")
            QMessageBox.information(
                self,
                "Download Berhasil",
                f"Hasil preprocessing dan matriks TF-IDF berhasil disimpan ke:\n{save_path}"
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "Gagal Menyimpan File",
                f"Terjadi kesalahan saat menyimpan CSV:\n{error}"
            )

    def on_select_local_wav(self, filename):
        if not filename:
            self.current_audio_path = None
            return

        self.stop_audio()
        self.current_audio_path = str(AUDIO_DATASET_DIR / filename)
        self.audio_before_path = os.path.abspath(self.current_audio_path)
        self.audio_after_path = None
        if hasattr(self, "btn_play_before"):
            self.btn_play_before.setEnabled(True)
            self.btn_play_after.setEnabled(False)

    def browse_wav_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Dataset Audio", "", "Audio Files (*.wav *.mp3)"
        )
        if not path:
            return

        self.stop_audio()
        source_path = Path(path)
        dest_path = AUDIO_DATASET_DIR / source_path.name
        try:
            if source_path.resolve() != dest_path.resolve():
                shutil.copy2(source_path, dest_path)

            self.refresh_local_files()
            self.combo_wav.setCurrentText(source_path.name)
            self.current_audio_path = str(dest_path)
            self.audio_before_path = str(dest_path.resolve())
            self.audio_after_path = None
            self.btn_play_before.setEnabled(True)
            self.btn_play_after.setEnabled(False)
            self.lbl_audio_status.setText(
                f"Tersimpan di datasets/audio/: {source_path.name}"
            )
        except Exception as error:
            QMessageBox.critical(self, "Gagal Menyimpan Audio", str(error))

    def play_audio_file(self, file_path, label_mode):
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(
                self, "Peringatan",
                f"File {label_mode} belum tersedia. Jalankan proses MFCC dulu!"
            )
            return

        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))
        self.player.play()
        self.lbl_audio_status.setText(
            f"Memutar: {label_mode} ({os.path.basename(file_path)})"
        )

    def stop_audio(self):
        self.player.stop()
        self.player.setSource(QUrl())

    def populate_table(self, table_widget, dataframe):
        table_widget.setRowCount(len(dataframe))
        table_widget.setColumnCount(len(dataframe.columns))
        table_widget.setHorizontalHeaderLabels(dataframe.columns.tolist())
        for row in range(len(dataframe)):
            for column in range(len(dataframe.columns)):
                table_widget.setItem(
                    row, column,
                    QTableWidgetItem(str(dataframe.iat[row, column]))
                )

    def run_text_preview(self):
        """Proses dataset CSV yang sedang dipilih, tanpa menimpa hasil dengan data dummy."""
        if not self.current_csv_path or not os.path.isfile(self.current_csv_path):
            QMessageBox.warning(self, "Peringatan", "Pilih file CSV terlebih dahulu!")
            return

        self.clear_text_results()

        try:
            selected_file = Path(self.current_csv_path).name
            language = self.combo_language.currentData() or "english"

            self.lbl_text_status.setText(f"Memproses dataset: {selected_file} ...")
            QApplication.processEvents()

            result = analyze_text_csv(
                self.current_csv_path,
                document_index=self.spin_doc_idx.value(),
                language=language,
                max_features=30
            )

            self.current_text_export_df = result["export_dataframe"]
            self.btn_download_text.setEnabled(True)

            selected = result["selected"]
            self.txt_raw[1].setText(selected["raw"])
            self.txt_case[1].setText(selected["case_folding"])
            self.txt_clean[1].setText(selected["cleaning"])
            self.txt_token[1].setText(str(selected["tokens"]))
            self.txt_lemma[1].setText(
                "Setelah stopword removal: " + str(selected["without_stopwords"]) +
                "\n\nHasil akhir (" + result["method_name"] + "):\n" +
                str(selected["final_tokens"])
            )

            tfidf_display = result["tfidf_matrix"].reset_index()
            self.populate_table(self.table_tfidf, tfidf_display)

            self.canvas_tfidf.axes.clear()
            self.canvas_tfidf.axes.set_facecolor("#0D0D0D")
            top_terms = result["top_terms"][:15]
            if top_terms:
                terms = [item[0] for item in top_terms][::-1]
                scores = [item[1] for item in top_terms][::-1]
                self.canvas_tfidf.axes.barh(terms, scores, color="#EAFF00")
                self.canvas_tfidf.axes.set_title(
                    f"Top TF-IDF Terms - Dokumen {result['selected_index'] + 1}"
                )
                self.canvas_tfidf.axes.set_xlabel("Nilai TF-IDF")
            else:
                self.canvas_tfidf.axes.text(
                    0.5, 0.5, "Tidak ada term yang dapat dihitung",
                    ha="center", va="center", color="#FFFFFF"
                )

            self.canvas_tfidf.fig.tight_layout()
            self.canvas_tfidf.draw()

            self.lbl_text_status.setText(
                f"Berhasil memproses {selected_file}! "
                f"{result['document_count']} dokumen | "
                f"Kolom: {result['text_column']} | "
                f"{result['feature_count']} fitur TF-IDF."
            )

        except Exception as error:
            self.clear_text_results()
            QMessageBox.critical(
                self, "Gagal Memproses Data Teks",
                f"Terjadi kesalahan saat preprocessing/TF-IDF:\n{error}"
            )
            self.lbl_text_status.setText("Status: Pemrosesan gagal.")

    def run_audio_preview(self):
        if not self.current_audio_path or not os.path.isfile(self.current_audio_path):
            QMessageBox.warning(self, "Peringatan", "Tidak ada file audio yang dipilih!")
            return

        try:
            self.stop_audio()
            self.lbl_audio_status.setText("Status: Memproses NoiseReduce & MFCC...")
            QApplication.processEvents()

            df_mfcc, meta = analyze_audio_mfcc(
                audio_path=self.current_audio_path,
                fig_mfcc=self.canvas_mfcc.fig,
                n_mfcc=13
            )

            self.audio_before_path = meta["before_path"]
            self.audio_after_path = meta["after_path"]
            self.btn_play_before.setEnabled(True)
            self.btn_play_after.setEnabled(True)

            self.canvas_mfcc.draw()
            self.populate_table(self.table_mfcc, df_mfcc)

            self.lbl_audio_status.setText(
                f"Selesai! {os.path.basename(self.current_audio_path)} | "
                f"Durasi: {meta['durasi']}s | SR: {meta['sample_rate']} Hz | "
                f"Matrix: {meta['shape_mfcc']}"
            )
        except Exception as error:
            QMessageBox.critical(
                self, "Error Pemrosesan Audio",
                f"Terjadi kesalahan:\n{error}"
            )
            self.lbl_audio_status.setText("Status: Gagal memproses audio.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureExtractionApp()
    window.show()
    sys.exit(app.exec())