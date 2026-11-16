import os
import sys
import shutil
from modules.text_tfidf import analyze_text_csv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit, QTabWidget,
    QFrame, QFileDialog, QTableWidget, QTableWidgetItem, QScrollArea,
    QHeaderView, QMessageBox, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from modules.audio_mfcc import analyze_audio_mfcc


def resource_path(relative_path):
    """Mendapatkan path absolut untuk mode script maupun PyInstaller .exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def apply_glass_shadow(widget):
    """Memberikan efek bayangan lembut agar kartu kaca tampak melayang"""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(28)
    shadow.setXOffset(0)
    shadow.setYOffset(8)
    shadow.setColor(QColor(15, 23, 42, 22))
    widget.setGraphicsEffect(shadow)


class MplCanvas(FigureCanvas):
    """Kanvas Matplotlib dengan warna dasar lembut menyatu dengan panel kaca"""
    def __init__(self, width=6, height=3.2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#F6F9FE")
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("#FFFFFF")
        super().__init__(self.fig)


class FeatureExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UTS Pengolahan Data Tidak Terstruktur • Glass Analyzer")
        self.resize(1360, 840)

        os.makedirs("datasets/text", exist_ok=True)
        os.makedirs("datasets/audio", exist_ok=True)

        self.current_csv_path = None
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
        qss_path = resource_path(os.path.join("assets", "style.qss"))
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(22, 18, 22, 18)
        main_layout.setSpacing(14)

        # ================= HEADER GLASS =================
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        apply_glass_shadow(header_card)
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(24, 16, 24, 16)
        header_layout.setSpacing(6)

        badge_row = QHBoxLayout()
        badge = QLabel("UTS Pengolahan Data Tidak Terstruktur • Glassmorphism UI")
        badge.setObjectName("HeaderBadge")
        badge_row.addWidget(badge)
        badge_row.addStretch()

        title = QLabel("Ekstraksi Fitur Data Teks (TF-IDF) & Data Audio (MFCC)")
        title.setObjectName("HeaderTitle")

        subtitle = QLabel(
            "Eksplorasi tahapan preprocessing (NLTK & NoiseReduce) serta ekstraksi fitur "
            "berbasis aplikasi desktop."
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

        left_card = QFrame()
        left_card.setObjectName("PanelCard")
        left_card.setFixedWidth(360)
        apply_glass_shadow(left_card)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)

        lbl_title = QLabel("Konfigurasi Dataset Teks")
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
        left_layout.addWidget(lbl_language)
        left_layout.addWidget(self.combo_language)
        left_layout.addSpacing(8)
        left_layout.addWidget(btn_run_text)
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

    # ---------------------------------------------------------
    # TAB 2: DATA AUDIO (MFCC) + PEMUTAR BEFORE (ASLI) & AFTER (CLEAN)
    # ---------------------------------------------------------
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

        lbl_title = QLabel("Konfigurasi Dataset Audio")
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

        btn_run_audio = QPushButton("Jalankan Preprocessing & MFCC")
        btn_run_audio.setObjectName("PrimaryButton")
        btn_run_audio.clicked.connect(self.run_audio_preview)

        # Pemutar Perbandingan Audio: Before (Asli) vs After (Noise Reduced)
        lbl_player = QLabel("Pemutar Audio (Before vs After):")
        lbl_player.setObjectName("FieldLabel")

        row_ba = QHBoxLayout()
        self.btn_play_before = QPushButton("▶ Before (Asli)")
        self.btn_play_before.setObjectName("SecondaryButton")
        self.btn_play_before.setEnabled(False)
        self.btn_play_before.clicked.connect(lambda: self.play_audio_file(self.audio_before_path, "Before (Audio Asli)"))

        self.btn_play_after = QPushButton("▶ After (Clean)")
        self.btn_play_after.setObjectName("SecondaryButton")
        self.btn_play_after.setEnabled(False)
        self.btn_play_after.clicked.connect(lambda: self.play_audio_file(self.audio_after_path, "After (Noise Reduced)"))

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

        sec_mfcc = QLabel("Visualisasi Ekstraksi Fitur (MFCC Heatmap & Statistik 13 Koefisien)")
        sec_mfcc.setObjectName("SectionTitle")
        right_layout.addWidget(sec_mfcc)

        self.canvas_mfcc = MplCanvas(width=7, height=3.8)
        self.canvas_mfcc.setMinimumHeight(320)
        right_layout.addWidget(self.canvas_mfcc)

        self.table_mfcc = QTableWidget()
        self.table_mfcc.setMinimumHeight(240)
        self.table_mfcc.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.table_mfcc)

        scroll.setWidget(right_card)

        layout.addWidget(left_card)
        layout.addWidget(scroll)
        return tab

    # ---------------------------------------------------------
    # EVENT HANDLER & PEMBANTU UI
    # ---------------------------------------------------------
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

    def _clear_text_results(self):
        if hasattr(self, "txt_raw"):
            for _, box in [self.txt_raw, self.txt_case, self.txt_clean, self.txt_token, self.txt_lemma]:
                box.clear()
        if hasattr(self, "table_tfidf"):
            self.table_tfidf.clearContents()
            self.table_tfidf.setRowCount(0)
            self.table_tfidf.setColumnCount(0)
        if hasattr(self, "canvas_tfidf"):
            self.canvas_tfidf.axes.clear()
            self.canvas_tfidf.draw()

    def on_select_local_csv(self, filename):
        if filename:
            self.current_csv_path = os.path.join("datasets", "text", filename)
            # Reset supaya hasil dataset sebelumnya tidak tertinggal di layar.
            if hasattr(self, "spin_doc_idx"):
                self.spin_doc_idx.setValue(0)
            self._clear_text_results()
            if hasattr(self, "lbl_text_status"):
                self.lbl_text_status.setText(
                    f"Status: Dataset dipilih -> {filename}. Klik proses untuk memperbarui hasil."
                )

    def browse_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Dataset CSV", "", "CSV Files (*.csv)")
        if path:
            filename = os.path.basename(path)
            dest_path = os.path.join("datasets", "text", filename)
            if os.path.abspath(path) != os.path.abspath(dest_path):
                shutil.copy2(path, dest_path)
            self.refresh_local_files()
            self.combo_csv.setCurrentText(filename)
            self.current_csv_path = dest_path
            self.lbl_text_status.setText(f"Tersimpan di datasets/text/: {filename}")

    def on_select_local_wav(self, filename):
        if filename:
            self.stop_audio()
            self.current_audio_path = os.path.join("datasets", "audio", filename)
            self.audio_before_path = os.path.abspath(self.current_audio_path)
            self.audio_after_path = None
            if hasattr(self, "btn_play_before"):
                self.btn_play_before.setEnabled(True)
                self.btn_play_after.setEnabled(False)

    def browse_wav_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Dataset Audio", "", "Audio Files (*.wav *.mp3)")
        if path:
            self.stop_audio()
            filename = os.path.basename(path)
            dest_path = os.path.join("datasets", "audio", filename)
            if os.path.abspath(path) != os.path.abspath(dest_path):
                shutil.copy2(path, dest_path)
            self.refresh_local_files()
            self.combo_wav.setCurrentText(filename)
            self.current_audio_path = dest_path
            self.audio_before_path = os.path.abspath(dest_path)
            self.btn_play_before.setEnabled(True)
            self.btn_play_after.setEnabled(False)
            self.lbl_audio_status.setText(f"Tersimpan di datasets/audio/: {filename}")

    def play_audio_file(self, file_path, label_mode):
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Peringatan", f"File {label_mode} belum tersedia. Jalankan proses MFCC dulu!")
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))
        self.player.play()
        self.lbl_audio_status.setText(f"Memutar: {label_mode} ({os.path.basename(file_path)})")

    def stop_audio(self):
        self.player.stop()
        self.player.setSource(QUrl())

    def populate_table(self, table_widget, df):
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        for r in range(len(df)):
            for c in range(len(df.columns)):
                table_widget.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))

    def run_text_preview(self):
        if not self.current_csv_path or not os.path.exists(self.current_csv_path):
            QMessageBox.warning(self, "Peringatan", "Pilih file CSV terlebih dahulu!")
            return

        try:
            language = self.combo_language.currentData() if hasattr(self, "combo_language") else "english"
            self._clear_text_results()
            self.lbl_text_status.setText("Status: Memproses preprocessing & TF-IDF...")
            QApplication.processEvents()
            result = analyze_text_csv(
                self.current_csv_path,
                document_index=self.spin_doc_idx.value(),
                language=language,
                max_features=30
            )

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

            # Tampilkan matriks TF-IDF seluruh dokumen.
            tfidf_display = result["tfidf_matrix"].reset_index()
            self.populate_table(self.table_tfidf, tfidf_display)

            # Grafik term dengan skor tertinggi pada dokumen yang dipilih.
            self.canvas_tfidf.axes.clear()
            top_terms = result["top_terms"][:15]
            if top_terms:
                terms = [item[0] for item in top_terms][::-1]
                scores = [item[1] for item in top_terms][::-1]
                self.canvas_tfidf.axes.barh(terms, scores, color="#2563EB")
                self.canvas_tfidf.axes.set_title(
                    f"Top TF-IDF Terms - Dokumen {result['selected_index'] + 1}"
                )
                self.canvas_tfidf.axes.set_xlabel("Nilai TF-IDF")
            else:
                self.canvas_tfidf.axes.text(
                    0.5, 0.5, "Tidak ada term yang dapat dihitung",
                    ha="center", va="center"
                )
            self.canvas_tfidf.fig.tight_layout()
            self.canvas_tfidf.draw()

            self.lbl_text_status.setText(
                f"Berhasil! {result['document_count']} dokumen | "
                f"Kolom: {result['text_column']} | "
                f"{result['feature_count']} fitur TF-IDF."
            )

        except Exception as error:
            QMessageBox.critical(
                self, "Gagal Memproses Data Teks",
                f"Terjadi kesalahan saat preprocessing/TF-IDF:\n{error}"
            )
            self._clear_text_results()
            self.lbl_text_status.setText("Status: Pemrosesan gagal.")

    def run_audio_preview(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
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
        except Exception as e:
            QMessageBox.critical(self, "Error Pemrosesan Audio", f"Terjadi kesalahan:\n{str(e)}")
            self.lbl_audio_status.setText("Status: Gagal memproses audio.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureExtractionApp()
    window.show()
    sys.exit(app.exec())