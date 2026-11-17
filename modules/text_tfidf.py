"""
Modul preprocessing dan ekstraksi fitur teks (TF-IDF).
Mendukung teks Bahasa Indonesia dan Bahasa Inggris.

Fungsi utama:
    analyze_text_csv(csv_path, document_index=0, language="english", max_features=30)

Hasilnya dipakai oleh antarmuka PyQt6 pada app.py.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "pada", "adalah", "atau", "juga", "karena", "dalam", "tidak", "bukan",
    "saya", "aku", "kami", "kita", "anda", "kamu", "dia", "mereka",
    "sebagai", "agar", "dapat", "bisa", "akan", "sudah", "telah", "belum",
    "lebih", "sangat", "sekali", "hanya", "oleh", "sebuah", "suatu",
    "ada", "jadi", "nya", "pun", "saat", "serta", "tetapi", "namun",
    "kalau", "jika", "maka", "ketika", "hingga", "sampai",
    "antara", "setiap", "beberapa", "para", "banyak", "masih", "lagi",
    "the", "and", "is", "are", "was", "were", "a", "an", "of", "to",
    "in", "on", "for", "with", "this", "that", "it", "as", "at", "by",
}

LANGUAGE_LABELS = {
    "indonesian": "Bahasa Indonesia",
    "english": "Bahasa Inggris",
}


def _clean_text(text):
    """Menghapus URL, mention, angka, tanda baca, dan spasi berlebih."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lemmatize_english(token):
    """Lemmatization NLTK; jika WordNet belum tersedia, token tetap dipakai."""
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(token)
    except LookupError:
        return token
    except ImportError:
        return token


def _stem_indonesian(token, stemmer=None):
    if stemmer is not None:
        return stemmer.stem(token)
    return token


def preprocess_text(text, language="english"):
    """
    Menghasilkan tahapan preprocessing:
    raw -> case_folding -> cleaning -> tokenizing -> stopword_removal
    -> lemmatization/stemming.
    """
    raw = "" if pd.isna(text) else str(text)
    case_folding = raw.lower()

    cleaned = _clean_text(case_folding)
    tokens = cleaned.split()

    language = language.lower()
    if language == "indonesian":
        stopwords = INDONESIAN_STOPWORDS
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            stemmer = StemmerFactory().create_stemmer()
        except ImportError:
            stemmer = None
        filtered = [word for word in tokens if word not in stopwords]
        final_tokens = [_stem_indonesian(word, stemmer) for word in filtered]
        method_name = "Stemming (Sastrawi)" if stemmer else "Normalisasi (Sastrawi belum terpasang)"
    else:
        stopwords = set(ENGLISH_STOP_WORDS) | INDONESIAN_STOPWORDS
        filtered = [word for word in tokens if word not in stopwords]
        final_tokens = [_lemmatize_english(word) for word in filtered]
        method_name = "Lemmatization (NLTK WordNet; fallback jika corpus belum tersedia)"

    final_tokens = [word for word in final_tokens if word]
    return {
        "raw": raw,
        "case_folding": case_folding,
        "cleaning": cleaned,
        "tokens": tokens,
        "without_stopwords": filtered,
        "final_tokens": final_tokens,
        "processed": " ".join(final_tokens),
        "method_name": method_name,
    }


def _choose_text_column(dataframe):
    """Memilih kolom teks secara otomatis berdasarkan nama dan isi kolom."""
    if dataframe.empty:
        raise ValueError("Dataset CSV tidak memiliki baris data.")

    candidates = []
    preferred_words = (
        "text", "teks", "review", "ulasan", "komentar",
        "comment", "content", "sentence", "kalimat", "tweet",
        "description", "deskripsi", "judul", "berita", "name",
        "nama", "team", "player", "tournament"
    )
    id_words = {"id", "idx", "index", "kode", "code", "nomor", "number", "key"}
    for column in dataframe.columns:
        series = dataframe[column].dropna().astype(str)
        if series.empty:
            continue

        letter_lengths = series.str.count(r"[A-Za-zÀ-ÿ]")
        if letter_lengths.mean() < 1:
            continue

        name = str(column).lower()
        name_tokens = set(re.split(r"[^a-z0-9]+", name))
        name_bonus = 1000 if any(word.strip() in name for word in preferred_words) else 0
        id_penalty = -1000 if name_tokens & id_words else 0
        score = name_bonus + id_penalty + min(letter_lengths.mean(), 500)
        candidates.append((score, column))

    if not candidates:
        raise ValueError("Tidak ditemukan kolom yang berisi teks.")
    return max(candidates, key=lambda item: item[0])[1]


def analyze_text_csv(csv_path, document_index=0, language="english", max_features=30):
    """
    Membaca CSV, melakukan preprocessing seluruh dokumen, lalu menghitung TF-IDF.
    Satu baris CSV dianggap sebagai satu dokumen.
    """
    csv_path = str(Path(csv_path))
    try:
        dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        dataframe = pd.read_csv(csv_path, encoding="latin-1")

    if dataframe.empty:
        raise ValueError("Dataset CSV kosong.")
    if len(dataframe.columns) == 0:
        raise ValueError("Dataset tidak memiliki kolom.")

    text_column = _choose_text_column(dataframe)
    documents = dataframe[text_column].fillna("").astype(str).tolist()

    if not any(document.strip() for document in documents):
        raise ValueError(f"Kolom '{text_column}' tidak memiliki teks yang dapat diproses.")

    processed_rows = [preprocess_text(document, language) for document in documents]
    processed_documents = [row["processed"] for row in processed_rows]

    valid_indices = [i for i, doc in enumerate(processed_documents) if doc.strip()]
    if not valid_indices:
        raise ValueError(
            "Semua teks menjadi kosong setelah preprocessing. "
            "Coba pilih bahasa yang sesuai atau periksa isi dataset."
        )

    valid_documents = [processed_documents[i] for i in valid_indices]
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(valid_documents)

    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = matrix.toarray()

    # Matriks TF-IDF untuk tampilan tabel UI
    tfidf_dataframe = pd.DataFrame(
        dense_matrix,
        columns=feature_names,
        index=[f"Dokumen {i + 1}" for i in valid_indices],
    )
    tfidf_dataframe.index.name = "Dokumen"

    # Gabungkan seluruh tahapan preprocessing + skor TF-IDF untuk fitur Download CSV
    full_tfidf = np.zeros((len(documents), len(feature_names)), dtype=float)
    full_tfidf[valid_indices] = np.round(dense_matrix, 6)

    prep_df = pd.DataFrame({
        "Dokumen": [f"Dokumen {i + 1}" for i in range(len(documents))],
        "Raw_Text": [row["raw"] for row in processed_rows],
        "Case_Folding": [row["case_folding"] for row in processed_rows],
        "Cleaning": [row["cleaning"] for row in processed_rows],
        "Tokenizing": [", ".join(row["tokens"]) for row in processed_rows],
        "Stopword_Removal": [", ".join(row["without_stopwords"]) for row in processed_rows],
        "Hasil_Preprocessing": [row["processed"] for row in processed_rows],
    })
    tfidf_export_cols = pd.DataFrame(
        full_tfidf,
        columns=[f"TFIDF_{name}" for name in feature_names]
    )
    export_dataframe = pd.concat([prep_df, tfidf_export_cols], axis=1)

    selected_index = max(0, min(int(document_index), len(documents) - 1))
    selected = processed_rows[selected_index]

    if selected_index in valid_indices:
        row_position = valid_indices.index(selected_index)
        scores = dense_matrix[row_position]
        top_order = scores.argsort()[::-1]
        top_terms = [
            (str(feature_names[i]), float(scores[i]))
            for i in top_order
            if scores[i] > 0
        ]
    else:
        top_terms = []

    mean_scores = dense_matrix.mean(axis=0)
    overall_order = mean_scores.argsort()[::-1]
    overall_terms = [
        (str(feature_names[i]), float(mean_scores[i]))
        for i in overall_order
    ]

    return {
        "dataframe": dataframe,
        "text_column": str(text_column),
        "document_count": len(documents),
        "selected_index": selected_index,
        "selected": selected,
        "processed_documents": processed_documents,
        "tfidf_matrix": tfidf_dataframe,
        "export_dataframe": export_dataframe,
        "top_terms": top_terms,
        "overall_terms": overall_terms,
        "feature_count": len(feature_names),
        "method_name": selected["method_name"],
    }