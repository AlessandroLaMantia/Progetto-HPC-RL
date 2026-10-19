"""Convalida uno stato target arbitrario."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


# Rimuove eventuali commenti iniziali inseriti nei file JSON dei target.
def _read_json_target(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("//")]
    return json.loads("\n".join(lines))


# Convalida uno stato quantistico target arbitrario e restituisce il vettore normalizzato e il numero di qubit.
def validate_target_state(state: np.ndarray, n_qubits: int | None = None) -> tuple[np.ndarray, int]:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    # Controlla che lo stato abbia un numero valido di ampiezze.
    if vector.size == 0 or vector.size & (vector.size - 1):
        raise ValueError("A target state must contain 2**n complex amplitudes")
    inferred_qubits = vector.size.bit_length() - 1
    # Controlla che il numero di qubit dichiarato corrisponda alla dimensione dello stato.
    if n_qubits is not None and n_qubits != inferred_qubits:
        raise ValueError(f"Target has {vector.size} amplitudes ({inferred_qubits} qubits), not {n_qubits} qubits")
    norm = np.linalg.norm(vector)
    # Controlla che lo stato abbia una norma valida e diversa da zero.
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("The target state must have finite, non-zero norm")
    return vector / norm, inferred_qubits


# Converte una riga di testo in uno stato target, usando il formato richiesto dall'utente.
def load_target_state_from_text(text: str, format: str = "space", n_qubits: int | None = None) -> tuple[np.ndarray, int]:
    if not isinstance(text, str):
        raise TypeError("The target state must be provided as a text string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("The text for the target state cannot be empty")
    fmt = format.lower()
    if fmt in {"space", "whitespace"}:
        tokens = cleaned.replace(",", " ").split()
    elif fmt in {"comma", "csv", "list"}:
        inner = cleaned.strip("[]()")
        tokens = [token.strip() for token in inner.split(",") if token.strip()]
    else:
        raise ValueError("Unsupported text format; choose 'space', 'comma', or 'list'")
    if not tokens:
        raise ValueError("No amplitudes were found in the text")
    try:
        amplitudes = [complex(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("The text must contain numeric amplitudes") from exc
    return validate_target_state(np.asarray(amplitudes, dtype=np.complex128), n_qubits)


# Carica uno stato target arbitrario da un file ``.npy`` o JSON.
def load_target_state(path: str | Path, n_qubits: int | None = None) -> tuple[np.ndarray, int]:
    # Converte il percorso in un oggetto Path per gestirlo in modo semplice.
    file_path = Path(path)
    # Controlla che il file esista prima di tentare di leggerlo.
    if not file_path.is_file():
        raise FileNotFoundError(f"Target file not found: {file_path}")
    # Legge lo stato da un file NumPy, evitando pickle per sicurezza.
    if file_path.suffix.lower() == ".npy":
        raw = np.load(file_path, allow_pickle=False)  # Niente pickle: file target sicuro e solo numerico.
    # Interpreta un file JSON come una sequenza di ampiezze complesse, se sono salvate come coppie [re, im].
    elif file_path.suffix.lower() == ".json":
        raw = _read_json_target(file_path)
        if raw and isinstance(raw[0], list):
            # Ogni elemento deve essere una coppia [reale, immaginario].
            if any(len(pair) != 2 for pair in raw):
                raise ValueError("Complex JSON amplitudes must be [real, imag] pairs")
            raw = [complex(real, imag) for real, imag in raw]  # Converte [reale, immaginaria] in complessi Python.
    # Blocca formati non supportati.
    else:
        raise ValueError("Target file must have .npy or .json extension")
    # Valida e normalizza lo stato prima di restituirlo.
    return validate_target_state(np.asarray(raw), n_qubits)
