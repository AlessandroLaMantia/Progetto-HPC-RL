"""Genera una baseline di circuiti casuali per uno stato target."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


# Gate usati dalla baseline; il CNOT viene escluso automaticamente per un solo qubit.
AVAILABLE_GATES = ("h", "x", "y", "z", "cx")
PROJECT_ROOT = Path(__file__).resolve().parent


# Risolve il percorso indicato, cercando anche nella cartella targets del progetto.
def resolve_target_path(target_file: str) -> Path:
    path = Path(target_file).expanduser()
    if path.is_file():
        return path.resolve()
    target_path = PROJECT_ROOT / "targets" / path.name
    if target_path.is_file():
        return target_path.resolve()
    raise FileNotFoundError(f"File target non trovato: {target_file}")


# Legge un target JSON o NPY, controlla le ampiezze e normalizza lo stato.
def load_target_state(target_path: Path) -> np.ndarray:
    # I file JSON possono contenere numeri reali o coppie [parte reale, parte immaginaria].
    if target_path.suffix.lower() == ".npy":
        state = np.asarray(np.load(target_path, allow_pickle=False), dtype=np.complex128).reshape(-1)
    elif target_path.suffix.lower() == ".json":
        lines = [
            line for line in target_path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("//")
        ]
        raw = json.loads("\n".join(lines))
        if raw and isinstance(raw[0], list):
            if any(not isinstance(amplitude, list) or len(amplitude) != 2 for amplitude in raw):
                raise ValueError("Le ampiezze complesse JSON devono essere coppie [reale, immaginaria].")
            state = np.asarray([complex(real, imaginary) for real, imaginary in raw], dtype=np.complex128)
        else:
            state = np.asarray(raw, dtype=np.complex128).reshape(-1)
    else:
        raise ValueError("Il target deve essere un file .json oppure .npy.")

    # La dimensione deve essere una potenza di due e rappresentare almeno un qubit.
    if state.size < 2 or state.size & (state.size - 1):
        raise ValueError("Il target deve contenere 2**n ampiezze per almeno un qubit.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Le ampiezze del target devono essere finite.")
    norm = np.linalg.norm(state)
    # Una norma infinita può derivare da ampiezze finite ma troppo grandi per il calcolo numerico.
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Il target deve avere una norma finita e diversa da zero.")
    return state / norm


def run_random_circuit(
    target_state: Statevector,
    n_qubits: int,
    max_gates: int,
    fidelity_threshold: float,
) -> tuple[bool, float, int, float, str]:
    # Aggiunge gate casuali e verifica la fidelity dopo ogni operazione.
    circuit = QuantumCircuit(n_qubits)
    start_time = time.perf_counter()
    fidelity = float(state_fidelity(Statevector.from_instruction(circuit), target_state))
    gate_sequence: list[str] = []

    if fidelity >= fidelity_threshold:
        elapsed_time = time.perf_counter() - start_time
        return True, fidelity, 0, elapsed_time, "I (nessun gate)"

    for _ in range(max_gates):
        # Sceglie un gate e i qubit su cui applicarlo.
        available_gates = AVAILABLE_GATES if n_qubits >= 2 else AVAILABLE_GATES[:-1]
        gate = str(np.random.choice(available_gates))
        if gate == "cx":
            control, target = np.random.choice(n_qubits, size=2, replace=False)
            circuit.cx(int(control), int(target))
            gate_description = f"CX({int(control)}, {int(target)})"
        else:
            qubit = int(np.random.randint(n_qubits))
            getattr(circuit, gate)(qubit)
            gate_description = f"{gate.upper()}({qubit})"
        gate_sequence.append(gate_description)
        # Simula il circuito aggiornato e interrompe la run se raggiunge la soglia.
        current_state = Statevector.from_instruction(circuit)
        fidelity = float(state_fidelity(current_state, target_state))
        if fidelity >= fidelity_threshold:
            break

    elapsed_time = time.perf_counter() - start_time
    circuit_description = " -> ".join(gate_sequence)
    return fidelity >= fidelity_threshold, fidelity, len(gate_sequence), elapsed_time, circuit_description


def main():
    # Legge dal terminale target, numero di run, soglia e percorso del CSV.
    parser = argparse.ArgumentParser(
        description="Genera una baseline di circuiti casuali rispetto allo stato target fornito."
    )
    # Parametri principali
    parser.add_argument("--target-file", type=str, required=True, help="File target .json o .npy")
    parser.add_argument("--runs", type=int, default=5, help="Numero di esecuzioni casuali")
    parser.add_argument("--output", type=str, default="baseline_summary.csv", help="File CSV di output")
    parser.add_argument("--max-gates", type=int, default=25, help="Numero massimo di gate per circuito")
    parser.add_argument("--fidelity-threshold", type=float, default=0.99, help="Soglia di fidelity per il successo")

    # Parametri addizionali per evitare incompatibilità se passati da script esterni
    parser.add_argument("--success-rate", type=float, default=None, help="Inutilizzato - mantenuto per compatibilità")
    parser.add_argument("--final-fidelity-base", type=float, default=None, help="Inutilizzato - mantenuto per compatibilità")
    parser.add_argument("--gate-count-base", type=float, default=None, help="Inutilizzato - mantenuto per compatibilità")
    parser.add_argument("--total-reward-base", type=float, default=None, help="Inutilizzato - mantenuto per compatibilità")
    parser.add_argument("--time-base", type=float, default=None, help="Inutilizzato - mantenuto per compatibilità")

    args = parser.parse_args()

    # Blocca parametri che renderebbero la simulazione o il criterio di successo invalidi.
    if args.runs < 1:
        parser.error("--runs deve essere almeno 1")
    if args.max_gates < 1:
        parser.error("--max-gates deve essere almeno 1")
    if not 0.0 <= args.fidelity_threshold <= 1.0:
        parser.error("--fidelity-threshold deve essere compresa tra 0 e 1")

    # Carica il target e ricava da esso il numero di qubit necessario.
    target_path = resolve_target_path(args.target_file)
    target_vector = load_target_state(target_path)
    n_qubits = target_vector.size.bit_length() - 1
    target_state = Statevector(target_vector)

    # Queste liste servono a costruire la riga media aggiunta in fondo al CSV.
    rows = []
    successes = []
    fidelities = []
    gate_counts = []
    elapsed_times = []

    print(
        f"Target: {target_path.name} ({n_qubits} qubit); "
        f"{args.runs} simulazioni casuali, massimo {args.max_gates} gate per run."
    )

    for run_index in range(1, args.runs + 1):
        success, fidelity, gates, elapsed, best_circuit = run_random_circuit(
            target_state=target_state,
            n_qubits=n_qubits,
            max_gates=args.max_gates,
            fidelity_threshold=args.fidelity_threshold,
        )
        # La reward premia la fidelity e penalizza i circuiti più lunghi.
        reward = fidelity * 10.0 - gates * 0.1
        success_value = int(success)
        rows.append({
            "training_i": run_index,
            "success": success_value,
            "final_fidelity": round(fidelity, 4),
            "gate_count": gates,
            "total_reward": round(reward, 4),
            "best_circuit_found": best_circuit,
            "training_time_seconds": round(elapsed, 4),
            "phase": "before_training",
        })
        successes.append(success_value)
        fidelities.append(fidelity)
        gate_counts.append(gates)
        elapsed_times.append(elapsed)

    # Riassume le run con una riga media, utile per il confronto con il training.
    mean_row = {
        "training_i": "mean",
        "success": round(float(np.mean(successes)), 4),
        "final_fidelity": round(float(np.mean(fidelities)), 4),
        "gate_count": round(float(np.mean(gate_counts)), 2),
        "total_reward": round(float(np.mean([
            fidelity * 10.0 - gates * 0.1 for fidelity, gates in zip(fidelities, gate_counts)
        ])), 4),
        "best_circuit_found": "-",
        "training_time_seconds": round(float(np.mean(elapsed_times)), 4),
        "phase": "before_training",
    }
    rows.append(mean_row)

    # Salva risultati e medie nel formato usato dagli script di analisi.
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "training_i", "success", "final_fidelity", "gate_count",
            "total_reward", "best_circuit_found", "training_time_seconds", "phase",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Baseline calcolata e salvata in: {out_path}")


if __name__ == "__main__":
    main()