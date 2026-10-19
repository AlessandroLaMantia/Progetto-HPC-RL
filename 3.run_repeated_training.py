from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "2.train.py"
EVAL_SCRIPT = ROOT / "5.evaluate.py"

# Estrai il circuito migliore dall'output di valutazione, se presente.
def parse_best_circuit(stdout: str) -> str:
    match = re.search(r"Best circuit:\s*(.+)", stdout)
    if not match:
        return "-"
    return match.group(1).strip()

# Analizza il file di report JSON e restituisce un dizionario con i valori numerici.
def parse_report(report_path: Path) -> dict:
    with report_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "mean_final_fidelity": float(payload["mean_final_fidelity"]),
        "best_final_fidelity": float(payload["best_final_fidelity"]),
        "mean_episode_reward": float(payload["mean_episode_reward"]),
        "mean_circuit_length": float(payload["mean_circuit_length"]),
        "success_rate_percent": float(payload["success_rate_percent"]),
    }

# Analizza l'output della valutazione per estrarre il circuito migliore, la fedeltà finale, la ricompensa media e la lunghezza media del circuito.
def parse_eval_stdout(stdout: str) -> tuple[str, float, float, float]:
    best_circuit = parse_best_circuit(stdout)
    best_match = re.search(r"Best final fidelity=([0-9.eE+-]+)", stdout)
    mean_reward_match = re.search(r"Mean episode reward=([0-9.eE+-]+)", stdout)
    mean_gate_length_match = re.search(r"Mean circuit length=([0-9.eE+-]+)", stdout)
    best_fid = float(best_match.group(1)) if best_match else float("nan")
    mean_reward = float(mean_reward_match.group(1)) if mean_reward_match else float("nan")
    mean_gate_length = float(mean_gate_length_match.group(1)) if mean_gate_length_match else float("nan")
    return best_circuit, best_fid, mean_reward, mean_gate_length

# Esegue un singolo ciclo di addestramento e valutazione, restituendo un dizionario con i risultati.
def run_training_once(run_index: int, args: argparse.Namespace) -> dict:
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = ROOT / checkpoint_root
    run_dir = checkpoint_root / f"run_{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    train_cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--target", args.target,
        "--max-gates", str(args.max_gates),
        "--iterations", str(args.iterations),
        "--workers", str(args.workers),
        "--checkpoint-dir", str(run_dir),
        "--metrics-file", str(run_dir / f"training_history_run_{run_index}.csv"),
    ]
    if args.target == "custom" and not args.target_file:
        raise ValueError("--target-file is required when --target custom")
    if args.target_file:
        target_file = Path(args.target_file)
        if not target_file.is_file():
            target_file = ROOT / "targets" / target_file.name
        train_cmd.extend(["--target-file", str(target_file.resolve())])
    if args.qubits is not None:
        train_cmd.extend(["--qubits", str(args.qubits)])
    if args.target == "ghz" and args.qubits is None and not args.target_file:
        train_cmd.extend(["--qubits", "3"])

    subprocess.run(train_cmd, cwd=str(ROOT), check=True, text=True)

    checkpoint_path = run_dir / f"iter_{args.iterations}"
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint atteso non trovato: {checkpoint_path}")

    eval_cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(checkpoint_path),
        "--target", args.target,
        "--episodes", str(args.episodes),
        "--report", str(run_dir / "report.json"),
        "--max-gates", str(args.max_gates),
    ]
    if args.target_file:
        eval_cmd.extend(["--target-file", str(target_file.resolve())])
    if args.qubits is not None:
        eval_cmd.extend(["--qubits", str(args.qubits)])

    result = subprocess.run(eval_cmd, cwd=str(ROOT), check=True, text=True, capture_output=True)
    stdout = (result.stdout or "") + (result.stderr or "")
    report = parse_report(run_dir / "report.json")
    best_circuit, best_fid, mean_reward, mean_gate_length = parse_eval_stdout(stdout)

    gate_count = len([g for g in best_circuit.split(" -> ") if g.strip()]) if best_circuit != "-" else 0
    success = 1 if best_fid >= 0.99 else 0
    elapsed = time.perf_counter() - start_time

    return {
        "training_i": run_index,
        "success": success,
        "final_fidelity": best_fid,
        "gate_count": gate_count,
        "total_reward": mean_reward,
        "best_circuit_found": best_circuit,
        "training_time_seconds": elapsed,
    }

# Somma i risultati di più run di addestramento e restituisce un dizionario con le medie.
def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "training_i": "mean",
            "success": 0.0,
            "final_fidelity": 0.0,
            "gate_count": 0.0,
            "total_reward": 0.0,
            "best_circuit_found": "-",
            "training_time_seconds": 0.0,
        }
    return {
        "training_i": "mean",
        "success": sum(r["success"] for r in rows) / len(rows),
        "final_fidelity": sum(r["final_fidelity"] for r in rows) / len(rows),
        "gate_count": sum(r["gate_count"] for r in rows) / len(rows),
        "total_reward": sum(r["total_reward"] for r in rows) / len(rows),
        "best_circuit_found": "-",
        "training_time_seconds": sum(r["training_time_seconds"] for r in rows) / len(rows),
    }

# Funzione principale per eseguire più addestramenti ripetuti e raccogliere un riepilogo in formato CSV.
def main():
    parser = argparse.ArgumentParser(description="Run the same training repeatedly on the same target and collect a summary table.")
    parser.add_argument("--target", choices=["bell", "ghz", "custom"], default="ghz")
    parser.add_argument("--target-file", type=str, default=None, help="Target file for custom states")
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--max-gates", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated trainings to run")
    parser.add_argument("--output", type=str, default="training_summary.csv", help="CSV summary file to write")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints", help="Directory for per-run checkpoints and reports")
    args = parser.parse_args()

    rows: list[dict] = []
    for i in range(1, args.runs + 1):
        print(f"\n=== training run {i}/{args.runs} ===")
        row = run_training_once(i, args)
        rows.append(row)
        print(row)

    summary = summarize_rows(rows)
    rows.append(summary)

    out_path = ROOT / args.output
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["training_i", "success", "final_fidelity", "gate_count", "total_reward", "best_circuit_found", "training_time_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
