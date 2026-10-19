from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Carica il CSV e rimuove la riga di media se presente, così i grafici lavorano solo sui dati veri.
def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not df.empty and "training_i" in df.columns and df["training_i"].astype(str).str.lower().iloc[-1] == "mean":
        df = df.iloc[:-1].copy()
    return df


def load_training_history(path: Path) -> pd.DataFrame:
    """Legge reward e iterazioni di una singola run."""
    columns = ["iterazione", "reward medio degli episodi"]
    history = pd.read_csv(path)
    missing = set(columns) - set(history.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {path}: {sorted(missing)}")
    history = history[columns].apply(pd.to_numeric, errors="coerce")
    history = history.replace([float("inf"), -float("inf")], float("nan"))
    if history.empty or history["iterazione"].isna().any():
        raise ValueError("La cronologia deve contenere iterazioni valide.")
    if history["iterazione"].duplicated().any():
        raise ValueError("Usa la cronologia di una sola run: iterazioni duplicate.")
    if history["reward medio degli episodi"].isna().all():
        raise ValueError("Il CSV non contiene valori validi di reward media.")
    return history.sort_values("iterazione")


def plot_training_history(histories: dict[str, pd.DataFrame], out_dir: Path, failed_runs: list[int]) -> None:
    """Figura 9: reward media contro iterazione PPO, senza altre metriche."""
    plt.figure(figsize=(10, 6))
    for label, history in histories.items():
        plt.plot(history["iterazione"], history["reward medio degli episodi"],
                 marker="o", linewidth=1.5, label=label)
    plt.title("Reward media durante gli addestramenti")
    plt.legend(title="Run", ncol=2)
    plt.xlabel("Iterazione PPO")
    plt.ylabel("Reward media degli episodi")
    plt.grid(True, alpha=0.3)
    note = "Run senza successo finale (migliore fidelity < 0,99): " + (", ".join(map(str, failed_runs)) or "nessuna")
    plt.figtext(0.5, 0.02, note, ha="center", wrap=True, fontsize=9)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(out_dir / "figure_training_progress.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Genera solo reward e fidelity di tutte le run rispetto alla baseline.")
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True, help="CSV della baseline sullo stesso target")
    parser.add_argument("--history-root", type=Path, required=True, help="Cartella checkpoints del target")
    parser.add_argument("--target", choices=["bell", "ghz", "custom"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    summary = load_summary(args.after)
    baseline = load_summary(args.before)
    if summary.empty or baseline.empty:
        raise ValueError("I CSV devono contenere risultati")
    baseline_values = pd.to_numeric(baseline["final_fidelity"], errors="raise")
    if not baseline_values.map(lambda x: math.isfinite(x) and 0 <= x <= 1).all():
        raise ValueError("Fidelity della baseline non valide")
    histories = {}
    final_fidelities = {}
    failed_runs = []
    for _, row in summary.iterrows():
        number = float(row["training_i"])
        if not number.is_integer() or number < 1:
            raise ValueError(f"Run non valida: {number}")
        run = int(number)
        histories[str(run)] = load_training_history(
            args.history_root / f"run_{run}" / f"training_history_run_{run}.csv")
        fidelity = float(row["final_fidelity"])
        if not math.isfinite(fidelity) or not 0 <= fidelity <= 1:
            raise ValueError(f"Fidelity non valida per run {run}")
        if str(run) in final_fidelities:
            raise ValueError(f"Run duplicata: {run}")
        final_fidelities[str(run)] = fidelity
        success = float(row["success"])
        if success not in (0, 1):
            raise ValueError(f"Success non valido per run {run}: {success}")
        if success == 0:
            failed_runs.append(run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(histories, args.output_dir, failed_runs)
    fig, ax = plt.subplots(figsize=(10, 7))
    groups = [baseline_values.tolist(), list(final_fidelities.values())]
    gate_groups = []
    for frame in (baseline, summary):
        gates = pd.to_numeric(frame["gate_count"], errors="raise")
        if not gates.map(lambda value: math.isfinite(value) and value >= 0 and float(value).is_integer()).all():
            raise ValueError("Il numero di gate deve essere un intero non negativo")
        gate_groups.append(gates.astype(int).tolist())
    colors = ["#64748b", "#2563eb"]
    labels = ["Baseline casuale", "Agente addestrato"]
    for position, (values, color, label) in enumerate(zip(groups, colors, labels)):
        # Spaziatura deterministica: anche valori uguali restano visibili.
        offsets = [0.0] if len(values) == 1 else [
            -0.22 + 0.44 * i / (len(values) - 1) for i in range(len(values))]
        ax.scatter([position + offset for offset in offsets], values,
                   color=color, s=65, alpha=0.85, zorder=3)
        for offset, value, gate_count in zip(offsets, values, gate_groups[position]):
            ax.annotate(str(gate_count), (position + offset, value),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=color)
        mean = sum(values) / len(values)
        ax.hlines(mean, position - 0.3, position + 0.3, color=color,
                  linewidth=3, label=f"Media {label.lower()}: {mean:.4f}")
        successes = sum(value >= 0.99 for value in values)
        ax.text(position, 1.075, f"n={len(values)}; successi: {successes}/{len(values)} "
                f"({100 * successes / len(values):.1f}%)", ha="center", fontsize=10)
    ax.axhline(0.99, color="green", linestyle=":", label="Soglia di successo: 0,99")
    ax.set_xticks([0, 1], ["Baseline: fidelity finale\ndi ogni circuito casuale",
                           "Training: fidelity finale di ogni run"])
    ax.set_ylabel("Fidelity finale")
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim(-0.55, 1.55)
    ax.set_title(f"Confronto finale della fidelity: {args.target}")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "final_fidelity_vs_baseline.png", dpi=200)
    plt.close(fig)
    # Elimina solo i grafici obsoleti prodotti dalle versioni precedenti.
    obsolete_names = {
        "figure_9_training_progress.png",
        "combined_metrics.png", "final_fidelity_per_run.png", "gate_count_per_run.png",
        "success_per_run.png", "time_vs_fidelity.png", "time_vs_gate_count.png",
        "time_vs_reward.png", "total_reward_per_run.png",
        "fidelity_vs_iterations_all_runs.png",
    }
    for path in args.output_dir.glob("*.png"):
        if path.name in obsolete_names or path.name.startswith("fidelity_vs_iterations_run_"):
            path.unlink()
    print(f"2 grafici salvati in {args.output_dir}; rimossi i grafici obsoleti dello script.")


if __name__ == "__main__":
    main()


