"""Avvia l'addestramento PPO per sintetizzare uno stato quantistico target."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.tune.logger import CSVLogger, JsonLogger, UnifiedLogger
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from quantum_rl.environment import QuantumCircuitEnv
from quantum_rl.model import MaskedTransformerModel
from quantum_rl.targets import load_target_state

# Salva un checkpoint completo in modo atomico, evitando checkpoint incompleti o solo metadati.
def save_complete_checkpoint(algo, checkpoint_dir: Path, iteration: int, best_circuit: dict | None = None) -> Path:
    """Save a checkpoint atomically and reject metadata-only checkpoints."""
    final_path = (checkpoint_dir / f"iter_{iteration}").resolve()
    temporary_path = (checkpoint_dir / f".iter_{iteration}.tmp").resolve()
    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    try:
        temporary_path.mkdir(parents=True)
        algo.save_checkpoint(str(temporary_path))
        state_path = temporary_path / "algorithm_state.pkl"
        metadata_path = temporary_path / "rllib_checkpoint.json"
        if not metadata_path.is_file() or not state_path.exists():
            raise RuntimeError(
                f"Incomplete checkpoint created at {temporary_path}: "
                "missing rllib_checkpoint.json or algorithm_state"
            )
        if final_path.exists():
            shutil.rmtree(final_path)
        temporary_path.replace(final_path)
        metadata_path = final_path / "rllib_checkpoint.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["state_file"] = str(final_path / "algorithm_state.pkl")
        if best_circuit is not None:
            metadata["best_circuit"] = {
                "fidelity": float(best_circuit["fidelity"]),
                "gate_count": int(best_circuit["gate_count"]),
                "gates": list(best_circuit["gates"]),
            }
            (final_path / "best_circuit.json").write_text(
                json.dumps(metadata["best_circuit"], indent=2) + "\n",
                encoding="utf-8",
            )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"checkpoint={final_path}")
        return final_path
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

# controlla se il checkpoint è completo e contiene sia i metadati che lo stato dell'algoritmo.
def evaluate_best_circuit(algo, env_config: dict, episodes: int = 20) -> dict | None:
    """Evaluate the current policy and return the best circuit seen in the sample."""
    best = None
    for _ in range(episodes):
        env = QuantumCircuitEnv(**env_config)
        obs, _ = env.reset()
        done = False
        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        candidate = (float(info["fidelity"]), len(info["gates"]), list(info["gates"]))
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if best is None:
        return None
    return {"fidelity": best[0], "gate_count": best[1], "gates": best[2]}

# analizza sintatticamente gli argomenti da linea di comando.
def parse_args():
    # Definisce gli argomenti da linea di comando per configurare l'esperimento.
    parser = argparse.ArgumentParser()
    # Definisce il target quantistico da sintetizzare: Bell, GHZ o uno stato custom.
    parser.add_argument("--target", choices=["bell", "ghz", "custom"], default="bell")
    # Se il target è custom, specifica il file contenente lo stato target.
    parser.add_argument("--target-file", help="Custom target: .npy complex vector or JSON amplitudes")
    # Numero di qubit del target; se non specificato, viene dedotto dallo stato target.
    parser.add_argument("--qubits", type=int, default=None)
    # Numero massimo di gate consentiti per sintetizzare il target.
    parser.add_argument("--max-gates", type=int, default=25)
    # Numero di iterazioni di addestramento PPO da eseguire.
    parser.add_argument("--iterations", type=int, default=100)
    # Numero di worker paralleli per raccogliere esperienze durante l'addestramento.
    parser.add_argument("--workers", type=int, default=2)
    # Cartella dove salvare i checkpoint dell'addestramento.
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    # File CSV in cui salvare le metriche di addestramento a ogni iterazione.
    parser.add_argument("--metrics-file", default="training_history.csv",
                        help="CSV in cui salvare reward media e passi raccolti a ogni iterazione")
    return parser.parse_args()


def resolve_target_path(target_file: str) -> Path:
    path = Path(target_file)
    if path.is_file():
        return path
    target_path = Path(__file__).resolve().parent / "targets" / path.name
    if target_path.is_file():
        return target_path
    return path


def main():
    # Legge il target, il numero massimo di gate e altre impostazioni dell'esperimento.
    args = parse_args()
    # Se è stato passato un file target custom, lo carica e usa quello come stato desiderato.
    if args.target_file:
        args.target_file = str(resolve_target_path(args.target_file))
        target_state, args.qubits = load_target_state(args.target_file, args.qubits)
        env_config = {"n_qubits": args.qubits, "target_state": target_state, "max_gates": args.max_gates}
    else:
        if args.target == "custom":
            raise ValueError("--target custom requires --target-file")
        # Per Bell usa 2 qubit; per GHZ usa 3 qubit di default se non specificato.
        args.qubits = 2 if args.target == "bell" else (args.qubits or 3)
        env_config = {"target": args.target, "n_qubits": args.qubits, "max_gates": args.max_gates}
    # Registra l'ambiente e il modello custom così RLlib li può usare.
    print(f"Training target={args.target_file or args.target} qubits={args.qubits} "
          f"max_gates={args.max_gates} iterations={args.iterations} "
          "gate_set=RX,RY,RZ,H,CNOT", flush=True)
    register_env("quantum_circuit", lambda config: QuantumCircuitEnv(**config))
    ModelCatalog.register_custom_model("masked_transformer", MaskedTransformerModel)
    # Avvia Ray e i worker paralleli per l'addestramento.
    # Rispetta le risorse del job: evita di avviare worker per tutti i core
    # fisici del nodo HPC o del PC quando sono stati richiesti pochi worker.
    ray.init(ignore_reinit_error=True, num_cpus=max(1, args.workers + 1),
             include_dashboard=False)
    # Configura PPO con il modello Transformer custom e l'ambiente quantistico.
    config = (PPOConfig().api_stack(enable_rl_module_and_learner=False,
                                    enable_env_runner_and_connector_v2=False)
              .environment("quantum_circuit", env_config=env_config)
              .framework("torch")
              .env_runners(num_env_runners=args.workers,
                           sample_timeout_s=600,
                           rollout_fragment_length=32)
              .training(model={"custom_model": "masked_transformer", "custom_model_config": {"hidden_size": 128}},
                        entropy_coeff=0.02,
                        lr=3e-4, train_batch_size=512, minibatch_size=64, num_epochs=5))
    # Crea l'algoritmo PPO con policy e critic già collegati al modello custom.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (checkpoint_dir / "ray_logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    algo = config.build(logger_creator=lambda cfg: UnifiedLogger(
        cfg, str(log_dir), loggers=[JsonLogger, CSVLogger]))
    metrics_path = Path(args.metrics_file)
    best_so_far = None
    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=[
                "iterazione", "reward medio degli episodi", "lunghezza media degli episodi", "step ambientali campionati",
            ])
            writer.writeheader()
            for iteration in range(1, args.iterations + 1):
                # Esegue un ciclo di raccolta esperienze e aggiornamento PPO.
                result = algo.train()
                env_runners = result.get("env_runners", {})
                episode_reward = env_runners.get("episode_reward_mean", float("nan"))
                episode_length = env_runners.get("episode_len_mean", float("nan"))
                env_steps = result.get("num_env_steps_sampled_lifetime", float("nan"))
                writer.writerow({
                    "iterazione": iteration,
                    "reward medio degli episodi": episode_reward,
                    "lunghezza media degli episodi": episode_length,
                    "step ambientali campionati": env_steps,
                })
                metrics_file.flush()
                print(f"iteration={iteration} mean_episode_return={episode_reward}")
                if iteration % 20 == 0 or iteration == args.iterations:
                    current_best = evaluate_best_circuit(algo, env_config, episodes=20)
                    if current_best is not None:
                        if best_so_far is None or current_best["fidelity"] > best_so_far["fidelity"] or (
                            current_best["fidelity"] == best_so_far["fidelity"] and current_best["gate_count"] < best_so_far["gate_count"]
                        ):
                            best_so_far = current_best
                    save_complete_checkpoint(algo, checkpoint_dir, iteration, best_so_far)
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
