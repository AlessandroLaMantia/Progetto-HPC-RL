"""Carica un checkpoint e mostra il miglior circuito trovato dalla policy addestrata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from quantum_rl.environment import QuantumCircuitEnv
from quantum_rl.model import MaskedTransformerModel
from quantum_rl.targets import load_target_state

# Controlla se il checkpoint è completo e contiene sia i metadati che lo stato dell'algoritmo.
def validate_checkpoint(path: str) -> None:
    checkpoint_path = Path(path).resolve()
    metadata_path = checkpoint_path / "rllib_checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state_file = Path(metadata.get("state_file", "algorithm_state.pkl"))
    state_path = state_file if state_file.is_absolute() else checkpoint_path / state_file
    if not state_path.exists():
        raise FileNotFoundError(
            f"Checkpoint incomplete: missing algorithm state at {state_path}"
        )

# Carica il miglior circuito trovato nel checkpoint, se presente.
def load_checkpoint_best_circuit(path: str) -> dict | None:
    checkpoint_path = Path(path).resolve()
    metadata_path = checkpoint_path / "rllib_checkpoint.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    best = metadata.get("best_circuit")
    if not best:
        return None
    return {
        "fidelity": float(best.get("fidelity", 0.0)),
        "gate_count": int(best.get("gate_count", 0)),
        "gates": list(best.get("gates", [])),
    }


def resolve_target_path(target_file: str) -> Path:
    path = Path(target_file)
    if path.is_file():
        return path
    target_path = Path(__file__).resolve().parent / "targets" / path.name
    if target_path.is_file():
        return target_path
    return path


# Funzione principale per valutare il checkpoint.
def main():
    # Definisce gli argomenti per caricare il checkpoint e riprodurre il comportamento dell'agente.
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", help="Checkpoint PPO da valutare")
    parser.add_argument("--random", action="store_true",
                        help="Usa una policy casuale come baseline pre-addestramento")
    parser.add_argument("--target", choices=["bell", "ghz", "custom"], default="bell")
    parser.add_argument("--target-file", help="Custom target used during training")
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--max-gates", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--report", help="Percorso del report JSON con le metriche aggregate")
    args = parser.parse_args()
    if args.random == (args.checkpoint is not None):
        parser.error("specifica un checkpoint oppure --random, ma non entrambi")
    # Se è stato passato un target custom, lo usa anche nella fase di valutazione.
    if args.target_file:
        args.target_file = str(resolve_target_path(args.target_file))
        target_state, args.qubits = load_target_state(args.target_file, args.qubits)
        env_kwargs = {"n_qubits": args.qubits, "target_state": target_state, "max_gates": args.max_gates}
    else:
        if args.target == "custom":
            raise ValueError("--target custom requires --target-file")
        args.qubits = 2 if args.target == "bell" else (args.qubits or 3)
        env_kwargs = {"target": args.target, "n_qubits": args.qubits, "max_gates": args.max_gates}
    policy = None
    reference_env = QuantumCircuitEnv(**env_kwargs)
    preprocessor = ModelCatalog.get_preprocessor_for_space(reference_env.observation_space)
    if not args.random:
        checkpoint_path = Path(args.checkpoint).resolve()
        validate_checkpoint(str(checkpoint_path))
        # Registra ambiente e modello per riprodurre il comportamento del checkpoint.
        register_env("quantum_circuit", lambda config: QuantumCircuitEnv(**config))
        ModelCatalog.register_custom_model("masked_transformer", MaskedTransformerModel)
        # Per l'inferenza basta la policy: non ricreare il cluster e i worker
        # di training sul nodo HPC per ciascuna valutazione delle run.
        policies = Policy.from_checkpoint(str(checkpoint_path))
        policy = policies["default_policy"] if isinstance(policies, dict) else policies
        if (policy.action_space.n != reference_env.action_space.n or
                policy.observation_space.shape != preprocessor.observation_space.shape):
            raise ValueError("Checkpoint incompatibile con target/max-gates: avviare un nuovo training")
        saved_env_config = policy.config.get("env_config")
        if saved_env_config:
            trained_env = QuantumCircuitEnv(**saved_env_config)
            if (trained_env.target_state.shape != reference_env.target_state.shape or
                    not np.allclose(trained_env.target_state, reference_env.target_state)):
                raise ValueError("Il target di valutazione non corrisponde a quello del training")
        checkpoint_best = load_checkpoint_best_circuit(str(checkpoint_path))
        if checkpoint_best is not None:
            print(f"Best circuit in checkpoint: {' -> '.join(checkpoint_best['gates'])}")
            print(f"Checkpoint best fidelity={checkpoint_best['fidelity']:.8f}")
    best = None
    fidelities, gate_counts, episode_rewards, successes = [], [], [], 0
    for _ in range(args.episodes):
        # Crea un nuovo episodio per testare la policy sullo stesso ambiente.
        env = QuantumCircuitEnv(**env_kwargs)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if args.random:
                # Baseline: seleziona uniformemente fra le sole azioni consentite.
                valid_actions = np.flatnonzero(obs["action_mask"])
                action = env.np_random.choice(valid_actions)
            else:
                # Usa solo la policy addestrata, senza esplorazione casuale.
                action, _, _ = policy.compute_single_action(preprocessor.transform(obs), explore=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            done = terminated or truncated
        # Salva la soluzione migliore in base a fedeltà e brevità.
        candidate = (info["fidelity"], len(info["gates"]), info["gates"])
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
        fidelities.append(float(info["fidelity"]))
        gate_counts.append(len(info["gates"]))
        episode_rewards.append(total_reward)
        successes += int(info["fidelity"] >= 1.0 - env.threshold)
    metrics = {
        "policy": "random" if args.random else "trained",
        "episodes": args.episodes,
        "mean_final_fidelity": float(np.mean(fidelities)),
        "best_final_fidelity": float(best[0]),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_circuit_length": float(np.mean(gate_counts)),
        "success_rate_percent": 100.0 * successes / args.episodes,
    }
    print(f"Best circuit: {' -> '.join(best[2])}")
    print(f"Mean final fidelity={metrics['mean_final_fidelity']:.8f}")
    print(f"Best final fidelity={metrics['best_final_fidelity']:.8f}")
    print(f"Mean episode reward={metrics['mean_episode_reward']:.4f}")
    print(f"Mean circuit length={metrics['mean_circuit_length']:.2f}")
    print(f"Success rate={metrics['success_rate_percent']:.2f}%")
    if args.report:
        Path(args.report).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"report={args.report}")


if __name__ == "__main__":
    main()
