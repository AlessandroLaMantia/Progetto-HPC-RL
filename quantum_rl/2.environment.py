"""
Si occupa di usare il target per far lavorare l’ambiente quantistico
Vengono utilizzate le seguenti librerie:
Gymnasium per l’ambiente RL,
NumPy per i vettori e gli array,
PennyLane per simulare il circuito quantistico,
PyTorch per ottimizzare gli angoli."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pennylane as qml
import torch

from .targets import validate_target_state


ROTATIONS = {"RX", "RY", "RZ"}


# Struttura che rappresenta una singola azione del circuito.
@dataclass(frozen=True)
class GateAction:
    name: str
    wires: tuple[int, ...]


# Classe principale: costruisce e fa evolvere il circuito quantistico.
class QuantumCircuitEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": ["human"]}

    # Inizializza l'ambiente, il target e tutte le strutture di supporto.
    def __init__(
        self,
        # Target predefinito, ad esempio "bell" o "ghz".
        target: str = "bell",
        # Numero di qubit del circuito.
        n_qubits: int | None = None,
        # Se lo passi, usa questo stato come target invece di "bell" o "ghz".
        target_state: np.ndarray | list[complex] | None = None,
        # Massimo numero di gate che l'agente può usare.
        max_gates: int = 6,
        # Soglia di successo: se la distanza dal target è sotto questa soglia, il problema è considerato risolto.
        threshold: float = 1e-2,
        # Parametri che regolano la reward.
        loss_mult: float = 10.0,
        step_penalty: float = 1.0,
        bonus_milestone: float = 10.0,
        final_rew: float = 100.0,
        inner_steps: int = 60,
        angle_lr: float = 0.08,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # Ogni worker Ray dispone di un core: evita pool PyTorch concorrenti
        # su tutti i core del nodo durante le piccole simulazioni quantistiche.
        torch.set_num_threads(1)
        # Se viene fornito uno stato target personalizzato, lo convalida e lo usa
        if target_state is not None:
            self.target_state, n_qubits = validate_target_state(np.asarray(target_state), n_qubits)
            self.target_name = "custom"
        # Altrimenti, costruisce uno stato target predefinito in base al nome
        else:
            n_qubits = 2 if target == "bell" else (n_qubits or 3)
            if target == "bell":
                if n_qubits != 2:
                    raise ValueError("The Bell target needs exactly 2 qubits")
                self.target_state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
            elif target == "ghz":
                if n_qubits < 2:
                    raise ValueError("GHZ requires at least two qubits")
                self.target_state = np.zeros(2**n_qubits, dtype=np.complex128)
                self.target_state[0] = self.target_state[-1] = 1 / np.sqrt(2)
            # Se il nome del target non è riconosciuto, solleva un errore
            else:
                raise ValueError(f"Unknown target {target!r}; choose bell or ghz")
            self.target_name = target
        # Salva i parametri dell'ambiente e costruisce lo spazio delle azioni e delle osservazioni (non crea le azioni/osservazioni, lo fa dopo)
        self.n_qubits = n_qubits
        self.max_gates, self.threshold = max_gates, threshold
        self.loss_mult, self.step_penalty = loss_mult, step_penalty
        self.bonus_milestone, self.final_rew = bonus_milestone, final_rew
        self.inner_steps, self.angle_lr = inner_steps, angle_lr
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.actions = self._build_action_catalogue()
        self.action_space = spaces.Discrete(len(self.actions))
        # Lo stato fisico da solo non identifica il circuito parametrico che
        # verrà riottimizzato: includi struttura e angoli, con padding fisso.
        state_size = 2 * (2**n_qubits) + 2 + 2 * max_gates
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-1.0, 1.0, shape=(state_size,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(len(self.actions),), dtype=np.int8),
        })
        self.current_gates: list[GateAction] = []
        self.rotation_angles = np.empty(0, dtype=np.float64)
        self.best_gate_count_ever: int | None = None
        self.reached_milestones: set[float] = set()
        self.prev_dist = 1.0
        self.current_state = self._simulate()
        self.reset(seed=seed)

    # Crea l'elenco di tutte le azioni discrete disponibili per l'agente.
    def _build_action_catalogue(self) -> list[GateAction]:
        # Crea azioni per le rotazioni su ogni qubit
        actions = [GateAction(name, (wire,)) for name in ("RX", "RY", "RZ") for wire in range(self.n_qubits)]
        # Una sovrapposizione fissa non viene annullata da Adam prima che
        # l'agente abbia aggiunto i gate di entanglement (plateau Bell/GHZ).
        actions.extend(GateAction("H", (wire,)) for wire in range(self.n_qubits))
        # Crea azioni per le porte CNOT tra ogni coppia di qubit (controllo, target)
        actions.extend(GateAction("CNOT", (control, target))
                       for control in range(self.n_qubits)
                       for target in range(self.n_qubits) if control != target)
        return actions


    # Simula lo stato quantistico del circuito corrente con PennyLane:
    # input: angoli + porte attuali;
    # output: come si presenta il qubit/circuito dopo queste operazioni
    def _state_torch(self, angles: torch.Tensor) -> torch.Tensor:
        # Fissa le porte già scelte, così il circuito non cambia durante la simulazione.
        gates = tuple(self.current_gates)
        # Definisce un circuito quantistico differenziabile con PennyLane e PyTorch.
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(parameters):
            # Tiene traccia dell'angolo da usare per ogni porta di rotazione.
            parameter_index = 0
            for gate in gates:
                # Se la porta è un CNOT, la applica sui qubit indicati.
                if gate.name == "CNOT":
                    qml.CNOT(wires=gate.wires)
                elif gate.name == "H":
                    qml.Hadamard(wires=gate.wires[0])
                else:
                    # Altrimenti applica una rotazione RX, RY o RZ con l'angolo corrente.
                    getattr(qml, gate.name)(parameters[parameter_index], wires=gate.wires[0])
                    parameter_index += 1
            # Restituisce lo stato quantistico finale del circuito.
            return qml.state()
        # Esegue il circuito e restituisce lo stato, mantenendo il grafo per il backpropagation.
        return circuit(angles)

    # Esegue la simulazione e restituisce lo stato come array NumPy.
    def _simulate(self) -> np.ndarray:
        angles = torch.tensor(self.rotation_angles, dtype=torch.float64)
        # evita di calcolare i gradienti durante la simulazione, poiché non è necessario in questo passaggio
        with torch.no_grad():
            state = self._state_torch(angles)
        return state.detach().cpu().numpy().astype(np.complex128)

    # Aggiorna gli angoli delle porte per avvicinare il circuito al target.
    def _optimise_angles(self) -> None:
        # Se non ci sono angoli da ottimizzare, non serve fare nulla.
        if not len(self.rotation_angles):
            return
        # Crea una copia dei parametri come variabile differenziabile di PyTorch.
        parameters = torch.tensor(self.rotation_angles, dtype=torch.float64, requires_grad=True)
        # Converte lo stato target in un tensore compatibile con il calcolo automatico.
        target = torch.tensor(self.target_state, dtype=torch.complex128)
        # Sceglie l'ottimizzatore interno per adattare gli angoli.
        optimiser = torch.optim.Adam([parameters], lr=self.angle_lr)
        best_parameters = parameters.detach().clone()
        best_loss = float("inf")
        # Esegue più passi di ottimizzazione per trovare angoli migliori.
        for _ in range(self.inner_steps):
            # Pulisce i gradienti accumulati dal passo precedente.
            optimiser.zero_grad()
            # Simula lo stato quantistico attuale con i parametri correnti.
            state = self._state_torch(parameters)
            # Calcola la fedeltà tra stato corrente e target.
            fidelity = torch.abs(torch.sum(torch.conj(target) * state)) ** 2
            # Trasforma la fedeltà in una perdita da minimizzare.
            loss = 1.0 - fidelity.real
            if not torch.isfinite(loss):
                break
            # Adam può oltrepassare il minimo: conserva il miglior iterato,
            # non soltanto l'ultimo passo dell'ottimizzatore.
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_parameters = parameters.detach().clone()
            if best_loss < self.threshold * 0.1:
                break
            loss.backward()
            # Un punto stazionario non migliora ripetendo Adam: il multi-start
            # in step() proverà un'altra inizializzazione se necessario.
            if parameters.grad is None or not torch.isfinite(parameters.grad).all():
                break
            if torch.max(torch.abs(parameters.grad)).item() < 1e-8:
                break
            # Aggiorna gli angoli con il passo di ottimizzazione.
            optimiser.step()
        with torch.no_grad():
            final_state = self._state_torch(parameters)
            final_loss = 1.0 - (torch.abs(torch.sum(torch.conj(target) * final_state)) ** 2).real
            if torch.isfinite(final_loss) and final_loss.item() < best_loss:
                best_parameters = parameters.detach().clone()
        # Riporta gli angoli in un intervallo leggibile, equivalente modulo $2\pi$.
        self.rotation_angles = np.remainder(best_parameters.cpu().numpy() + np.pi, 2 * np.pi) - np.pi

    # Misura la distanza tra lo stato corrente e il target desiderato.
    def _distance(self) -> float:
        # Calcola la fedeltà tra lo stato attuale e il target.
        fidelity = abs(np.vdot(self.target_state, self.current_state)) ** 2
        # Converte la fedeltà in una distanza di errore: 0 = perfetto, 1 = completamente diverso.
        return float(np.clip(1.0 - fidelity, 0.0, 1.0)) 

    # Impedisce azioni ridondanti o non utili.
    def _action_mask(self) -> np.ndarray:
        # Inizia con tutte le azioni disponibili.
        mask = np.ones(self.action_space.n, dtype=np.int8)
        # Qubit potenzialmente attivi per qualsiasi valore degli angoli:
        # non usare lo stato numerico attuale, che Adam può aver azzerato.
        active = set()
        for gate in self.current_gates:
            if gate.name in {"RX", "RY", "H"}:
                active.add(gate.wires[0])
            elif gate.name == "CNOT" and gate.wires[0] in active:
                active.add(gate.wires[1])
        for index, candidate in enumerate(self.actions):
            if candidate.name == "CNOT" and candidate.wires[0] not in active:
                mask[index] = 0
            elif candidate.name == "RZ" and candidate.wires[0] not in active:
                mask[index] = 0
            # Blocca duplicati anche attraverso gate su qubit disgiunti.
            # Per gate che condividono qubit, mantieni l'ordine originale.
            for previous in reversed(self.current_gates):
                if previous == candidate:
                    mask[index] = 0
                    break
                if set(previous.wires).intersection(candidate.wires):
                    break
        return mask

    # Prepara l'osservazione che l'agente riceve dopo ogni passo.
    def _get_obs(self) -> dict[str, np.ndarray]:
        """Observation after an action: the measured/simulated quantum state."""
        # Trasforma lo stato complesso in una forma numerica semplice per l'agente.
        flat_state = np.concatenate((self.current_state.real, self.current_state.imag))
        # Aggiunge informazioni sul progresso del circuito e sulla distanza dal target.
        progress = np.array([len(self.current_gates) / self.max_gates, self._distance()], dtype=np.float64)
        history = np.zeros((self.max_gates, 2), dtype=np.float64)
        angle_index = 0
        for index, gate in enumerate(self.current_gates):
            history[index, 0] = (self.actions.index(gate) + 1) / len(self.actions)
            if gate.name in ROTATIONS:
                history[index, 1] = self.rotation_angles[angle_index] / np.pi
                angle_index += 1
        return {"state": np.concatenate((flat_state, progress, history.ravel())).astype(np.float32), "action_mask": self._action_mask()}

    # Converte le porte scelte in una forma leggibile per il debug.
    def _format_gates(self) -> list[str]:
        # Costruisce una lista di stringhe che descrive il circuito in modo leggibile.
        labels, angle_index = [], 0
        for gate in self.current_gates:
            if gate.name in ROTATIONS:
                # Per le rotazioni mostra anche il valore numerico dell'angolo attuale.
                labels.append(f"{gate.name}({self.rotation_angles[angle_index]:.6f}, {gate.wires[0]})")
                angle_index += 1
            elif gate.name == "H":
                labels.append(f"H({gate.wires[0]})")
            else:
                # Per i CNOT mostra i qubit di controllo e target.
                labels.append(f"CNOT({gate.wires[0]}, {gate.wires[1]})")
        return labels

    # Restituisce metriche utili per monitorare il progresso.
    def _get_info(self) -> dict[str, Any]:
        # Calcola la distanza corrente dal target e la trasforma in una metrica leggibile.
        distance = self._distance()
        return {"distance": distance, "fidelity": 1 - distance, "gates": self._format_gates()}

    # Riporta l'ambiente allo stato iniziale di un nuovo episodio.
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Inizializza il seed di Gymnasium per la riproducibilità.
        super().reset(seed=seed)
        # Ogni episodio riparte dal circuito identità, senza porte applicate.
        self.current_gates = []
        self.rotation_angles = np.empty(0, dtype=np.float64)
        self.reached_milestones.clear()
        self.current_state = self._simulate()
        self.prev_dist = self._distance()
        return self._get_obs(), self._get_info()

    # Applica un'azione, aggiorna il circuito e calcola reward e stato finale.
    def step(self, action: int):
        # Controlla che l'azione sia valida e non sia stata bloccata dal mask.
        if not self.action_space.contains(action) or not self._action_mask()[action]:
            raise ValueError(f"Invalid or masked action: {action}")
        # Estrae la porta scelta dall'agente.
        gate = self.actions[action]
        # Aggiunge la porta al circuito corrente.
        self.current_gates.append(gate)

        # INIZIALIZZA L'ANGOLAZIONE DELLA NUOVA ROTAZIONE A 0
        # aggiorna il nuovo angolo a 0
        if gate.name in ROTATIONS:
            self.rotation_angles = np.append(self.rotation_angles, 0.0)

        # Riottimizza tutti gli angoli anche dopo un CNOT.
        if len(self.rotation_angles):
            initial_angles = self.rotation_angles.copy()
            # Conserva anche il candidato prima dell'ottimizzazione.
            best_angles = initial_angles.copy()
            best_fid = float(abs(np.vdot(self.target_state, self._simulate())) ** 2)

            # PROVA PRIMA GLI ANGOLI ATTUALI, POI RIPARTI DA UN INIZIO CASUALE SE NON FUNZIONA.
            for restart in range(2):
                if restart == 0:
                    self.rotation_angles = initial_angles.copy()
                else:
                    self.rotation_angles = self.np_random.uniform(
                        -np.pi, np.pi, size=initial_angles.shape
                    )
                self._optimise_angles()
                candidate_state = self._simulate()
                fidelity = float(abs(np.vdot(self.target_state, candidate_state)) ** 2)
                if fidelity > best_fid:
                    best_fid = fidelity
                    best_angles = self.rotation_angles.copy()
                # Un piccolo miglioramento non esclude un minimo locale:
                # prova anche la ripartenza casuale se il target non è risolto.
                if 1.0 - best_fid < self.threshold:
                    break

            self.rotation_angles = best_angles

        # Simula lo stato quantistico dopo l'azione.
        self.current_state = self._simulate()
        distance = self._distance()

        # Reward fornito dal docente: miglioramento, costo dei gate, traguardi e bonus finale.
        epsilon = 1e-12
        reward = (-math.log(distance + epsilon) + math.log(self.prev_dist + epsilon)) * self.loss_mult - self.step_penalty
        self.prev_dist = distance
        # Aggiunge un bonus quando il circuito supera una soglia di precisione.
        for exponent in range(2, int(-math.log10(self.threshold)) + 1, 2):
            milestone = 10 ** (-exponent)
            if distance < milestone and milestone not in self.reached_milestones:
                reward += self.bonus_milestone
                self.reached_milestones.add(milestone)
        # Il problema è risolto se la distanza dal target è sotto la soglia.
        terminated = distance < self.threshold
        # L'episodio termina anche se si raggiunge il numero massimo di gate.
        truncated = len(self.current_gates) >= self.max_gates
        if terminated:
            # Bonus aggiuntivi per circuiti più brevi e più veloci da risolvere.
            gates_saved = self.max_gates - len(self.current_gates)
            base_bonus = gates_saved * 5
            exp_bonus = 0.0 if self.best_gate_count_ever is None else 50 * math.exp(
                -0.5 * max(len(self.current_gates) - self.best_gate_count_ever, -2)
            )
            reward += self.final_rew + base_bonus + exp_bonus
            if self.best_gate_count_ever is None or len(self.current_gates) < self.best_gate_count_ever:
                self.best_gate_count_ever = len(self.current_gates)
        info = self._get_info() | {"solved": terminated}
        return self._get_obs(), float(reward), terminated, truncated, info

    # Stampa a schermo la sequenza di gate usata fino a quel momento.
    def render(self) -> None:
        # Mostra il circuito corrente in forma leggibile, oppure il caso neutro di identità.
        print(" -> ".join(self._format_gates()) or "identity")
