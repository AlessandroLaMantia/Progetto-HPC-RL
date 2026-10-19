"""
l'obiettivo di questo modello è quello di codificare lo stato vettoriale come token, e mascherare i dati delle azioni non valide """

from __future__ import annotations

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from torch import nn


class MaskedTransformerModel(TorchModelV2, nn.Module):
    """Encode the statevector as tokens, then mask invalid gate logits."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        # Inizializza l'interfaccia richiesta da RLlib.
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        # Inizializza anche l'interfaccia standard di PyTorch.
        nn.Module.__init__(self)
        # Determina la dimensione dello stato in base allo spazio di osservazione fornito.
        original_obs_space = getattr(obs_space, "original_space", obs_space)
        if hasattr(original_obs_space, "spaces") and "state" in original_obs_space.spaces:
            state_shape = original_obs_space.spaces["state"].shape
            self._dict_obs = True
        else:
            state_shape = getattr(obs_space, "shape", (num_outputs,))
            self._dict_obs = False
        self.state_size = int(state_shape[0]) if hasattr(state_shape, "__len__") else int(state_shape)
        # Dimensione nascosta del modello, con valore di default 128.
        hidden = int(model_config.get("custom_model_config", {}).get("hidden_size", 128))
        # Ogni valore dello stato viene trattato come un token indipendente da elaborare.
        self.token_embedding = nn.Linear(1, hidden)
        # Identifica ampiezza, parte reale/immaginaria e indicatori di progresso.
        self.position_embedding = nn.Parameter(torch.empty(1, self.state_size, hidden))
        nn.init.normal_(self.position_embedding, std=0.02)
        # Strato encoder basato su transformer con self-attention.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden * 2,
                                                   batch_first=True, dropout=0.0)
        # Due blocchi di self-attention.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Policy head: produce i logit per ogni possibile azione.
        self.policy = SlimFC(hidden, num_outputs, activation_fn=None)
        # Value head: stima il valore di stato per PPO.
        self.value = SlimFC(hidden, 1, activation_fn=None)
        self._value_out = None
        self.action_mask_size = int(num_outputs)

    def forward(self, input_dict, state, seq_lens):
        # Estrae lo stato e il mask delle azioni dall'osservazione fornita da RLlib.
        obs = input_dict.get("obs", input_dict)
        if isinstance(obs, dict):
            state_tensor = obs.get("state", obs.get("obs", obs.get("obs_flat")))
            action_mask = obs.get("action_mask", input_dict.get("action_mask"))
        else:
            state_tensor = obs
            action_mask = input_dict.get("action_mask")

        if state_tensor is None:
            raise ValueError("Custom model received an observation without a state tensor")

        if hasattr(state_tensor, "dtype") and state_tensor.dtype in (torch.float64, torch.float32, torch.float16):
            pass
        elif hasattr(state_tensor, "__iter__") and not isinstance(state_tensor, (str, bytes)):
            state_tensor = torch.as_tensor(state_tensor, dtype=torch.float32)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Trasforma lo stato in una sequenza di token, uno per elemento.
        values = state_tensor.float().unsqueeze(-1)  # [batch, token, 1]
        # Passa i token attraverso il transformer e li riduce a un singolo vettore.
        tokens = self.token_embedding(values) + self.position_embedding
        encoded = self.encoder(tokens).mean(dim=1)
        # Calcola il valore di stato previsto dal critic.
        self._value_out = self.value(encoded).squeeze(1)
        # Produce i logit della policy per tutte le azioni possibili.
        logits = self.policy(encoded)

        if action_mask is None:
            mask = torch.ones(logits.shape, device=logits.device, dtype=logits.dtype)
        else:
            mask = action_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[-1] != logits.shape[-1]:
                raise ValueError("Action mask size does not match policy logits")

        # Applica il mask delle azioni non ammesse, rendendole praticamente impossibili.
        # exp(-1e9) è zero in float32, senza overflow nelle statistiche RLlib.
        return logits.masked_fill(mask <= 0, -1e9), state

    def value_function(self):
        return self._value_out
