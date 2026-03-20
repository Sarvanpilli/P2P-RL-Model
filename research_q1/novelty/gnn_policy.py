import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch.distributions import Normal
from typing import Tuple, Optional, List, Dict, Type, Union

from torch_geometric.nn import GATv2Conv, global_mean_pool
from research_q1.novelty.grid_graph import get_grid_graph


class CTDEGNNPolicy(ActorCriticPolicy):
    """
    Phase C: Multi-Agent CTDE Graph Policy using PyG GATv2.

    Architecture:
    - Shared GNN backbone: 2x GATv2Conv layers that propagate local information.
    - Actor (Decentralized per-node MLP): only uses local node embedding -> action.
    - Critic (Centralized global pool): global_mean_pool over all node embeddings -> value.

    The class overrides _build() and _build_mlp_extractor() to suppress SB3's default
    MLP construction, preventing duplication/conflict with the GNN layers.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        n_agents: int = 4,
        **kwargs,
    ):
        # Strip out net_arch to avoid conflicts; we completely own the network construction
        kwargs.pop("net_arch", None)
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)

        self.n_agents = n_agents

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass empty net_arch so SB3 doesn't build any MLP on top
            net_arch=[],
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Override: suppress SB3's default MlpExtractor.
        Our GNN layers ARE the feature extractor.
        """
        # SB3 expects self.mlp_extractor + latent dims to exist after _build()
        # We set them to dummy placeholders that will not be called in forward().
        self.mlp_extractor = nn.Identity()
        self.latent_dim_pi = 0
        self.latent_dim_vf = 0

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Override _build() to construct only our custom GNN components.
        Avoids SB3's internal action_net / value_net being built on top.
        """
        self._build_mlp_extractor()

        obs_dim_total = self.observation_space.shape[0]
        act_dim_total = self.action_space.shape[0]

        self.obs_dim_per_agent = obs_dim_total // self.n_agents
        self.action_dim_per_agent = act_dim_total // self.n_agents

        edge_index = get_grid_graph(self.n_agents)
        self.register_buffer("edge_index", edge_index)

        hidden_dim = 64
        heads = 2

        # ── Shared GNN Backbone ──────────────────────────────────────────────
        self.conv1 = GATv2Conv(
            self.obs_dim_per_agent, hidden_dim, heads=heads, concat=True
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False
        )

        # ── Decentralized Actor Head (per-node) ──────────────────────────────
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim_per_agent),
        )

        # Log-std is a global learnable parameter (one value per action dim)
        self.log_std = nn.Parameter(
            torch.zeros(act_dim_total), requires_grad=True
        )

        # ── Centralized Critic Head (global pooled embedding) ────────────────
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Optimizer — SB3 expects this to be created in _build()
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_graph_embeddings(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch-Graph Bridge:
        SB3 passes [Batch, N_agents * ObsDim].  PyG needs [Batch*N, ObsDim] + batch_vec.
        """
        batch_size = obs.shape[0]
        x = obs.view(batch_size * self.n_agents, self.obs_dim_per_agent)

        batch_vec = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.n_agents
        )

        # Offset edge_index for each graph in the batch  [2, Batch * E]
        num_nodes = self.n_agents
        offset = torch.arange(
            0, batch_size * num_nodes, step=num_nodes, device=x.device
        )
        batched_edge_index = self.edge_index.unsqueeze(1) + offset.view(1, batch_size, 1)
        batched_edge_index = batched_edge_index.view(2, -1)

        # Two-layer GATv2 message passing
        h = torch.relu(self.conv1(x, batched_edge_index))          # [B*N, H*heads]
        node_emb = torch.relu(self.conv2(h, batched_edge_index))   # [B*N, H]

        global_emb = global_mean_pool(node_emb, batch_vec)         # [B, H]

        return node_emb, global_emb

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_emb, global_emb = self._get_graph_embeddings(obs)
        batch_size = obs.shape[0]

        # Decentralized Actor
        mean_actions = self.actor_net(node_emb).view(batch_size, -1)  # [B, N*ActDim]
        action_std = torch.exp(self.log_std).expand_as(mean_actions)
        dist = Normal(mean_actions, action_std)
        actions = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=1)

        # Centralized Critic
        values = self.critic_net(global_emb)  # [B, 1]

        return actions, values, log_prob

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        node_emb, global_emb = self._get_graph_embeddings(obs)
        batch_size = obs.shape[0]

        mean_actions = self.actor_net(node_emb).view(batch_size, -1)
        action_std = torch.exp(self.log_std).expand_as(mean_actions)
        dist = Normal(mean_actions, action_std)

        log_prob = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        values = self.critic_net(global_emb)

        return values, log_prob, entropy

    # ─────────────────────────────────────────────────────────────────────────
    def get_distribution(self, obs: torch.Tensor) -> Normal:
        node_emb, _ = self._get_graph_embeddings(obs)
        batch_size = obs.shape[0]
        mean_actions = self.actor_net(node_emb).view(batch_size, -1)
        action_std = torch.exp(self.log_std).expand_as(mean_actions)
        return Normal(mean_actions, action_std)

    # ─────────────────────────────────────────────────────────────────────────
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Override SB3's _predict to handle raw PyTorch Normal distributions correctly."""
        node_emb, _ = self._get_graph_embeddings(observation)
        batch_size = observation.shape[0]
        mean_actions = self.actor_net(node_emb).view(batch_size, -1)
        
        if deterministic:
            return mean_actions
            
        action_std = torch.exp(self.log_std).expand_as(mean_actions)
        return Normal(mean_actions, action_std).rsample()

    # ─────────────────────────────────────────────────────────────────────────
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Required by SB3's rollout collection to bootstrap value estimates."""
        _, global_emb = self._get_graph_embeddings(obs)
        return self.critic_net(global_emb)

    # ─────────────────────────────────────────────────────────────────────────
    def extract_attention(self, obs: torch.Tensor):
        """
        Phase E — Interpretability:
        Returns (edge_index, alpha) from conv1 using return_attention_weights=True.
        """
        batch_size = obs.shape[0]
        x = obs.view(batch_size * self.n_agents, self.obs_dim_per_agent)

        num_nodes = self.n_agents
        offset = torch.arange(
            0, batch_size * num_nodes, step=num_nodes, device=x.device
        )
        batched_edge_index = self.edge_index.unsqueeze(1) + offset.view(1, batch_size, 1)
        batched_edge_index = batched_edge_index.view(2, -1)

        _, (edges, alpha) = self.conv1(
            x, batched_edge_index, return_attention_weights=True
        )
        return edges, alpha
