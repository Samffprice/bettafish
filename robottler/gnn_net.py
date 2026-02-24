"""Graph Neural Network for Catan board evaluation.

Pure GNN architecture that processes raw per-node features on the board graph,
replacing the MLP that operates on 176 aggregated features.

The Catan board is a fixed graph: 54 vertices, ~144 edges. Message passing
is implemented as dense matmuls (no PyG/DGL dependency needed).

Architecture:
    Per-node features [B, 54, 18] → GNN layers → MeanPool → [B, D_gnn]
    Global features [B, 76] → FC → [B, D_global]
    Concat → Body → Value head (tanh) + Policy head (logits)

Default (small):  gnn_dim=32, global_dim=64, body_dim=96   ~45K params
Medium:           gnn_dim=64, global_dim=128, body_dim=192  ~130K params
Large:            gnn_dim=128, global_dim=256, body_dim=256  ~300K params
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robottler.bitboard.masks import (
    NUM_NODES, ADJACENT_NODES, NODE_BIT, TILE_NODES,
    INCIDENT_EDGES, EDGE_BIT_WORD, EDGE_BIT_MASK,
)
from robottler.bitboard.state import (
    PS_VP, PS_ROADS_AVAIL, PS_SETTLE_AVAIL, PS_CITY_AVAIL,
    PS_HAS_ROAD, PS_HAS_ARMY, PS_HAS_ROLLED, PS_HAS_PLAYED_DEV,
    PS_ACTUAL_VP, PS_LONGEST_ROAD,
    PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
    PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
    PS_PLAYED_KNIGHT, PS_PLAYED_YOP, PS_PLAYED_MONO, PS_PLAYED_RB,
    PS_RESOURCE_START, PS_RESOURCE_END,
    PROMPT_MOVE_ROBBER, PROMPT_DISCARD,
)
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

# Per-node feature layout
NODE_FEAT_DIM = 18   # 4 settlement + 4 city + 4 road_adj + 5 production + 1 robber
GLOBAL_FEAT_DIM = 76  # See bb_fill_global_features

# Dev card PS indices
_PS_DEV_HAND = (PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND)
_PS_DEV_PLAYED = (PS_PLAYED_KNIGHT, PS_PLAYED_YOP, PS_PLAYED_MONO, PS_PLAYED_RB)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adjacency_matrix():
    """Build row-normalized adjacency matrix [54, 54] from board topology.

    Returns D^{-1}A where A is the binary adjacency and D is the degree matrix.
    This is a constant — the Catan board topology never changes.
    """
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if ADJACENT_NODES[i] & NODE_BIT[j]:
                A[i, j] = 1.0
    D = A.sum(axis=1, keepdims=True)
    D[D == 0] = 1.0
    return A / D


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def bb_fill_node_features(state, p0_color, buf, node_prod):
    """Extract per-node features from BitboardState.

    Args:
        state: BitboardState
        p0_color: Color of the player from whose perspective features are extracted
        buf: Pre-zeroed numpy array of shape [54, 18] (float32)
        node_prod: [54, 5] array of node production probabilities (from FeatureIndexer)
    """
    p0_idx = state.color_to_index[p0_color]
    n = state.num_players
    robber_mask = TILE_NODES[state.robber_tile]

    for node in range(NUM_NODES):
        bit = NODE_BIT[node]
        off = 0

        # Building ownership: P0-P3 settlement (4) + P0-P3 city (4) = 8
        for slot in range(min(n, 4)):
            pidx = (p0_idx + slot) % n
            buf[node, off] = 1.0 if (state.settlement_bb[pidx] & bit) else 0.0
            buf[node, off + 1] = 1.0 if (state.city_bb[pidx] & bit) else 0.0
            off += 2
        # Pad remaining slots with 0 if fewer than 4 players
        off = 8

        # Road adjacency: P0-P3 has_road_adjacent (4)
        for slot in range(min(n, 4)):
            pidx = (p0_idx + slot) % n
            has_road = False
            for eidx in INCIDENT_EDGES[node]:
                w = int(EDGE_BIT_WORD[eidx])
                m = EDGE_BIT_MASK[eidx]
                if state.road_bb[pidx, w] & m:
                    has_road = True
                    break
            buf[node, off] = 1.0 if has_road else 0.0
            off += 1
        off = 12

        # Production per resource (5)
        buf[node, 12:17] = node_prod[node]

        # Robber on adjacent tile (1)
        buf[node, 17] = 1.0 if (robber_mask & bit) else 0.0


def bb_fill_global_features(state, p0_color, buf):
    """Extract global (non-spatial) features from BitboardState.

    Args:
        state: BitboardState
        p0_color: Color of current player
        buf: Pre-zeroed numpy array of shape [76] (float32)

    Feature layout (76 total):
        [0-1]   P0: actual_vps, has_played_dev
        [2-6]   P0: resources in hand (wood, brick, sheep, wheat, ore)
        [7-11]  P0: dev cards in hand (knight, yop, mono, rb, vp)
        [12-51] P0-P3 × 10: public_vps, has_army, has_road, roads_left,
                settlements_left, cities_left, has_rolled, longest_road,
                num_resources, num_devs
        [52-67] P0-P3 × 4: dev_played (knight, yop, mono, rb)
        [68-73] Bank: dev_cards_left, wood, brick, sheep, wheat, ore
        [74-75] Game: is_moving_robber, is_discarding
    """
    p0_idx = state.color_to_index[p0_color]
    n = state.num_players
    p0_ps = state.player_state[p0_idx]

    # P0 specific (0-11)
    buf[0] = float(p0_ps[PS_ACTUAL_VP])
    buf[1] = float(p0_ps[PS_HAS_PLAYED_DEV])
    for ri in range(5):
        buf[2 + ri] = float(p0_ps[PS_RESOURCE_START + ri])
    for di in range(5):
        buf[7 + di] = float(p0_ps[_PS_DEV_HAND[di]])

    # Per-player stats (12-51): 4 players × 10 features
    for slot in range(min(n, 4)):
        pidx = (p0_idx + slot) % n
        ps = state.player_state[pidx]
        base = 12 + slot * 10
        buf[base + 0] = float(ps[PS_VP])
        buf[base + 1] = float(ps[PS_HAS_ARMY])
        buf[base + 2] = float(ps[PS_HAS_ROAD])
        buf[base + 3] = float(ps[PS_ROADS_AVAIL])
        buf[base + 4] = float(ps[PS_SETTLE_AVAIL])
        buf[base + 5] = float(ps[PS_CITY_AVAIL])
        buf[base + 6] = float(ps[PS_HAS_ROLLED])
        buf[base + 7] = float(ps[PS_LONGEST_ROAD])
        buf[base + 8] = float(int(np.sum(ps[PS_RESOURCE_START:PS_RESOURCE_END])))
        buf[base + 9] = float(
            int(ps[PS_KNIGHT_HAND]) + int(ps[PS_YOP_HAND]) +
            int(ps[PS_MONO_HAND]) + int(ps[PS_RB_HAND]) + int(ps[PS_VP_HAND])
        )

    # Per-player dev played (52-67): 4 players × 4 dev types
    for slot in range(min(n, 4)):
        pidx = (p0_idx + slot) % n
        ps = state.player_state[pidx]
        base = 52 + slot * 4
        for di in range(4):
            buf[base + di] = float(ps[_PS_DEV_PLAYED[di]])

    # Bank (68-73)
    buf[68] = float(25 - state.dev_deck_idx)
    for ri in range(5):
        buf[69 + ri] = float(state.bank[ri])

    # Game state (74-75)
    buf[74] = 1.0 if state.current_prompt == PROMPT_MOVE_ROBBER else 0.0
    buf[75] = 1.0 if state.current_prompt == PROMPT_DISCARD else 0.0


# ---------------------------------------------------------------------------
# GNN Architecture
# ---------------------------------------------------------------------------

class CatanGNNet(nn.Module):
    """Graph Neural Network for Catan board evaluation.

    Processes per-node features through message passing on the fixed board graph,
    merges with global (non-spatial) features, and outputs value + policy.
    """

    def __init__(self, node_feat_dim=NODE_FEAT_DIM, global_feat_dim=GLOBAL_FEAT_DIM,
                 num_actions=ACTION_SPACE_SIZE,
                 gnn_dim=32, global_dim=64, body_dim=96,
                 gnn_layers=2, dropout=0.2, edge_dropout=0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.global_feat_dim = global_feat_dim
        self.num_actions = num_actions
        self.gnn_dim = gnn_dim
        self.global_dim = global_dim
        self.body_dim = body_dim
        self.gnn_layers = gnn_layers
        self.dropout_rate = dropout
        self.edge_dropout_rate = edge_dropout

        # Adjacency matrix (constant, moves with .to(device))
        A_norm = build_adjacency_matrix()
        self.register_buffer('adj_norm', torch.tensor(A_norm, dtype=torch.float32))

        # Node projection
        self.node_proj = nn.Linear(node_feat_dim, gnn_dim)

        # SAGEConv layers (hand-rolled)
        self.conv_self = nn.ModuleList()
        self.conv_neigh = nn.ModuleList()
        self.conv_bias = nn.ParameterList()
        for _ in range(gnn_layers):
            self.conv_self.append(nn.Linear(gnn_dim, gnn_dim, bias=False))
            self.conv_neigh.append(nn.Linear(gnn_dim, gnn_dim, bias=False))
            self.conv_bias.append(nn.Parameter(torch.zeros(gnn_dim)))

        self.gnn_dropout = nn.Dropout(dropout)

        # Global projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, global_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Merged body
        merge_dim = gnn_dim + global_dim
        self.body = nn.Sequential(
            nn.Linear(merge_dim, body_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Value head: body_dim -> head_hidden -> 1 (tanh)
        head_hidden = max(48, body_dim // 2)
        self.value_head = nn.Sequential(
            nn.Linear(body_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
            nn.Tanh(),
        )

        # Policy head: body_dim -> head_hidden -> num_actions (logits)
        self.policy_head = nn.Sequential(
            nn.Linear(body_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
        )

    def forward(self, node_x, global_x, action_mask=None):
        """Forward pass.

        Args:
            node_x: [B, 54, node_feat_dim] per-node features
            global_x: [B, global_feat_dim] global features
            action_mask: [B, num_actions] bool tensor (True = legal)

        Returns:
            value: [B, 1] in [-1, +1]
            policy_logits: [B, num_actions] masked logits
        """
        # --- GNN path ---
        h = F.relu(self.node_proj(node_x))  # [B, 54, gnn_dim]

        # Edge dropout during training (DropEdge regularization)
        adj = self.adj_norm
        if self.training and self.edge_dropout_rate > 0:
            mask = torch.rand_like(adj) > self.edge_dropout_rate
            adj = adj * mask
            # Re-normalize rows
            row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
            adj = adj / row_sum

        for i in range(self.gnn_layers):
            # SAGEConv: h_new = W_self(h) + W_neigh(A_norm @ h) + bias
            h_agg = torch.matmul(adj, h)  # [B, 54, gnn_dim]
            h_new = self.conv_self[i](h) + self.conv_neigh[i](h_agg) + self.conv_bias[i]
            h = F.relu(h_new) + h  # residual connection
            h = self.gnn_dropout(h)

        # Mean pooling over nodes → graph embedding
        gnn_out = h.mean(dim=1)  # [B, gnn_dim]

        # --- Global path ---
        global_out = self.global_proj(global_x)  # [B, global_dim]

        # --- Merge ---
        merged = torch.cat([gnn_out, global_out], dim=-1)  # [B, gnn_dim + global_dim]
        body_out = self.body(merged)  # [B, body_dim]

        # --- Heads ---
        value = self.value_head(body_out)
        logits = self.policy_head(body_out)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return value, logits

    def predict(self, node_x, global_x, action_mask=None):
        """Inference: returns value scalar and policy probability vector.

        Args:
            node_x: [B, 54, node_feat_dim]
            global_x: [B, global_feat_dim]
            action_mask: [B, num_actions] bool tensor

        Returns:
            value: [B,] float tensor in [-1, +1]
            policy: [B, num_actions] probability tensor
        """
        value, logits = self.forward(node_x, global_x, action_mask)
        policy = F.softmax(logits, dim=-1)
        return value.squeeze(-1), policy


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint_gnn(net, optimizer, iteration, node_feat_means, node_feat_stds,
                        global_feat_means, global_feat_stds, path, extra=None):
    """Save GNN checkpoint with all metadata."""
    data = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'iteration': iteration,
        'is_gnn': True,
        'node_feat_dim': net.node_feat_dim,
        'global_feat_dim': net.global_feat_dim,
        'num_actions': net.num_actions,
        'gnn_dim': net.gnn_dim,
        'global_dim': net.global_dim,
        'body_dim': net.body_dim,
        'gnn_layers': net.gnn_layers,
        'dropout': net.dropout_rate,
        'edge_dropout': net.edge_dropout_rate,
        'node_feat_means': node_feat_means,
        'node_feat_stds': node_feat_stds,
        'global_feat_means': global_feat_means,
        'global_feat_stds': global_feat_stds,
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint_gnn(path, device='cpu'):
    """Load GNN checkpoint, returning model and metadata."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    net = CatanGNNet(
        node_feat_dim=ckpt.get('node_feat_dim', NODE_FEAT_DIM),
        global_feat_dim=ckpt.get('global_feat_dim', GLOBAL_FEAT_DIM),
        num_actions=ckpt.get('num_actions', ACTION_SPACE_SIZE),
        gnn_dim=ckpt.get('gnn_dim', 32),
        global_dim=ckpt.get('global_dim', 64),
        body_dim=ckpt.get('body_dim', 96),
        gnn_layers=ckpt.get('gnn_layers', 2),
        dropout=ckpt.get('dropout', 0.2),
        edge_dropout=ckpt.get('edge_dropout', 0.1),
    )

    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net, ckpt
