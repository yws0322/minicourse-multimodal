"""Early / intermediate / late fusion architectures for WSI + MRI + clinical survival
prediction, plus the shared building blocks (MIL pooling, task heads, survival loss/metric)
all three reuse.

Developed and explained step-by-step in `notebooks/03_fusion_models.ipynb` (including the
design reasoning behind each choice below) -- this module is the finalized version, imported
by `notebooks/04_train_and_evaluate.ipynb` for real training. Per this project's convention,
the *architectures themselves* are the lesson, so treat this file as the source of truth, not
something to read cold; the notebook is where the reasoning lives.
"""
import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ABMIL(nn.Module):
    """Attention-based MIL pooling: turns a bag of N patch embeddings into one vector,
    weighting each patch by a learned attention score instead of averaging them uniformly.
    Used identically by all three fusion strategies below, so WSI pooling quality is held
    constant and the comparison isolates the fusion strategy itself, not the pooling method."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) patch embeddings for one WSI -> (D,) bag-level embedding."""
        attn_scores = self.attention(x)               # (N, 1)
        attn_weights = F.softmax(attn_scores, dim=0)  # (N, 1)
        return (attn_weights * x).sum(dim=0)          # (D,)


class PredictionHead(nn.Module):
    """Shared head architecture for *both* tasks -- only the output dimension differs (one
    logit for classification, one hazard logit per time bin for survival). Every fusion
    strategy builds its head through `build_task_head`, so the head's layers/capacity are held
    identical everywhere a comparison is being made (fusion strategy, or task) -- only the
    backbone feeding into it differs. Matches HIMF-Surv's own `MLPPredictionHead`: a 3-layer
    MLP (input -> 64 -> 32 -> output), BatchNorm + ReLU after the first two layers."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_task_head(input_dim: int, task: str, num_time_bins: int = 15) -> nn.Module:
    """Classification -> one logit. Survival -> one hazard logit per time bin. Same
    `PredictionHead` architecture either way -- only `output_dim` changes."""
    if task == "classification":
        return PredictionHead(input_dim, output_dim=1)
    elif task == "survival":
        return PredictionHead(input_dim, output_dim=num_time_bins)
    raise ValueError(f"Unknown task: {task!r} (expected 'classification' or 'survival')")


# ---------------------------------------------------------------------------
# Survival loss + metric (classification just uses BCEWithLogitsLoss + AUROC directly)
# ---------------------------------------------------------------------------

def discretize_time(time: torch.Tensor, num_bins: int, max_time: float, device: str) -> torch.Tensor:
    """Bucket continuous follow-up time into `num_bins` equal-width bins."""
    time = torch.as_tensor(time, dtype=torch.float32, device=device)
    bins = torch.linspace(0, max_time, num_bins + 1, device=device)
    discretized = torch.bucketize(time, bins, right=True) - 1
    return torch.clamp(discretized, 0, num_bins - 1)


class NLLLoss(nn.Module):
    """Discrete-time negative log-likelihood survival loss (as in HIMF-Surv):
    turns per-bin hazard logits into a survival curve, and scores the observed
    time bin's likelihood -- the event-hazard term if BCR occurred, the
    still-surviving term if the patient was censored."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_time: torch.Tensor, y_event: torch.Tensor) -> torch.Tensor:
        y_time = y_time.long().unsqueeze(1)
        y_event = y_event.long().unsqueeze(1)
        num_bins = logits.shape[1]
        y_time = torch.clamp(y_time, 0, num_bins - 1)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        survival_padded = torch.cat([torch.ones_like(y_event, dtype=torch.float), survival], dim=1)

        s_prev = torch.gather(survival_padded, 1, y_time).clamp(min=1e-7)
        h_this = torch.gather(hazards, 1, y_time).clamp(min=1e-7)
        log_lik_event = torch.log(s_prev) + torch.log(h_this)

        y_time_next = torch.clamp(y_time + 1, 0, num_bins)
        log_lik_censored = torch.log(torch.gather(survival_padded, 1, y_time_next).clamp(min=1e-7))

        neg_log_lik = -(y_event * log_lik_event + (1 - y_event) * log_lik_censored)
        return neg_log_lik.mean() if self.reduction == "mean" else neg_log_lik.sum()


def concordance_index(event_times, predicted_risk, event_observed) -> float:
    """Fraction of comparable (patient i had the earlier event) pairs the model
    ranks correctly by predicted risk. 0.5 = random, 1.0 = perfect."""
    if isinstance(event_times, torch.Tensor):
        event_times = event_times.detach().cpu().numpy()
    if isinstance(predicted_risk, torch.Tensor):
        predicted_risk = predicted_risk.detach().cpu().numpy()
    if isinstance(event_observed, torch.Tensor):
        event_observed = event_observed.detach().cpu().numpy()

    n = len(event_times)
    concordant, permissible = 0.0, 0
    for i in range(n):
        if event_observed[i] == 0:
            continue
        for j in range(n):
            if i == j or event_times[i] >= event_times[j]:
                continue
            permissible += 1
            if predicted_risk[i] > predicted_risk[j]:
                concordant += 1
            elif predicted_risk[i] == predicted_risk[j]:
                concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5


# ---------------------------------------------------------------------------
# Three fusion strategies -- all three pool WSI patches with the *same* ABMIL module and build
# their task head through the *same* `build_task_head`, so the comparison isolates the fusion
# strategy, not the pooling method or head capacity.
# ---------------------------------------------------------------------------

class EarlyFusionModel(nn.Module):
    """Concatenate modality vectors *before* any joint modeling, then let a single shared
    trunk do all the work. No per-modality projection layers -- that would start to look like
    intermediate fusion's per-modality encoders. Two parameter-free (not learned, not
    modality-specific) normalization steps keep this practical without compromising "early":
    each modality vector is L2-normalized before concatenation (otherwise wsi(1536) and
    mri(2048) dwarf clinical(22) by sheer feature count), then the concatenated vector is
    BatchNorm'd (standard per-feature normalization, same purpose as the BatchNorm inside
    PredictionHead but applied to the raw multimodal input instead of a hidden layer)."""
    def __init__(self, wsi_dim=1536, mri_dim=2048, clinical_dim=22, hidden_dim=256,
                 task="classification", num_time_bins=15):
        super().__init__()
        self.abmil = ABMIL(wsi_dim, hidden_dim=128)

        fused_dim = wsi_dim + mri_dim + clinical_dim
        self.input_norm = nn.BatchNorm1d(fused_dim)
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.head = build_task_head(hidden_dim // 2, task, num_time_bins)

    def forward(self, batch: dict) -> torch.Tensor:
        wsi_pooled = torch.stack([self.abmil(w) for w in batch["wsi"]])  # (B, wsi_dim)

        wsi_pooled = F.normalize(wsi_pooled, p=2, dim=1)
        mri = F.normalize(batch["mri"], p=2, dim=1)
        clinical = F.normalize(batch["clinical"], p=2, dim=1)

        fused = torch.cat([wsi_pooled, mri, clinical], dim=1)
        fused = self.input_norm(fused)
        return self.head(self.trunk(fused))


class IntermediateFusionModel(nn.Module):
    """Each modality gets its own projection into a shared embedding space, then a *single*
    self-attention pass lets the three modality tokens attend to each other once before mean
    pooling. This is HIMF-Surv's own approach (per-modality projection + attention across
    modalities), minus the layer-wise aggregation this course skips *and* minus the deep
    multi-layer transformer stack -- one round of cross-modal attention is enough to
    demonstrate what "intermediate" fusion means. `shared_dim=256` (vs. HIMF-Surv's 1536) is a
    deliberate choice for this course's much smaller cohort: at 1536, the attention
    in/out-projections alone would put this model over 10x the parameter count of early/late
    fusion, undermining a fair comparison and inviting overfitting on ~95 patients."""
    def __init__(self, wsi_dim=1536, mri_dim=2048, clinical_dim=22, shared_dim=256,
                 task="classification", num_time_bins=15):
        super().__init__()
        self.abmil = ABMIL(wsi_dim, hidden_dim=128)
        self.proj_wsi = nn.Linear(wsi_dim, shared_dim)
        self.proj_mri = nn.Linear(mri_dim, shared_dim)
        self.proj_clinical = nn.Linear(clinical_dim, shared_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(shared_dim)
        self.head = build_task_head(shared_dim, task, num_time_bins)

    def forward(self, batch: dict) -> torch.Tensor:
        wsi_pooled = torch.stack([self.abmil(w) for w in batch["wsi"]])  # (B, wsi_dim)
        tokens = torch.stack([
            self.proj_wsi(wsi_pooled),
            self.proj_mri(batch["mri"]),
            self.proj_clinical(batch["clinical"]),
        ], dim=1)  # (B, 3, shared_dim)

        attn_out, _ = self.self_attn(tokens, tokens, tokens)  # one round of cross-modal attention
        tokens = self.attn_norm(tokens + attn_out)            # residual + norm
        fused = tokens.mean(dim=1)                             # (B, shared_dim)
        return self.head(fused)


class LateFusionModel(nn.Module):
    """Each modality gets a fully independent prediction head -- they never share features or
    an intermediate representation. WSI still needs ABMIL to turn its patch bag into a single
    vector (that's an unavoidable data-shape step, not a fusion choice), but there's no extra
    branch MLP after it -- pooled WSI, raw MRI, and raw clinical vectors go straight into their
    own `PredictionHead` (itself already a 3-layer MLP, so there's no missing capacity). Only
    the three independent predictions are combined at the very end, via a learned
    per-modality weight."""
    def __init__(self, wsi_dim=1536, mri_dim=2048, clinical_dim=22,
                 task="classification", num_time_bins=15):
        super().__init__()
        self.abmil = ABMIL(wsi_dim, hidden_dim=128)

        self.wsi_head = build_task_head(wsi_dim, task, num_time_bins)
        self.mri_head = build_task_head(mri_dim, task, num_time_bins)
        self.clinical_head = build_task_head(clinical_dim, task, num_time_bins)

        self.modality_logits = nn.Parameter(torch.zeros(3))  # learned combination weights

    def forward(self, batch: dict) -> torch.Tensor:
        wsi_pooled = torch.stack([self.abmil(w) for w in batch["wsi"]])

        out_wsi = self.wsi_head(wsi_pooled)
        out_mri = self.mri_head(batch["mri"])
        out_clinical = self.clinical_head(batch["clinical"])

        weights = F.softmax(self.modality_logits, dim=0)
        return weights[0] * out_wsi + weights[1] * out_mri + weights[2] * out_clinical
