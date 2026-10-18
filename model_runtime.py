"""Exact inference architecture from TFC_HAR_SGR_Train/train_export.py.

This is the *original Transformer backbone*, not DINOv3.
The checkpoint must contain encoder_state_dict + classifier_state_dict.
Only sanitized locally-generated checkpoints are accepted (weights_only=True).
"""
from pathlib import Path
import torch
from torch import nn

LABELS = ["typing", "teeth", "drinking", "writing", "clapping", "folding"]
TRANSLATIONS = {
    "typing": "Gõ phím",
    "teeth": "Đánh răng",
    "drinking": "Uống nước",
    "writing": "Viết chữ",
    "clapping": "Vỗ tay",
    "folding": "Gấp đồ",
}


class TFC(nn.Module):
    """Two 2-layer Transformer encoders + two projections, B x 3 x 206."""

    def __init__(self, length=206):
        super().__init__()
        self.transformer_encoder_t = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=length, nhead=2,
                                       dim_feedforward=length * 2,
                                       batch_first=True),
            num_layers=2,
        )
        self.projector_t = nn.Sequential(
            nn.Linear(3 * length, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Linear(256, 128),
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=length, nhead=2,
                                       dim_feedforward=length * 2,
                                       batch_first=True),
            num_layers=2,
        )
        self.projector_f = nn.Sequential(
            nn.Linear(3 * length, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Linear(256, 128),
        )

    @staticmethod
    def _branch(x, encoder, projector):
        b, c, length = x.shape
        h = encoder(x)
        return projector(h.reshape(b, c * length))

    def forward(self, x):
        freq = torch.fft.fft(x, dim=-1).abs()
        time_embedding = self._branch(x, self.transformer_encoder_t, self.projector_t)
        freq_embedding = self._branch(freq, self.transformer_encoder_f, self.projector_f)
        return time_embedding, freq_embedding


class Classifier(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.logits = nn.Linear(256, 64)
        self.logits_simple = nn.Linear(64, classes)

    def forward(self, zt, zf):
        return self.logits_simple(torch.sigmoid(self.logits(torch.cat([zt, zf], dim=1))))


def load_model(checkpoint_path):
    """Load only simple-tensor checkpoint created by prepare_checkpoint.py."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Chưa tìm thấy model: {p}. Xem README: Bước chuẩn bị checkpoint.")
    ck = torch.load(str(p), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "encoder_state_dict" not in ck or "classifier_state_dict" not in ck:
        raise ValueError("Checkpoint cần có encoder_state_dict và classifier_state_dict.")
    config = ck.get("config", {})
    if int(config.get("length", 206)) != 206 or int(config.get("channels", 3)) != 3:
        raise ValueError("Checkpoint không dùng đầu vào 3 kênh x 206 điểm.")
    labels = list(config.get("labels", LABELS))
    if labels != LABELS:
        raise ValueError("Thứ tự nhãn trong checkpoint không khớp 6 nhãn SGR của ứng dụng.")
    encoder, classifier = TFC(), Classifier()
    encoder.load_state_dict(ck["encoder_state_dict"], strict=True)
    classifier.load_state_dict(ck["classifier_state_dict"], strict=True)
    encoder.eval()
    classifier.eval()
    return encoder, classifier, config


def predict(encoder, classifier, samples, batch_size=16):
    """Return probabilities [N, 6] for float32 samples [N,3,206]. CPU only."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (3, 206):
        raise ValueError(f"Đầu vào phải có [N,3,206], nhận được {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Đầu vào chứa NaN hoặc Inf.")
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            t = torch.from_numpy(x[start:start + batch_size])
            zt, zf = encoder(t)
            values.append(torch.softmax(classifier(zt, zf), dim=1).cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, 6), dtype=np.float32)
