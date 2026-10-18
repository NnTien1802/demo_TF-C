"""Convert a trusted local training checkpoint into a weights_only-safe deployment file.

Run on YOUR machine. Only ever use weights_only=False for your own checkpoint.
This command is NOT invoked by the public Streamlit app.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from model_runtime import load_model


def plain(obj):
    """Explicitly convert numpy scalar metadata to safe Python primitives."""
    # Check np.generic FIRST: on some NumPy versions np.float64 is also a float.
    if isinstance(obj, np.generic):
        return plain(obj.item())
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [plain(v) for v in obj]
    raise TypeError(f"Không hỗ trợ kiểu metadata {type(obj).__name__}")


def main():
    ap = argparse.ArgumentParser(description='Xuất checkpoint TF-C SGR dạng weights_only-safe.')
    ap.add_argument('--source', required=True, help='File .pt được tạo bởi train_export.py trên máy của bạn')
    ap.add_argument('--output', default='models/tfc_har_to_sgr_safe.pt')
    args = ap.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f'Không tìm thấy checkpoint: {source}')
    if source == output:
        raise SystemExit('Nguồn và đích không được trùng nhau.')
    print('Đang đọc checkpoint TỰ HUẤN LUYỆN từ:', source)
    print('Lưu ý: weights_only=False chỉ an toàn với file bạn tin cậy.')
    raw = torch.load(str(source), map_location='cpu', weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError('Checkpoint phải là dictionary.')
    keep = {k: raw[k] for k in ('encoder_state_dict', 'classifier_state_dict', 'config', 'epoch', 'val_macro_f1') if k in raw}
    if 'encoder_state_dict' not in keep or 'classifier_state_dict' not in keep:
        raise ValueError('Thiếu trọng số encoder hoặc classifier.')
    safe = plain(keep)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(safe, str(output))
    enc, clf, config = load_model(output)
    print('Đã tạo:', output)
    print('Kích thước:', round(output.stat().st_size / 1024 / 1024, 2), 'MB')
    print('Load test an toàn: SUCCESS; labels=', config.get('labels'))


if __name__ == '__main__':
    main()
