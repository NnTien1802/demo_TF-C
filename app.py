"""GestureLab 2.0 — TF-C SGR Streamlit demo (UI-only update).

Model architecture, trained weights, preprocessing, and predictions are intentionally
unchanged from the previous release. Uploaded pickle/PyTorch files are never read.
"""
from html import escape
from pathlib import Path
import hashlib
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_io import SensorData, csv_bytes, npz_bytes, zip_bytes
from model_runtime import LABELS, TRANSLATIONS, load_model, predict

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tfc_har_to_sgr_safe.pt"
EXAMPLES_PATH = ROOT / "samples" / "sgr_examples.npz"
RESULTS_PATH = ROOT / "models" / "training_results.json"

st.set_page_config(
    page_title="GestureLab | TF-C SGR",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Presentation only: inference and data loading remain in Python. */
.block-container {max-width: 1210px; padding-top: 1.6rem; padding-bottom: 3.5rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(132,151,185,.15);}
[data-testid="stSidebar"] .stMarkdown p {font-size: .9rem;}
.gl-hero {
  padding: 1.38rem 1.6rem; border: 1px solid rgba(126,150,219,.24);
  border-radius: 18px; background: linear-gradient(116deg,rgba(61,91,171,.18),rgba(35,133,126,.09));
  margin: 0 0 1rem;
}
.gl-eyebrow {font-size:.72rem; font-weight:800; letter-spacing:.15em; color:#8caafc;}
.gl-hero h1 {margin:.35rem 0 .45rem; font-size:clamp(1.45rem,2.3vw,2.05rem);line-height:1.25;}
.gl-hero p {margin:0; opacity:.80; font-size:.92rem; line-height:1.5;}
.gl-section {font-size:.76rem; letter-spacing:.11em; font-weight:800; color:#8a9dcc; margin:1rem 0 .45rem;}
.gl-tip {border:1px solid rgba(136,152,188,.24);border-radius:12px;padding:.8rem 1rem;
 background:rgba(83,114,158,.065);font-size:.83rem;line-height:1.55;}
.gl-result {border:1px solid rgba(125,150,202,.28);border-radius:16px;
 padding:1.05rem 1.1rem;background:linear-gradient(125deg,rgba(66,93,171,.14),rgba(34,123,119,.065));margin:.8rem 0 .5rem;}
.gl-result .gl-kicker {font-size:.74rem; opacity:.7;font-weight:700;letter-spacing:.05em;}
.gl-result .gl-main {font-size:1.6rem;font-weight:800;line-height:1.25;margin:.25rem 0;}
.gl-result .gl-score {font-weight:800;font-size:1.05rem;color:#a9bcff;}
.gl-result .gl-note {font-size:.82rem;opacity:.86;margin-top:.45rem;line-height:1.45;}
.gl-status {display:inline-block;border-radius:100px;padding:.24rem .65rem;font-size:.75rem;font-weight:700;
 margin-top:.4rem;background:rgba(131,148,172,.18);color:#d3deef;}
.gl-status.good {background:rgba(29,143,115,.17);color:#72d8bd;}
.gl-status.bad {background:rgba(213,126,75,.17);color:#ffc38c;}
.gl-focus {border:1px solid rgba(122,146,187,.22);border-radius:13px;padding:.8rem 1rem;
 background:rgba(79,107,153,.07);}
.gl-focus-label {font-size:.78rem;opacity:.68;}
.gl-focus-value {font-size:1.4rem;font-weight:800;line-height:1.4;}
.gl-empty {padding:1.3rem 1rem;border:1px dashed rgba(130,152,190,.35);border-radius:14px;
 text-align:center;opacity:.8;line-height:1.6;}
/* Added: compact signal analysis below the 3-axis chart. */
.gl-signal-heading {font-size:.75rem;letter-spacing:.08em;font-weight:800;
 color:#A6B9E4;margin:.95rem 0 .6rem;}
.gl-signal-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6rem;}
.gl-signal-card {min-width:0;border:1px solid rgba(133,153,194,.23);
 border-radius:12px;background:rgba(74,105,161,.10);padding:.85rem .75rem;}
.gl-signal-label {font-size:.74rem;color:#B4C1D6;line-height:1.25;}
.gl-signal-value {font-size:1.35rem;font-weight:800;color:#EDF4FF;
 margin:.25rem 0 .15rem;line-height:1.3;overflow-wrap:anywhere;}
.gl-signal-help {font-size:.71rem;line-height:1.35;color:#94A7C1;}
.gl-signal-note {margin-top:.7rem;padding:.85rem .95rem;
 border-radius:12px;border:1px solid rgba(130,154,194,.18);
 background:rgba(44,63,97,.19);font-size:.81rem;line-height:1.55;color:#D2DDF1;}
.gl-signal-note strong {color:#AFC7FF;}
@media(max-width:450px) {
 .gl-signal-grid {grid-template-columns:repeat(3,minmax(0,1fr));gap:.3rem;}
 .gl-signal-card {padding:.65rem .45rem;}
 .gl-signal-value {font-size:1.04rem;}
 .gl-signal-help {font-size:.64rem;}
}
/* Keep form elements and button easily tappable. */
.stButton > button[kind="primary"] {border-radius:10px;font-weight:800;min-height:2.9rem;}
[data-testid="stMetricValue"] {font-size:1.35rem;}
@media(max-width:750px) {
 .block-container {padding-top:.9rem;padding-left:1rem;padding-right:1rem;}
 .gl-hero {padding:1.05rem 1.1rem;}
 .gl-result .gl-main {font-size:1.35rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang nạp mô hình TF-C trên CPU…")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_examples():
    if not EXAMPLES_PATH.is_file():
        return None
    with np.load(str(EXAMPLES_PATH), allow_pickle=False) as src:
        return {
            "samples": src["samples"].astype(np.float32),
            "labels": src["labels"].astype(int),
            "indices": src["indices"].astype(int),
        }


def signal_frame(data, index):
    axis = np.asarray(data.time[index], dtype=np.float64)
    axis = axis - axis[0] if axis.size else axis
    return pd.DataFrame({
        "X": data.plot[index, :, 0],
        "Y": data.plot[index, :, 1],
        "Z": data.plot[index, :, 2],
    }, index=axis)


def signal_statistics(data, index):
    """Describe the DISPLAYED 3-axis window, not the standardized model input.

    Returns descriptive statistics only; it does not classify human activity.
    RMS and peak refer to vector magnitude |a| = sqrt(x^2 + y^2 + z^2).
    For standardized SGR samples all quantities are dimensionless.
    """
    xyz = np.asarray(data.plot[index], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("Cửa sổ tín hiệu phải có dạng [số điểm, 3 trục].")
    if not np.isfinite(xyz).all():
        raise ValueError("Cửa sổ tín hiệu chứa NaN hoặc Inf.")

    magnitude = np.linalg.norm(xyz, axis=1)
    halfway = max(1, len(magnitude) // 2)
    early_std = float(np.std(magnitude[:halfway]))
    late_std = float(np.std(magnitude[halfway:])) if halfway < len(magnitude) else early_std
    epsilon = 1e-10

    if early_std > 1.4 * max(late_std, epsilon):
        observation = "Độ biến động lớn hơn ở nửa đầu cửa sổ."
    elif late_std > 1.4 * max(early_std, epsilon):
        observation = "Độ biến động lớn hơn ở nửa cuối cửa sổ."
    else:
        observation = "Mức biến động ở hai nửa cửa sổ tương đối gần nhau."

    peak_at = int(np.argmax(magnitude)) + 1
    return {
        "std": float(np.std(magnitude)),
        "peak": float(np.max(magnitude)),
        "rms": float(np.sqrt(np.mean(np.square(magnitude)))),
        "peak_at": peak_at,
        "count": int(len(magnitude)),
        "observation": observation,
    }


def show_signal_analysis(data, index, is_normalized):
    """Present window-dependent descriptive statistics beneath the line chart."""
    stats = signal_statistics(data, index)
    units = "đơn vị chuẩn hóa" if is_normalized else "đơn vị theo tệp gốc"
    cards = [
        ("Độ biến động", stats["std"], "Std của |a|"),
        ("Biên độ đỉnh", stats["peak"], "Max của |a|"),
        ("Mức RMS", stats["rms"], "RMS của |a|"),
    ]
    html_cards = "".join(
        '<div class="gl-signal-card">'
        '<div class="gl-signal-label">' + escape(label) + '</div>'
        '<div class="gl-signal-value">' + f'{value:.2f}' + '</div>'
        '<div class="gl-signal-help">' + escape(detail) + '</div></div>'
        for label, value, detail in cards
    )
    st.markdown(
        '<div class="gl-signal-heading">PHÂN TÍCH TÍN HIỆU · CỬA SỔ '
        + str(index + 1) + '/' + str(len(data.samples)) + '</div>'
        '<div class="gl-signal-grid">' + html_cards + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="gl-signal-note"><strong>Nhận xét:</strong> '
        + escape(stats["observation"])
        + ' Biên độ vector lớn nhất xuất hiện tại điểm '
        + str(stats["peak_at"]) + '/' + str(stats["count"])
        + '. <span style="opacity:.78">(' + escape(units) + ')</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Std, Peak và RMS được tính từ độ lớn vector |a| = √(X² + Y² + Z²) của "
        "cửa sổ đang hiển thị. Chúng chỉ mô tả tín hiệu, không phải dự đoán cử chỉ."
    )


def data_signature(source_name, fingerprint, is_raw):
    return (source_name, fingerprint, "raw" if is_raw else "normalized")


def truth_for_window(data, focus):
    if data.labels is None or len(data.labels) != len(data.samples):
        return None
    try:
        truth = int(data.labels[focus])
    except (TypeError, ValueError):
        return None
    return truth if 0 <= truth < len(LABELS) else None


def compact_result(probs, index, data):
    pred = int(np.argmax(probs[index]))
    score = float(probs[index, pred] * 100)
    label = escape(TRANSLATIONS[LABELS[pred]])
    truth = truth_for_window(data, index)
    if truth is None:
        state = '<span class="gl-status">Không có nhãn đối chiếu</span>'
        note = "Dữ liệu tự tải lên không chứa nhãn thật của mẫu."
    elif truth == pred:
        state = '<span class="gl-status good">✓ Dự đoán khớp nhãn</span>'
        note = f'Nhãn tham chiếu: {escape(TRANSLATIONS[LABELS[truth]])}'
    else:
        state = '<span class="gl-status bad">! Dự đoán chưa khớp</span>'
        note = f'Nhãn tham chiếu: {escape(TRANSLATIONS[LABELS[truth]])}'
    st.markdown(
        '<div class="gl-result">'
        '<div class="gl-kicker">KẾT QUẢ NHANH · CỬA SỔ ' + str(index + 1) + '</div>'
        '<div class="gl-main">' + label + '</div>'
        '<div class="gl-score">' + f'{score:.1f}%' + ' <span style="font-size:.73rem;opacity:.7;font-weight:500">điểm Softmax</span></div>'
        + state + '<div class="gl-note">' + note + '</div></div>',
        unsafe_allow_html=True,
    )


def probability_chart(probs, predicted_index):
    names = [TRANSLATIONS[name] for name in LABELS]
    df = pd.DataFrame({
        "Cử chỉ": names,
        "Điểm Softmax (%)": np.asarray(probs, dtype=float) * 100,
        "Đang dự đoán": [i == predicted_index for i in range(len(LABELS))],
    })
    df["Hiển thị"] = df["Điểm Softmax (%)"].map(lambda x: f"{x:.1f}%")
    y = alt.Y("Cử chỉ:N", sort=names, title=None, axis=alt.Axis(labelFontSize=13, labelLimit=170))
    x = alt.X(
        "Điểm Softmax (%):Q", scale=alt.Scale(domain=[0, 112]),
        axis=alt.Axis(title=None, values=[0, 20, 40, 60, 80, 100], labelExpr="datum.value + '%'"),
    )
    base = alt.Chart(df).encode(y=y, x=x)
    bars = base.mark_bar(size=23, cornerRadiusEnd=5).encode(
        color=alt.condition("datum['Đang dự đoán']", alt.value("#778DFF"), alt.value("#66798F")),
        tooltip=["Cử chỉ:N", alt.Tooltip("Điểm Softmax (%):Q", format=".2f")],
    )
    labels = base.mark_text(align="left", baseline="middle", dx=6, fontSize=12).encode(
        text="Hiển thị:N", color=alt.value("#D9E3F4")
    )
    chart = (bars + labels).properties(height=245).configure_view(strokeWidth=0).configure_axis(
        gridColor="#39465C", gridOpacity=.35, labelColor="#AFC0D6"
    )
    st.altair_chart(chart, use_container_width=True)


def show_prediction_table(probs, data):
    predicted = np.argmax(probs, axis=1)
    rows = []
    for i, p in enumerate(predicted):
        p = int(p)
        row = {
            "window": i + 1,
            "prediction": LABELS[p],
            "prediction_vi": TRANSLATIONS[LABELS[p]],
            "softmax_score": round(float(probs[i, p]), 6),
        }
        truth = truth_for_window(data, i)
        if truth is not None:
            row["true_label"] = LABELS[truth]
            row["correct"] = bool(truth == p)
        for j, name in enumerate(LABELS):
            row["prob_" + name] = round(float(probs[i, j]), 6)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(rows) > 1:
        with st.expander(f"Xem kết quả của {len(rows)} cửa sổ", expanded=False):
            columns = [c for c in ("window", "prediction_vi", "softmax_score", "true_label", "correct") if c in frame]
            st.dataframe(frame[columns], hide_index=True, use_container_width=True)
    st.download_button(
        "↓ Tải bảng dự đoán CSV", frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="tfc_sgr_predictions.csv", mime="text/csv", use_container_width=False,
    )


# Sidebar: keep static information short so the main task dominates.
with st.sidebar:
    st.markdown("### 〰️ GestureLab")
    st.caption("TF-C · Human Gesture Recognition")
    st.divider()
    if MODEL_PATH.is_file():
        st.success("Mô hình đã sẵn sàng")
    else:
        st.error("Chưa tìm thấy checkpoint")
        st.caption("Chuyển checkpoint theo README, đặt tại models/tfc_har_to_sgr_safe.pt.")
    st.markdown("**Mô hình**")
    st.caption("TF-C Transformer gốc · HAR → SGR · 6 lớp · CPU")
    with st.expander("6 cử chỉ được hỗ trợ", expanded=False):
        for name in LABELS:
            st.write("• " + TRANSLATIONS[name] + " / " + name)
    st.divider()
    st.caption("Bản demo nghiên cứu. Không sử dụng DINOv3 trong nhánh triển khai.")

st.markdown(
    '<div class="gl-hero"><div class="gl-eyebrow">GESTURELAB / TF-C DEMO</div>'
    '<h1>Nhận dạng cử chỉ từ tín hiệu gia tốc</h1>'
    '<p>Chọn mẫu, tải dữ liệu CSV/NPZ/ZIP, xem tín hiệu ba trục và nhận dự đoán bằng mô hình TF-C đã huấn luyện.</p></div>',
    unsafe_allow_html=True,
)

tab_predict, tab_model, tab_help = st.tabs(["Dự đoán", "Về mô hình", "Hướng dẫn dữ liệu"])

with tab_predict:
    left, right = st.columns([0.95, 1.25], gap="large")
    sensor_data = None
    fingerprint = ""
    is_raw = False
    source = "Mẫu SGR"

    with left:
        st.markdown('<div class="gl-section">01 / DỮ LIỆU ĐẦU VÀO</div>', unsafe_allow_html=True)
        source = st.radio(
            "Nguồn dữ liệu", ["Mẫu SGR", "Tải file"], horizontal=True,
            help="Dữ liệu mẫu lấy từ SGR test; file có thể là CSV, NPZ hoặc ZIP Sensor Logger.",
        )
        if source == "Mẫu SGR":
            examples = get_examples()
            if examples is None:
                st.warning("Chưa có samples/sgr_examples.npz. Hãy tải CSV hoặc NPZ.")
            else:
                available_labels = [i for i in range(len(LABELS)) if np.any(examples["labels"] == i)]
                selected_label = st.selectbox(
                    "Chọn loại cử chỉ", available_labels,
                    format_func=lambda i: TRANSLATIONS[LABELS[i]], key="demo_label",
                )
                options = [i for i in range(len(examples["samples"]))
                           if int(examples["labels"][i]) == selected_label]
                if st.session_state.get("demo_idx") not in options:
                    st.session_state["demo_idx"] = options[0]
                sample_index = st.selectbox(
                    "Chọn mẫu trong tập SGR test", options, key="demo_idx",
                    format_func=lambda i: f"Mẫu SGR test #{int(examples['indices'][i])}",
                )
                values = examples["samples"][sample_index:sample_index + 1]
                sensor_data = SensorData(
                    samples=values, plot=values.transpose(0, 2, 1),
                    time=np.arange(206, dtype=float)[None, :],
                    names=[f"SGR test #{int(examples['indices'][sample_index])}"],
                    labels=examples["labels"][sample_index:sample_index + 1],
                    raw_rows=206,
                    notices=["Dữ liệu đã chuẩn hóa theo SGR train, không phải dữ liệu Sensor Logger."],
                )
                fingerprint = f"sample-{sample_index}"
        else:
            uploaded = st.file_uploader(
                "Tải dữ liệu từ máy", type=["csv", "npz", "zip"],
                help="CSV với x/y/z, NPZ [N,3,206], hoặc ZIP có Accelerometer.csv.",
            )
            mode = st.radio(
                "Chuẩn hóa dữ liệu", ["Gia tốc thô — chuẩn hóa theo SGR", "Đã chuẩn hóa theo SGR"],
                help="Chỉ chọn 'Đã chuẩn hóa' khi file đã dùng đúng mean/std của SGR train.",
            )
            is_raw = mode.startswith("Gia tốc thô")
            if uploaded:
                try:
                    content = uploaded.getvalue()
                    fingerprint = hashlib.sha256(content).hexdigest()
                    ext = Path(uploaded.name).suffix.lower()
                    if ext == ".csv":
                        sensor_data = csv_bytes(content, is_raw=is_raw)
                    elif ext == ".npz":
                        sensor_data = npz_bytes(content, is_raw=is_raw)
                    elif ext == ".zip":
                        sensor_data = zip_bytes(content, is_raw=is_raw)
                except (ValueError, OSError, KeyError, RuntimeError) as exc:
                    st.error(f"Không thể đọc file: {exc}")

        signature = data_signature(source, fingerprint, is_raw) if sensor_data is not None else None
        saved = st.session_state.get("prediction_cache")
        if sensor_data is None:
            saved = None
            st.markdown(
                '<div class="gl-tip">Chọn một mẫu SGR có sẵn hoặc tải file chứa gia tốc <b>X, Y, Z</b> '
                'để bắt đầu. Ứng dụng không yêu cầu file mô hình của người dùng.</div>',
                unsafe_allow_html=True,
            )
        else:
            a, b, c = st.columns(3)
            a.metric("Điểm đo", f"{sensor_data.raw_rows:,}")
            b.metric("Cửa sổ", str(len(sensor_data.samples)))
            c.metric("Trục", "X/Y/Z")
            with st.expander("Chi tiết tiền xử lý", expanded=False):
                for note in sensor_data.notices:
                    st.caption("• " + note)
                if sensor_data.sampling_hz is not None:
                    st.caption(f"• Tần số ghi ước tính: {sensor_data.sampling_hz:.2f} Hz")
            if source == "Tải file":
                st.warning("Mô hình học từ SGR của đồng hồ thông minh. Dữ liệu từ điện thoại có thể cho kết quả kém tin cậy.", icon="⚠️")
            if st.button(
                "Phân loại bằng TF-C  →", type="primary", use_container_width=True,
                disabled=not MODEL_PATH.is_file(),
            ):
                try:
                    with st.spinner("Mô hình đang phân loại dữ liệu…"):
                        encoder, classifier, _ = get_model()
                        scores = predict(encoder, classifier, sensor_data.samples)
                    st.session_state["prediction_cache"] = {
                        "signature": signature, "probs": scores,
                    }
                except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
                    st.error(f"Không thể dự đoán: {exc}")
                    st.session_state.pop("prediction_cache", None)
            saved = st.session_state.get("prediction_cache")
            if saved is not None and saved["signature"] == signature:
                focus_idx = min(int(st.session_state.get("signal_window", 0)), len(sensor_data.samples) - 1)
                compact_result(saved["probs"], focus_idx, sensor_data)
            else:
                st.caption("Chọn dữ liệu và nhấn **Phân loại bằng TF-C** để xem kết quả tại đây.")

    with right:
        st.markdown('<div class="gl-section">02 / BIỂU ĐỒ TÍN HIỆU</div>', unsafe_allow_html=True)
        if sensor_data is None:
            st.markdown(
                '<div class="gl-empty">Biểu đồ gia tốc X, Y, Z sẽ hiện tại đây<br/>sau khi bạn chọn hoặc tải dữ liệu.</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.session_state.get("signal_window", 0) >= len(sensor_data.samples):
                st.session_state["signal_window"] = 0
            focused = st.selectbox(
                "Cửa sổ đang xem", range(len(sensor_data.samples)), key="signal_window",
                format_func=lambda i: sensor_data.names[i],
            )
            df_signal = signal_frame(sensor_data, focused)
            st.line_chart(df_signal, height=280, color=["#6EA8FA", "#4DC7B0", "#F49C8D"])
            st.caption("Đường X/Y/Z phản ánh giá trị có trong file; trục ngang dùng timestamp nếu có.")
            # Analyze the same plotted window; do not alter samples used for TF-C.
            show_signal_analysis(
                sensor_data,
                focused,
                is_normalized=(source == "Mẫu SGR" or not is_raw),
            )

    st.divider()
    st.markdown('<div class="gl-section">03 / PHÂN TÍCH DỰ ĐOÁN</div>', unsafe_allow_html=True)
    saved = st.session_state.get("prediction_cache")
    if sensor_data is None or saved is None or saved["signature"] != signature:
        st.markdown(
            '<div class="gl-empty">Kết quả chi tiết sẽ xuất hiện sau khi bạn nhấn <b>Phân loại bằng TF-C</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        probs = saved["probs"]
        focus = min(int(st.session_state.get("signal_window", 0)), len(probs) - 1)
        p = int(np.argmax(probs[focus]))
        truth = truth_for_window(sensor_data, focus)
        score = float(probs[focus, p] * 100)
        a, b, c = st.columns([1.2, 1.2, 1])
        with a:
            st.caption("Dự đoán của TF-C")
            st.markdown(f'<div class="gl-focus-value">{escape(TRANSLATIONS[LABELS[p]])}</div>', unsafe_allow_html=True)
        with b:
            st.caption("Nhãn thực tế (nếu có)")
            st.markdown(
                '<div class="gl-focus-value">' + (escape(TRANSLATIONS[LABELS[truth]]) if truth is not None else "Không có nhãn") + '</div>',
                unsafe_allow_html=True,
            )
        with c:
            st.caption("Điểm Softmax")
            st.markdown(f'<div class="gl-focus-value">{score:.1f}%</div>', unsafe_allow_html=True)
        if truth is not None:
            if p == truth:
                st.success("Nhãn dự đoán khớp nhãn thật của mẫu SGR.", icon="✅")
            else:
                st.warning("Nhãn dự đoán chưa khớp nhãn thật. Đây là một trường hợp phân loại sai của mô hình.", icon="⚠️")
        else:
            st.info("Tệp tải lên không có nhãn đối chiếu; không thể kết luận dự đoán đúng hay sai.")
        st.markdown("**Phân bố điểm Softmax theo 6 cử chỉ**")
        probability_chart(probs[focus], p)
        st.caption("Điểm Softmax là đầu ra mô hình, không đồng nghĩa với xác suất đúng đã được hiệu chuẩn.")
        show_prediction_table(probs, sensor_data)

with tab_model:
    st.subheader("Về mô hình TF-C")
    st.markdown(
        "**TF-C Transformer gốc** sử dụng hai encoder cho miền thời gian và miền tần số, "
        "hai projection heads và classifier sáu lớp. Mô hình được pre-train trên HAR "
        "(tên MAR trong lần chạy cũ) và fine-tune trên SGR."
    )
    st.code("Đầu vào: [N, 3, 206]   →   Đầu ra: [N, 6] (Softmax)", language="text")
    st.caption("Không sử dụng kiến trúc DINOv3 trong ứng dụng này.")
    st.divider()
    if RESULTS_PATH.is_file():
        try:
            stats = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            test = stats.get("test_metrics", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("SGR Test Accuracy", f"{float(test.get('accuracy', 0)) * 100:.2f}%")
            m2.metric("SGR Test Macro-F1", f"{float(test.get('macro_f1', 0)) * 100:.2f}%")
            m3.metric("Epoch tốt nhất", str(stats.get("best_epoch", "—")))
            st.warning("Validation và test từng có chênh lệch lớn. Chưa đủ cơ sở khẳng định mô hình tổng quát hóa tốt trên thiết bị mới.")
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            st.warning(f"Không thể đọc thống kê huấn luyện: {exc}")
    else:
        st.info("Chưa có models/training_results.json. Kết quả huấn luyện không được mô phỏng.")
    st.markdown(
        "**Nguồn tham khảo:** [TF-C gốc — Harvard](https://github.com/mims-harvard/TFC-pretraining) · "
        "[Dataset trên Kaggle](https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c)"
    )

with tab_help:
    st.subheader("Cách chuẩn bị tệp dữ liệu")
    st.markdown(
        """
**CSV:** Cần các cột `x`, `y`, `z` với tối thiểu **206 dòng**. Có thể bổ sung `seconds_elapsed` hoặc `time`.

```csv
time,x,y,z
0.00,0.3,-0.1,0.5
0.02,0.2,-0.2,0.4
```

**NPZ:** Khóa `samples`, kích thước `[N,3,206]` hoặc `[N,206,3]`. Có thể kèm `labels` nếu đây là tập tham chiếu SGR.

**ZIP Sensor Logger:** Chứa duy nhất một `Accelerometer.csv` của một phiên ghi.

**Tiền xử lý:** CSV/ZIP được chia cửa sổ 206 điểm, bước trượt 100 điểm. Dữ liệu thô được chuẩn hóa bằng mean/std của SGR train. Không tự nội suy tăng tần số cảm biến.

**Giới hạn:** TF-C hiện dự đoán **Typing, Teeth, Drinking, Writing, Clapping, Folding** (SGR) — không phải Walking/Stairs/Sitting/Standing (HAR). Dữ liệu điện thoại khác nguồn đồng hồ thông minh và có thể bị sai lệch phân bố.
"""
    )
    st.info("Ứng dụng chỉ nhận CSV, NPZ hoặc ZIP gia tốc; không nhận tệp PyTorch/pickle do người dùng tải lên.")
