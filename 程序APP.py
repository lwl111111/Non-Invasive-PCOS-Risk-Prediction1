import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import textwrap

# ========================
# 兼容性补丁（部分环境下 SHAP 可能需要）
# ========================
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

# ========================
# 页面基础设置
# ========================
CONTAINER_W = 820
st.set_page_config(page_title="Non-Invasive PCOS Risk Prediction", layout="centered")

st.markdown(f"""
<style>
.main .block-container {{
  max-width: {CONTAINER_W}px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

div[data-testid="stPlotlyChart"] {{
  padding: 12px 10px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}

.badge {{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  font-size:14px;
  font-weight:600;
  color:#fff;
  margin-left:10px;
  vertical-align:middle;
}}
</style>
""", unsafe_allow_html=True)

# ========================
# 常量
# ========================
MODEL_PATH = "WeightedVoting_XGBoost_RF.pkl"
BACKGROUND_PATH = "shap_background.csv"

# 训练时真实建模顺序：不能改
MODEL_FEATURES = [
    "Acne",
    "Body fat",
    "Hirsutism",
    "Menstrual cycle",
    "Menstrual duration",
    "Menstrual flow"
]

# 前端显示顺序：分类变量放前面
FORM_FEATURES = [
    ("Menstrual cycle",    "Menstrual cycle",    "categorical", "21–35 days"),
    ("Menstrual duration", "Menstrual duration", "categorical", "2–8 days"),
    ("Menstrual flow",     "Menstrual flow",     "categorical", "Normal"),
    ("Acne",               "Acne",               "numerical",   0.00),
    ("Hirsutism",          "Hirsutism",          "numerical",   0.00),
    ("Body fat",           "Body fat",           "numerical",   30.00),
]

CATEGORICAL_COLS = [
    "Menstrual cycle",
    "Menstrual duration",
    "Menstrual flow"
]

NUMERICAL_COLS = [
    "Acne",
    "Body fat",
    "Hirsutism"
]

# 内部固定阈值：只用于徽章判断，不在页面显示
FIXED_THRESHOLD = 0.4207

# ========================
# 分类变量映射
# 如训练数据编码相反，只需把 0/1 对调
# ========================
BINARY_OPTION_MAP = {
    "Menstrual cycle": {
        "21–35 days": 0,
        "<21 or >35 days": 1
    },
    "Menstrual duration": {
        "2–8 days": 0,
        "<2 or >8 days": 1
    },
    "Menstrual flow": {
        "Normal": 0,
        "Light or heavy": 1
    }
}

BINARY_VALUE_TO_TEXT = {
    feat: {v: k for k, v in mapping.items()}
    for feat, mapping in BINARY_OPTION_MAP.items()
}

DISPLAY_NAME = {
    "Acne": "Acne",
    "Body fat": "Body fat",
    "Hirsutism": "Hirsutism",
    "Menstrual cycle": "Menstrual cycle",
    "Menstrual duration": "Menstrual duration",
    "Menstrual flow": "Menstrual flow"
}

# ========================
# 加载模型
# ========================
@st.cache_resource
def load_model_bundle(model_path):
    obj = joblib.load(model_path)

    model = obj
    if isinstance(obj, dict):
        for key in ["final_model", "model", "estimator", "best_model"]:
            if key in obj:
                model = obj[key]
                break

    if not hasattr(model, "predict_proba"):
        raise ValueError("加载的模型对象不支持 predict_proba。")

    return model


# ========================
# 加载背景数据
# ========================
@st.cache_data
def load_background_data(background_path):
    if not os.path.exists(background_path):
        return None

    bg = pd.read_csv(background_path)

    missing = [c for c in MODEL_FEATURES if c not in bg.columns]
    if missing:
        raise ValueError(f"shap_background.csv 缺少以下列: {missing}")

    bg = bg[MODEL_FEATURES].copy()

    for col in CATEGORICAL_COLS:
        bg[col] = pd.to_numeric(bg[col], errors="raise").astype(int)

    for col in NUMERICAL_COLS:
        bg[col] = pd.to_numeric(bg[col], errors="raise")

    # 背景样本过多会让 SHAP 太慢
    if len(bg) > 100:
        bg = bg.sample(n=100, random_state=42)

    return bg


model = load_model_bundle(MODEL_PATH)
background_df = load_background_data(BACKGROUND_PATH)

# ========================
# 输入整理
# ========================
def prepare_input_df(data_like):
    arr = np.asarray(data_like)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    df = pd.DataFrame(arr, columns=MODEL_FEATURES)

    for col in CATEGORICAL_COLS:
        df[col] = np.rint(pd.to_numeric(df[col], errors="raise")).astype(int)

    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def predict_positive_proba(data_like):
    df = prepare_input_df(data_like)
    return model.predict_proba(df)[:, 1]


# ========================
# 标签与格式
# ========================
def classify_prediction(p: float, threshold: float):
    if p >= threshold:
        return "Higher predicted likelihood", "#C62828"
    return "Lower predicted likelihood", "#2E7D32"


def format_feature_value(feature, value):
    if feature in BINARY_VALUE_TO_TEXT:
        try:
            value_int = int(round(float(value)))
        except Exception:
            return str(value)
        return BINARY_VALUE_TO_TEXT[feature].get(value_int, str(value_int))

    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def format_contribution(x):
    if abs(x) < 0.005:
        return "0.00"
    return f"{x:+.2f}"


# ========================
# SHAP explainer
# ========================
def get_shap_explainer():
    if background_df is None:
        return None

    if "kernel_shap_explainer" not in st.session_state:
        st.session_state["kernel_shap_explainer"] = shap.KernelExplainer(
            predict_positive_proba,
            background_df,
            link="identity"
        )
    return st.session_state["kernel_shap_explainer"]


def compute_kernel_shap_probability(X_one_row):
    explainer = get_shap_explainer()
    if explainer is None:
        return None, None

    shap_values = explainer.shap_values(X_one_row, nsamples=200)

    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[0])
    else:
        shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        local_vals = shap_values[0]
    else:
        local_vals = shap_values.reshape(-1)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]

    return local_vals, float(base_value)


# ========================
# 绘图
# ========================
def plot_pp_bar(df_plot):
    df_plot = df_plot.copy()

    labels = [
        textwrap.fill(f"{DISPLAY_NAME.get(f, f)} = {vtxt}", width=28)
        for f, vtxt in zip(df_plot["feature"], df_plot["value_text"])
    ]

    x_vals = df_plot["dpp"].to_numpy()
    colors = np.where(x_vals >= 0, "#E45756", "#4C78A8")

    texts, textpos, textcolor = [], [], []
    for x in x_vals:
        texts.append(format_contribution(x))
        if x < 0:
            textpos.append("inside")
            textcolor.append("white")
        else:
            if abs(x) >= 1:
                textpos.append("inside")
                textcolor.append("white")
            else:
                textpos.append("outside")
                textcolor.append("black")

    labels_rev = labels[::-1]
    x_vals_rev = x_vals[::-1]
    colors_rev = colors[::-1]
    texts_rev = texts[::-1]
    textpos_rev = textpos[::-1]
    textcolor_rev = textcolor[::-1]

    fig = go.Figure(go.Bar(
        y=labels_rev,
        x=x_vals_rev,
        orientation="h",
        marker_color=colors_rev,
        text=texts_rev,
        texttemplate="%{text}",
        textposition=textpos_rev,
        insidetextanchor="end",
        textfont=dict(color=textcolor_rev, size=14),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.2f} percentage points<extra></extra>",
    ))

    fig.update_layout(
        height=max(180, 42 * len(labels) + 60),
        margin=dict(l=270, r=70, t=12, b=12),
        font=dict(family="Times New Roman", size=17),
        yaxis=dict(title="", type="category", tickfont=dict(size=15), automargin=True),
        xaxis=dict(
            title="Approximate contribution to model-predicted probability (percentage points)",
            zeroline=True,
            zerolinewidth=1.2,
            zerolinecolor="#B0BEC5",
            showgrid=True,
            gridcolor="#EFEFEF",
            automargin=True,
        ),
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        uniformtext_minsize=12,
        uniformtext_mode="hide",
    )

    fig.add_vline(x=0, line_dash="dot", line_color="#B0BEC5", line_width=1)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ========================
# 页面标题
# ========================
st.title("Non-Invasive PCOS Risk Prediction")
st.caption("Estimate individualized PCOS risk using six non-invasive features.")

# ========================
# 初始化默认值
# ========================
for feat_name, label, ftype, default in FORM_FEATURES:
    key = f"{feat_name}_input"
    if key not in st.session_state:
        st.session_state[key] = default

# ========================
# 输入表单
# ========================
with st.form("prediction_form", clear_on_submit=False):
    for feat_name, label, ftype, default in FORM_FEATURES:
        key = f"{feat_name}_input"

        if ftype == "categorical":
            options = list(BINARY_OPTION_MAP[feat_name].keys())
            current_value = st.session_state[key]
            if current_value not in options:
                current_value = options[0]
                st.session_state[key] = current_value

            st.radio(
                label,
                options,
                key=key,
                index=options.index(current_value),
                horizontal=True
            )
        else:
            st.number_input(
                label,
                value=float(st.session_state[key]),
                step=0.01,
                format="%.2f",
                key=key
            )

    submitted = st.form_submit_button("Predict", type="primary")

# ========================
# 预测与解释
# ========================
if submitted:
    form_values = {}

    for feat_name, label, ftype, default in FORM_FEATURES:
        key = f"{feat_name}_input"

        if ftype == "categorical":
            form_values[feat_name] = int(BINARY_OPTION_MAP[feat_name][st.session_state[key]])
        else:
            form_values[feat_name] = float(st.session_state[key])

    # 按模型真实输入顺序构造 X
    X = pd.DataFrame([[form_values[col] for col in MODEL_FEATURES]], columns=MODEL_FEATURES)
    X = prepare_input_df(X)

    # 预测概率
    p1 = float(model.predict_proba(X)[0, 1])
    pred_label, pred_color = classify_prediction(p1, FIXED_THRESHOLD)

    st.markdown(
        f"""
        <div style='font-family:Times New Roman; font-size:20px;'>
          <b>Model-predicted probability of PCOS: {p1 * 100:.2f}%.</b>
          <span class="badge" style="background:{pred_color};">{pred_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # SHAP 局部解释
    if background_df is None:
        st.info("Prediction is available. To display local feature contributions, add 'shap_background.csv' to the same folder as the app.")
    else:
        with st.spinner("Calculating local feature contributions..."):
            shap_vals, base_value = compute_kernel_shap_probability(X)

        if shap_vals is None:
            st.info("Prediction is available, but local feature contributions could not be calculated.")
        else:
            feat_vals = X.iloc[0].to_numpy()
            dpp = shap_vals * 100.0

            order = np.argsort(-np.abs(dpp), kind="mergesort")
            ordered_features = X.columns.to_numpy()[order]
            ordered_values = feat_vals[order]

            df_sorted = pd.DataFrame({
                "feature": ordered_features,
                "value": ordered_values,
                "value_text": [
                    format_feature_value(f, v)
                    for f, v in zip(ordered_features, ordered_values)
                ],
                "dpp": dpp[order],
            })

            st.caption(f"Reference baseline probability for local explanation: {base_value * 100:.2f}%")
            plot_pp_bar(df_sorted)

            with st.expander("Show full contributions"):
                table = df_sorted[["feature", "value_text", "dpp"]].copy()
                table["feature"] = table["feature"].map(lambda x: DISPLAY_NAME.get(x, x))
                table.columns = ["Feature", "Value", "Contribution (percentage points)"]
                table.insert(0, "Rank", np.arange(1, len(table) + 1))
                st.dataframe(table, use_container_width=True, hide_index=True)