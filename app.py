# app.py
# Fixes:
# - Metrics (loss/accuracy) not “hidden”: make the metrics panel shorter + stack 2 compact horizontal charts (Accuracy then Loss)
# - Show best model KPIs (ACC + LOSS) clearly at top of metrics panel
# - Class distribution axis: force integer formatting (no 3e2 / scientific) + linear scale + explicit domain up to max (~300)

import os
import time
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import altair as alt

st.set_page_config(page_title="Dogs — Dashboard", layout="wide", initial_sidebar_state="expanded")

MODELS_DIR = "models"
ASSETS_DIR = "assets"
SAMPLES_DIR = "sample_images"

RESNET_PATH = os.path.join(MODELS_DIR, "resnet50_baseline.h5")
# CONVNEXT_PATH = os.path.join(MODELS_DIR, "convnext_baseline.h5")
CONVNEXT_PATH = os.path.join(MODELS_DIR, "convnext_model")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.npy")

DIST_CSV = os.path.join(ASSETS_DIR, "class_distribution.csv")        # class,count
MODEL_SCORES_CSV = os.path.join(ASSETS_DIR, "model_scores.csv")      # model,accuracy,loss

CM_PNG_RESNET = os.path.join(ASSETS_DIR, "confusion_matrix_resnet50.png")
CM_PNG_CONVNEXT = os.path.join(ASSETS_DIR, "confusion_matrix_convnext.png")
CM_CSV_RESNET = os.path.join(ASSETS_DIR, "confusion_matrix_resnet50.csv")
CM_CSV_CONVNEXT = os.path.join(ASSETS_DIR, "confusion_matrix_convnext.csv")

IMG_SIZE = (299, 299)

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
h1, h2, h3 {letter-spacing: 0.2px;}

.kpi {
  background: #0f172a;
  border: 1px solid rgba(148,163,184,.18);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,.16);
  height: 96px;
}
.kpi-title {color: rgba(226,232,240,.75); font-size: 12px; margin-bottom: 6px;}
.kpi-value {color: #e2e8f0; font-size: 26px; font-weight: 800; line-height: 1.05;}
.kpi-sub {color: rgba(226,232,240,.6); font-size: 12px; margin-top: 6px;}

.panel {
  background: #0b1220;
  border: 1px solid rgba(148,163,184,.16);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,.12);
}
.panel-title {display:flex; align-items:center; justify-content:space-between; gap:12px;}
.badge {
  background: rgba(34,197,94,.18);
  border: 1px solid rgba(34,197,94,.28);
  color: #bbf7d0;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}
.badge-warn{
  background: rgba(245,158,11,.14);
  border: 1px solid rgba(245,158,11,.25);
  color: #fde68a;
}
.small-note {color: rgba(226,232,240,.70); font-size: 12px;}
hr {border: none; border-top: 1px solid rgba(148,163,184,.14); margin: 14px 0;}

section[data-testid="stSidebar"] > div {
  background: linear-gradient(180deg, #0b1220 0%, #070b13 100%);
  border-right: 1px solid rgba(148,163,184,.12);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def safe_exists(path: str) -> bool:
    return path is not None and os.path.exists(path)

def kpi(col, title, value, sub=""):
    col.markdown(
        f"""
<div class="kpi">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def load_csv_safe(path: str) -> pd.DataFrame | None:
    if not safe_exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def prettify_model_name(x: str) -> str:
    s = str(x).strip()
    low = s.lower()
    if "convnext" in low:
        return "ConvNeXt"
    if "resnet" in low:
        return "ResNet50"
    return s

def choose_best_model(scores_df: pd.DataFrame | None) -> str | None:
    if scores_df is None or scores_df.empty:
        return None
    df = scores_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "model" not in df.columns:
        return None

    df["model"] = df["model"].apply(prettify_model_name)
    df["accuracy"] = pd.to_numeric(df.get("accuracy"), errors="coerce")
    df["loss"] = pd.to_numeric(df.get("loss"), errors="coerce")

    if df["accuracy"].isna().all() and df["loss"].isna().all():
        return None

    # Best = max accuracy, tie-breaker min loss
    df = df.sort_values(["accuracy", "loss"], ascending=[False, True], na_position="last")
    return str(df.iloc[0]["model"])

def load_confusion_matrix(best_model: str | None) -> tuple[str, pd.DataFrame | None]:
    if best_model is None:
        return "none", None

    low = best_model.lower()
    if "convnext" in low:
        if safe_exists(CM_PNG_CONVNEXT):
            return "png_convnext", None
        df = load_csv_safe(CM_CSV_CONVNEXT)
        if df is not None and df.shape[0] > 0:
            return "csv", df
        return "none", None

    if "resnet" in low:
        if safe_exists(CM_PNG_RESNET):
            return "png_resnet", None
        df = load_csv_safe(CM_CSV_RESNET)
        if df is not None and df.shape[0] > 0:
            return "csv", df
        return "none", None

    return "none", None

def plot_confusion_heatmap(cm_df: pd.DataFrame):
    df = cm_df.copy()
    if df.shape[1] >= 2:
        first = df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[first]):
            df = df.set_index(first)

    num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    melted = num.reset_index().melt(id_vars=[num.reset_index().columns[0]], var_name="Pred", value_name="Count")
    melted = melted.rename(columns={num.reset_index().columns[0]: "True"})
    melted["Count"] = pd.to_numeric(melted["Count"], errors="coerce").fillna(0)

    heat = (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("Pred:N", title="Prédit", sort=None),
            y=alt.Y("True:N", title="Vrai", sort=None),
            tooltip=["True:N", "Pred:N", alt.Tooltip("Count:Q", format=",.0f")],
            color=alt.Color("Count:Q", title="Nb"),
        )
        .properties(height=360)
    )
    text = (
        alt.Chart(melted)
        .mark_text(baseline="middle", fontSize=10)
        .encode(
            x="Pred:N",
            y="True:N",
            text=alt.Text("Count:Q", format=".0f"),
            tooltip=["True:N", "Pred:N", alt.Tooltip("Count:Q", format=",.0f")],
        )
    )
    st.altair_chart(heat + text, use_container_width=True)

def load_image_from_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Impossible de décoder l’image.")
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

def preprocess_like_notebook(img_rgb: np.ndarray, target_size=(299, 299)) -> np.ndarray:
    img = cv.resize(img_rgb, target_size, interpolation=cv.INTER_LINEAR)
    img_yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    dst_img = cv.fastNlMeansDenoisingColored(
        src=img_equ,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return dst_img.astype(np.float32)

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    s = float(probs.sum())
    if s <= 0:
        return np.ones_like(probs) / max(1, probs.size)
    return probs / s

# -----------------------------
# Custom layers (ConvNeXt)
# -----------------------------
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        dim = self.projection_dim if self.projection_dim is not None else int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init_values": self.init_values, "projection_dim": self.projection_dim})
        return cfg

class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_path_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = float(drop_path_rate)

    def call(self, x, training=None):
        if (not training) or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * binary_tensor

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_path_rate": self.drop_path_rate})
        return cfg

CUSTOM_OBJECTS = {"LayerScale": LayerScale, "StochasticDepth": StochasticDepth}

@st.cache_resource(show_spinner=False)
def load_classes(classes_path: str) -> np.ndarray:
    if not safe_exists(classes_path):
        raise FileNotFoundError(
            f"classes.npy introuvable: {classes_path} — "
            "Sauvegarde: np.save('models/classes.npy', encoder.classes_)"
        )
    return np.load(classes_path, allow_pickle=True)

@st.cache_resource(show_spinner=False)
def load_model_safe(model_path: str) -> tf.keras.Model:
    if not safe_exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)

# -----------------------------
# Header
# -----------------------------
left, right = st.columns([0.74, 0.26])
with left:
    st.markdown("## Dashboard de classification de races de chiens")
    st.markdown('<div class="small-note">EDA (5 classes) + Démo de prédiction (ResNet / ConvNeXt)</div>', unsafe_allow_html=True)
with right:
    st.markdown("")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("### Navigation")
section = st.sidebar.radio("Section", ["Overview", "Predict"], index=0)

# -----------------------------
# Load classes
# -----------------------------
classes = None
try:
    classes = load_classes(CLASSES_PATH)
except Exception as e:
    st.error(str(e))

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    if classes is None:
        st.stop()

    dist_df = load_csv_safe(DIST_CSV)
    if dist_df is not None:
        dist_df.columns = [c.strip().lower() for c in dist_df.columns]
        if "class" not in dist_df.columns:
            for altc in ["classe", "label", "breed"]:
                if altc in dist_df.columns:
                    dist_df = dist_df.rename(columns={altc: "class"})
                    break
        if "count" not in dist_df.columns:
            for altc in ["n", "images", "nb", "freq"]:
                if altc in dist_df.columns:
                    dist_df = dist_df.rename(columns={altc: "count"})
                    break
        if not set(["class", "count"]).issubset(dist_df.columns):
            dist_df = None

    scores_df = load_csv_safe(MODEL_SCORES_CSV)
    if scores_df is not None:
        scores_df.columns = [c.strip().lower() for c in scores_df.columns]
        if "model" not in scores_df.columns:
            for altc in ["modèle", "name", "arch"]:
                if altc in scores_df.columns:
                    scores_df = scores_df.rename(columns={altc: "model"})
                    break

    best_model = choose_best_model(scores_df)
    cm_mode, cm_df = load_confusion_matrix(best_model)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    n_classes = len(classes)

    total_imgs = None
    dominant_class = None
    if dist_df is not None:
        dtmp = dist_df.copy()
        dtmp["count"] = pd.to_numeric(dtmp["count"], errors="coerce").fillna(0).astype(int)
        total_imgs = int(dtmp["count"].sum())
        if total_imgs > 0:
            top = dtmp.sort_values("count", ascending=False).iloc[0]
            dominant_class = str(top["class"])

    best_acc = None
    best_loss = None
    if scores_df is not None and best_model is not None:
        tmp = scores_df.copy()
        tmp["model"] = tmp["model"].apply(prettify_model_name)
        tmp["accuracy"] = pd.to_numeric(tmp.get("accuracy"), errors="coerce")
        tmp["loss"] = pd.to_numeric(tmp.get("loss"), errors="coerce")
        row = tmp[tmp["model"].str.lower() == best_model.lower()]
        if not row.empty:
            best_acc = row.iloc[0].get("accuracy")
            best_loss = row.iloc[0].get("loss")

    kpi(k1, "Classes", f"{n_classes}", "Stanford Dogs (subset)")
    kpi(k2, "Images (total)", f"{total_imgs:,}" if total_imgs is not None else "—", "Depuis export notebook")
    kpi(k3, "Classe dominante", dominant_class if dominant_class is not None else "—", "")
    if best_model is None:
        kpi(k4, "Meilleur modèle", "—", "Ajoute model_scores.csv")
    else:
        sub = []
        if best_acc is not None and not pd.isna(best_acc):
            sub.append(f"ACC={float(best_acc):.3f}")
        if best_loss is not None and not pd.isna(best_loss):
            sub.append(f"LOSS={float(best_loss):.3f}")
        kpi(k4, "Meilleur modèle", best_model, " • ".join(sub) if sub else "accuracy/loss")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Layout: 36% / 29% / 35%
    c1, c2, c3 = st.columns([0.36, 0.29, 0.35])

    # Distribution (vertical) — NO scientific notation + explicit domain
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Distribution des classes")

        if dist_df is None:
            st.info("Ajoute assets/class_distribution.csv avec colonnes: class,count")
        else:
            d = dist_df.copy()
            d["class"] = d["class"].astype(str)
            d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(int)
            d = d.sort_values("count", ascending=False)

            max_count = int(d["count"].max()) if len(d) else 0
            ymax = int(np.ceil(max(1, max_count) * 1.05))  # ~300 -> 315

            sel = alt.selection_point(fields=["class"], empty="none")
            chart = (
                alt.Chart(d)
                .mark_bar()
                .encode(
                    x=alt.X("class:N", title="", sort="-y"),
                    y=alt.Y(
                        "count:Q",
                        title="Images",
                        scale=alt.Scale(domain=[0, ymax], nice=False, zero=True),
                        axis=alt.Axis(format="d")  # <— forces integer ticks, avoids 3e2
                    ),
                    tooltip=[
                        alt.Tooltip("class:N", title="Classe"),
                        alt.Tooltip("count:Q", title="Images", format="d"),
                    ],
                    opacity=alt.condition(sel, alt.value(1.0), alt.value(0.55)),
                )
                .add_params(sel)
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Metrics (horizontal) — compact + clearly visible (no overlap feeling)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        badge = f'<span class="badge">Best: {best_model}</span>' if best_model else '<span class="badge badge-warn">Missing</span>'
        st.markdown(f'<div class="panel-title"><h3 style="margin:0;">Loss & Accuracy</h3>{badge}</div>', unsafe_allow_html=True)

        if scores_df is None:
            st.info("Ajoute assets/model_scores.csv (model,accuracy,loss)")
        else:
            df = scores_df.copy()
            df.columns = [c.strip().lower() for c in df.columns]
            df["model"] = df["model"].apply(prettify_model_name)
            df["accuracy"] = pd.to_numeric(df.get("accuracy"), errors="coerce")
            df["loss"] = pd.to_numeric(df.get("loss"), errors="coerce")
            df = df.dropna(subset=["accuracy", "loss"], how="all")

            # Best KPIs inside this panel (so they're always visible)
            if best_model is not None:
                row = df[df["model"].str.lower() == best_model.lower()]
                if not row.empty:
                    acc_v = row.iloc[0].get("accuracy")
                    loss_v = row.iloc[0].get("loss")
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy (best)", f"{float(acc_v):.3f}" if acc_v is not None and not pd.isna(acc_v) else "—")
                    m2.metric("Loss (best)", f"{float(loss_v):.3f}" if loss_v is not None and not pd.isna(loss_v) else "—")

            # Two compact horizontal charts (thin bars), stacked (no facet overflow)
            # Accuracy chart
            acc_df = df.dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False)
            if not acc_df.empty:
                acc_chart = (
                    alt.Chart(acc_df)
                    .mark_bar(size=8)  # thin bars
                    .encode(
                        x=alt.X("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("model:N", title="", sort=None),
                        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("accuracy:Q", title="Accuracy", format=".3f")],
                    )
                    .properties(height=110)
                )
                st.altair_chart(acc_chart, use_container_width=True)

            # Loss chart (domain auto)
            loss_df = df.dropna(subset=["loss"]).sort_values("loss", ascending=True)
            if not loss_df.empty:
                loss_chart = (
                    alt.Chart(loss_df)
                    .mark_bar(size=8)  # thin bars
                    .encode(
                        x=alt.X("loss:Q", title="Loss"),
                        y=alt.Y("model:N", title="", sort=None),
                        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("loss:Q", title="Loss", format=".4f")],
                    )
                    .properties(height=110)
                )
                st.altair_chart(loss_chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Confusion matrix
    with c3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        badge = f'<span class="badge">Best</span>' if best_model else f'<span class="badge badge-warn">Missing scores</span>'
        st.markdown(f'<div class="panel-title"><h3 style="margin:0;">Matrice de confusion</h3>{badge}</div>', unsafe_allow_html=True)

        if best_model is None:
            st.info("Ajoute assets/model_scores.csv pour identifier le meilleur modèle.")
        else:
            if cm_mode == "png_resnet":
                st.image(CM_PNG_RESNET, use_container_width=True)
            elif cm_mode == "png_convnext":
                st.image(CM_PNG_CONVNEXT, use_container_width=True)
            elif cm_mode == "csv" and cm_df is not None:
                plot_confusion_heatmap(cm_df)
            else:
                st.info(
                    "Ajoute une matrice de confusion:\n"
                    "• assets/confusion_matrix_resnet50.png ou .csv\n"
                    "• assets/confusion_matrix_convnext.png ou .csv"
                )

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PREDICT (inchangé)
# -----------------------------
else:
    if classes is None:
        st.stop()

    scores_df = load_csv_safe(MODEL_SCORES_CSV)
    if scores_df is not None:
        scores_df.columns = [c.strip().lower() for c in scores_df.columns]
        if "model" not in scores_df.columns:
            for altc in ["modèle", "name", "arch"]:
                if altc in scores_df.columns:
                    scores_df = scores_df.rename(columns={altc: "model"})
                    break
    best_model = choose_best_model(scores_df)

    resnet_model = None
    convnext_model = None
    resnet_err = None
    convnext_err = None

    try:
        resnet_model = load_model_safe(RESNET_PATH)
    except Exception as e:
        resnet_err = str(e)

    try:
        convnext_model = load_model_safe(CONVNEXT_PATH)
    except Exception as e:
        convnext_err = str(e)

    a1, a2, a3, a4 = st.columns(4)
    def pill(ok: bool) -> str:
        return "✅ OK" if ok else "❌ KO"

    kpi(a1, "ResNet50", pill(resnet_model is not None), "Chargement modèle")
    kpi(a2, "ConvNeXt", pill(convnext_model is not None), "Chargement modèle")
    kpi(a3, "Input size", f"{IMG_SIZE[0]}×{IMG_SIZE[1]}", "RGB")
    kpi(a4, "Best (offline)", best_model if best_model is not None else "—", "accuracy/loss")

    if resnet_err:
        st.error(f"ResNet50 indisponible: {resnet_err}")
    if convnext_err:
        if "LayerScale" in convnext_err:
            st.warning("ConvNeXt: problème LayerScale — loader custom activé. Si KO: re-sauvegarde en `.keras` ou SavedModel.")
        st.warning(f"ConvNeXt indisponible: {convnext_err}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    L, R = st.columns([0.44, 0.56])

    def predict_with(model: tf.keras.Model, model_name: str, img_rgb_in: np.ndarray, top_k: int):
        img_proc = preprocess_like_notebook(img_rgb_in, IMG_SIZE)
        x = np.expand_dims(img_proc, axis=0)
        if "resnet" in model_name.lower():
            x = tf.keras.applications.resnet.preprocess_input(x)

        t0 = time.time()
        probs = model.predict(x, verbose=0)[0]
        dt = (time.time() - t0) * 1000.0

        probs = normalize_probs(probs)
        idx = np.argsort(probs)[::-1][:top_k]
        df = pd.DataFrame([{"Classe": str(classes[i]), "Probabilité": float(probs[i])} for i in idx])
        return df, dt, str(classes[int(idx[0])]), float(probs[int(idx[0])])

    with L:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Image d’entrée")

        mode = st.radio("Source", ["Upload", "Exemple"], horizontal=True)
        img_rgb = None

        if mode == "Upload":
            up = st.file_uploader("Importer (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if up is not None:
                try:
                    img_rgb = load_image_from_upload(up)
                except Exception as e:
                    st.error(f"Erreur image: {e}")
        else:
            if os.path.isdir(SAMPLES_DIR):
                imgs = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                imgs.sort()
                if imgs:
                    name = st.selectbox("Choisir", imgs, index=0)
                    path = os.path.join(SAMPLES_DIR, name)
                    bgr = cv.imread(path)
                    if bgr is not None:
                        img_rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                else:
                    st.info("Ajoute des images dans sample_images/")
            else:
                st.info("Crée sample_images/ (optionnel).")

        if img_rgb is not None:
            st.image(img_rgb, use_container_width=True)
        else:
            st.info("Sélectionne une image pour activer la prédiction.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("### Paramètres")
        top_k = st.slider("Top-K", 1, 5, 3, 1)
        compare = st.checkbox("Comparer ResNet50 vs ConvNeXt", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with R:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        badge = f'<span class="badge">Best offline: {best_model}</span>' if best_model else ""
        st.markdown(f'<div class="panel-title"><h3 style="margin:0;">Résultats</h3>{badge}</div>', unsafe_allow_html=True)

        if img_rgb is None:
            st.markdown('<div class="small-note">Résultats après upload/sélection.</div>', unsafe_allow_html=True)
        else:
            can_run_resnet = resnet_model is not None
            can_run_convnext = convnext_model is not None

            if not can_run_resnet and not can_run_convnext:
                st.error("Aucun modèle disponible pour prédire.")
            else:
                if compare:
                    if st.button("▶ Lancer", type="primary", use_container_width=True):
                        cA, cB = st.columns(2)

                        if can_run_resnet:
                            df_r, dt_r, top1_r, p1_r = predict_with(resnet_model, "ResNet50", img_rgb, top_k)
                            with cA:
                                st.markdown("#### ResNet50")
                                st.metric("Top-1", top1_r, f"{p1_r:.3f}")
                                st.caption(f"Inference: {dt_r:.1f} ms")
                                chart_r = (
                                    alt.Chart(df_r)
                                    .mark_bar(size=10)
                                    .encode(
                                        x=alt.X("Probabilité:Q", scale=alt.Scale(domain=[0, 1])),
                                        y=alt.Y("Classe:N", sort="-x"),
                                        tooltip=[alt.Tooltip("Classe:N"), alt.Tooltip("Probabilité:Q", format=".3f")],
                                    )
                                    .properties(height=260)
                                )
                                st.altair_chart(chart_r, use_container_width=True)
                        else:
                            with cA:
                                st.warning("ResNet50 indisponible")

                        if can_run_convnext:
                            df_c, dt_c, top1_c, p1_c = predict_with(convnext_model, "ConvNeXt", img_rgb, top_k)
                            with cB:
                                st.markdown("#### ConvNeXt")
                                st.metric("Top-1", top1_c, f"{p1_c:.3f}")
                                st.caption(f"Inference: {dt_c:.1f} ms")
                                chart_c = (
                                    alt.Chart(df_c)
                                    .mark_bar(size=10)
                                    .encode(
                                        x=alt.X("Probabilité:Q", scale=alt.Scale(domain=[0, 1])),
                                        y=alt.Y("Classe:N", sort="-x"),
                                        tooltip=[alt.Tooltip("Classe:N"), alt.Tooltip("Probabilité:Q", format=".3f")],
                                    )
                                    .properties(height=260)
                                )
                                st.altair_chart(chart_c, use_container_width=True)
                        else:
                            with cB:
                                st.warning("ConvNeXt indisponible (LayerScale)")

                        st.markdown("<hr/>", unsafe_allow_html=True)
                        if can_run_resnet and can_run_convnext:
                            winner = "ResNet50" if p1_r >= p1_c else "ConvNeXt"
                            win_prob = max(p1_r, p1_c)
                            st.metric("Winner (sur cette image)", winner, f"{win_prob:.3f}")
                        elif can_run_convnext:
                            st.metric("Winner (sur cette image)", "ConvNeXt", f"{p1_c:.3f}")
                        elif can_run_resnet:
                            st.metric("Winner (sur cette image)", "ResNet50", f"{p1_r:.3f}")

                else:
                    preferred = None
                    if best_model is not None and best_model.lower() == "convnext" and can_run_convnext:
                        preferred = ("ConvNeXt", convnext_model)
                    elif best_model is not None and best_model.lower() == "resnet50" and can_run_resnet:
                        preferred = ("ResNet50", resnet_model)
                    elif can_run_convnext:
                        preferred = ("ConvNeXt", convnext_model)
                    else:
                        preferred = ("ResNet50", resnet_model)

                    model_name, model = preferred
                    if st.button(f"▶ Lancer ({model_name})", type="primary", use_container_width=True):
                        df, dt, top1, p1 = predict_with(model, model_name, img_rgb, top_k)
                        st.metric("Top-1", top1, f"{p1:.3f}")
                        st.caption(f"Inference: {dt:.1f} ms")
                        chart = (
                            alt.Chart(df)
                            .mark_bar(size=10)
                            .encode(
                                x=alt.X("Probabilité:Q", scale=alt.Scale(domain=[0, 1])),
                                y=alt.Y("Classe:N", sort="-x"),
                                tooltip=[alt.Tooltip("Classe:N"), alt.Tooltip("Probabilité:Q", format=".3f")],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)