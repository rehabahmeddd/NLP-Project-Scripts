# app.py
# Streamlit UI for demo.py diacritization pipeline (single file)

import io
import os
import pickle
import unicodedata
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# UI config
# ------------------------------
st.set_page_config(
    page_title="Arabic Diacritization Studio",
    page_icon="🌍",
    layout="wide",
)

# ------------------------------
# Constants (same as demo.py)
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PATH = "model.pt"
DEFAULT_DIACRITIC2ID_PATH = "diacritic2id.pickle"
BERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"

ARABIC_DIACRITICS = set([
    "\u064b", "\u064c", "\u064d", "\u064e", "\u064f",
    "\u0650", "\u0651", "\u0652", "\u0670"
])

SUN_LETTERS = set("تثدذرزسشصضطظنل")
MOON_LETTERS = set("ءأإابجحخعغفقكمهوي")
ARABIC_PREFIXES = set("وفبكلس")
ARABIC_SUFFIXES = set("هاكني")
ALEF_VARIANTS = set("اأإآى")
WAW_YA = set("وي")
TA_MARBUTA = "ة"
HAMZA_VARIANTS = set("ءأإؤئ")
NUM_ENHANCED_FEATURES = 24


# ------------------------------
# Helpers (same logic as demo.py)
# ------------------------------
def is_diacritic(ch: str) -> bool:
    return ch in ARABIC_DIACRITICS


def is_arabic_letter(ch: str) -> bool:
    if not ("\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"):
        return False
    if is_diacritic(ch):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("L")


def strip_diacritics_keep_text(s: str) -> str:
    return "".join(ch for ch in s if not is_diacritic(ch))


def line_to_struct(line: str):
    # identical idea as demo.py: remove diacritics, split to words, build char2word
    base_chars = [ch for ch in line if not is_diacritic(ch)]
    plain_text = "".join(base_chars)
    words = plain_text.split()

    char2word = []
    current_word_idx = -1
    inside_word = False

    for ch in plain_text:
        if ch.isspace():
            char2word.append(-1)
            if inside_word:
                inside_word = False
        else:
            if not inside_word:
                inside_word = True
                current_word_idx += 1
            char2word.append(current_word_idx)

    return base_chars, plain_text, words, char2word


def extract_enhanced_features(plain_text: str, char2word: List[int], words: List[str]) -> List[List[float]]:
    # same 24-dim feature design as demo.py
    features = []
    n = len(plain_text)

    word_starts = set()
    word_ends = set()
    pos = 0
    for word in words:
        word_starts.add(pos)
        word_ends.add(pos + len(word) - 1)
        pos += len(word) + 1

    for i, ch in enumerate(plain_text):
        f = []

        f.append(1.0 if is_arabic_letter(ch) else 0.0)
        f.append(1.0 if ch.isspace() else 0.0)
        f.append(1.0 if ch.isdigit() else 0.0)
        f.append(1.0 if unicodedata.category(ch).startswith("P") else 0.0)

        f.append(1.0 if ch in SUN_LETTERS else 0.0)
        f.append(1.0 if ch in MOON_LETTERS else 0.0)
        f.append(1.0 if ch in ALEF_VARIANTS else 0.0)
        f.append(1.0 if ch in HAMZA_VARIANTS else 0.0)
        f.append(1.0 if ch in WAW_YA else 0.0)
        f.append(1.0 if ch == TA_MARBUTA else 0.0)

        is_word_start = i in word_starts
        is_word_end = i in word_ends
        f.append(1.0 if is_word_start else 0.0)
        f.append(1.0 if is_word_end else 0.0)
        f.append(1.0 if is_word_start and is_word_end else 0.0)

        w_idx = char2word[i] if i < len(char2word) else -1
        if 0 <= w_idx < len(words):
            word_len = len(words[w_idx])
            word_start_pos = sum(len(words[j]) + 1 for j in range(w_idx))
            pos_in_word = i - word_start_pos
            f.append(pos_in_word / max(word_len - 1, 1) if word_len > 1 else 0.5)
        else:
            f.append(0.0)

        f.append(1.0 if is_word_start and ch in ARABIC_PREFIXES else 0.0)
        f.append(1.0 if is_word_end and ch in ARABIC_SUFFIXES else 0.0)

        is_alef_lam = False
        if is_word_start and ch == 'ا' and i + 1 < n and plain_text[i + 1] == 'ل':
            is_alef_lam = True
        if i > 0 and plain_text[i - 1] == 'ا' and ch == 'ل' and (i - 1) in word_starts:
            is_alef_lam = True
        f.append(1.0 if is_alef_lam else 0.0)

        after_al = False
        if i >= 2 and w_idx >= 0:
            word_start_pos = sum(len(words[j]) + 1 for j in range(w_idx))
            if i - word_start_pos == 2:
                if plain_text[word_start_pos:word_start_pos+2] == "ال":
                    after_al = True
        f.append(1.0 if after_al else 0.0)

        f.append(1.0 if ch == TA_MARBUTA and is_word_end else 0.0)

        prev_ch = plain_text[i - 1] if i > 0 else ' '
        f.append(1.0 if is_arabic_letter(prev_ch) else 0.0)
        f.append(1.0 if prev_ch in ALEF_VARIANTS else 0.0)

        next_ch = plain_text[i + 1] if i + 1 < n else ' '
        f.append(1.0 if is_arabic_letter(next_ch) else 0.0)
        f.append(1.0 if next_ch.isspace() or i + 1 >= n else 0.0)
        f.append(1.0 if next_ch == TA_MARBUTA else 0.0)

        features.append(f)

    return features


# ------------------------------
# Model (same architecture as demo.py)
# ------------------------------
class EnhancedDiacritizer(nn.Module):
    def __init__(self, bert_model, vocab_size, num_labels, num_enhanced_feats=24,
                 emb_dim=300, feat_hidden_dim=48, lstm_hidden_dim=256,
                 lstm_layers=3, dropout=0.3, freeze_bert=True, use_crf=False):
        super().__init__()
        self.bert = bert_model
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.bert_hidden_size = self.bert.config.hidden_size
        self.use_crf = use_crf
        self.num_labels = num_labels

        # padding_idx will be set after char2id is loaded; we pass it in later via init args
        self.char_emb = nn.Embedding(vocab_size, emb_dim)

        self.feat_proj = nn.Sequential(
            nn.Linear(num_enhanced_feats, feat_hidden_dim),
            nn.LayerNorm(feat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feat_hidden_dim, feat_hidden_dim),
            nn.ReLU()
        )

        input_dim = emb_dim + feat_hidden_dim + self.bert_hidden_size

        self.lstm = nn.LSTM(
            input_dim, lstm_hidden_dim, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.binary_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(lstm_hidden_dim, 1)
        )

        self.multi_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(lstm_hidden_dim, num_labels)
        )

    def forward(self, batch, tokenizer: AutoTokenizer):
        char_ids = batch["char_ids"].to(DEVICE)
        enhanced_feats = batch["enhanced_feats"].to(DEVICE)
        words_list = batch["words"]
        char2word = batch["char2word"].to(DEVICE)

        B, T = char_ids.shape

        encoding = tokenizer(
            words_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        bert_out = self.bert(**encoding)
        token_embeddings = bert_out.last_hidden_state  # (B, tokens, H)

        bert_char_context = torch.zeros((B, T, self.bert_hidden_size), device=DEVICE)

        for i in range(B):
            word_ids = encoding.word_ids(batch_index=i)
            num_words = len(words_list[i])
            if num_words == 0:
                continue

            tokens = token_embeddings[i]
            H = tokens.size(-1)
            word_sums = torch.zeros((num_words, H), device=DEVICE)
            word_counts = torch.zeros((num_words, 1), device=DEVICE)

            for tok_idx, w_id in enumerate(word_ids):
                if w_id is not None and w_id < num_words:
                    word_sums[w_id] += tokens[tok_idx]
                    word_counts[w_id] += 1.0

            word_counts = torch.clamp(word_counts, min=1.0)
            word_embs = word_sums / word_counts

            char_indices = char2word[i, :T]
            valid_mask = (char_indices >= 0) & (char_indices < num_words)
            valid_chars = char_indices[valid_mask]
            if len(valid_chars) > 0:
                bert_char_context[i, valid_mask] = word_embs[valid_chars]

        char_embs = self.char_emb(char_ids)
        feat_proj = self.feat_proj(enhanced_feats)
        x = torch.cat([char_embs, feat_proj, bert_char_context], dim=-1)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        binary_logits = self.binary_head(lstm_out).squeeze(-1)  # (B, T)
        multi_logits = self.multi_head(lstm_out)                # (B, T, num_labels)
        return binary_logits, multi_logits


# ------------------------------
# Loading (cached)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_assets(model_path: str, diacritic2id_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not os.path.exists(diacritic2id_path):
        raise FileNotFoundError(f"Missing mapping file: {diacritic2id_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    id2label = checkpoint.get("id2label")
    char2id = checkpoint.get("char2id")
    num_labels = checkpoint.get("num_labels")
    vocab_size = checkpoint.get("vocab_size")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)

    model = EnhancedDiacritizer(
        bert_model=bert_model,
        vocab_size=vocab_size,
        num_labels=num_labels,
        num_enhanced_feats=NUM_ENHANCED_FEATURES,
        emb_dim=300,
        feat_hidden_dim=48,
        lstm_hidden_dim=256,
        lstm_layers=3,
        dropout=0.3,
        freeze_bert=True,
        use_crf=False
    ).to(DEVICE)

    # set padding_idx like your script does (uses <PAD> if present)
    pad_idx = char2id.get("<PAD>", 0)
    model.char_emb = nn.Embedding(vocab_size, 300, padding_idx=pad_idx).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diacritic2id = pickle.load(open(diacritic2id_path, "rb"))
    id2diacritic = {v: k for k, v in diacritic2id.items()}

    return {
        "model": model,
        "tokenizer": tokenizer,
        "char2id": char2id,
        "id2label": id2label,
        "diacritic2id": diacritic2id,
        "id2diacritic": id2diacritic,
        "num_labels": num_labels,
        "vocab_size": vocab_size,
    }


# ------------------------------
# Prediction core (same mapping semantics as demo.py)
# ------------------------------
def predict_line(text: str, assets: Dict[str, Any]) -> Dict[str, Any]:
    model = assets["model"]
    tokenizer = assets["tokenizer"]
    char2id = assets["char2id"]
    id2label = assets["id2label"]
    diacritic2id = assets["diacritic2id"]
    id2diacritic = assets["id2diacritic"]

    base_chars, plain, words, char2word = line_to_struct(text)
    char_ids = [char2id.get(ch, char2id.get("<UNK>", 1)) for ch in base_chars]
    enhanced_feats = extract_enhanced_features(plain, char2word, words)

    batch = {
        "char_ids": torch.tensor([char_ids], dtype=torch.long),
        "enhanced_feats": torch.tensor([enhanced_feats], dtype=torch.float32),
        "mask": torch.ones((1, len(char_ids)), dtype=torch.float32),
        "plain_text": [plain],
        "words": [words],
        "char2word": torch.tensor([char2word], dtype=torch.long),
    }

    with torch.no_grad():
        binary_logits, multi_logits = model(batch, tokenizer=tokenizer)
        probs = torch.softmax(multi_logits[0], dim=-1).detach().cpu().numpy()  # (T, num_labels)
        pred_ids = multi_logits[0].argmax(dim=-1).tolist()

    # Build char-level results (for Arabic letters only: competition format)
    rows = []
    diacritized_text = ""
    arabic_char_counter = 0

    for idx, (ch, pred_id) in enumerate(zip(plain, pred_ids)):
        if is_arabic_letter(ch):
            model_label = id2label.get(pred_id, "NONE")
            if model_label == "NONE" or model_label == "":
                comp_id = diacritic2id.get("", 14)
            else:
                comp_id = diacritic2id.get(model_label, 14)

            rows.append({
                "id": None,  # filled by batch runner
                "label": comp_id,
                "char": ch,
                "model_label": model_label,
                "p_top1": float(probs[idx, pred_id]) if idx < probs.shape[0] else None,
            })

            diacritized_text += ch + id2diacritic.get(comp_id, "")
            arabic_char_counter += 1
        else:
            diacritized_text += ch

    return {
        "plain": plain,
        "words": words,
        "char2word": char2word,
        "enhanced_feats": enhanced_feats,
        "rows": rows,
        "diacritized": diacritized_text,
    }


def predict_lines(lines: List[str], assets: Dict[str, Any]) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    all_predictions = []
    diacritized_all = []
    running_id = 0

    for line in lines:
        out = predict_line(line, assets)
        # fill ids in the same sequential style demo.py uses
        for r in out["rows"]:
            all_predictions.append({"id": running_id, "label": r["label"]})
            running_id += 1
        diacritized_all.append(out["diacritized"])

    pred_df = pd.DataFrame(all_predictions)
    diacritized_text = "\n".join(diacritized_all) + ("\n" if diacritized_all else "")

    # optional "debug" view (extra columns)
    debug_df = None
    if lines:
        # just show first line’s detailed info
        dbg = predict_line(lines[0], assets)
        debug_df = pd.DataFrame(dbg["rows"])

    return pred_df, diacritized_text, debug_df


# ------------------------------
# UI
# ------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; }
      .big-title { font-size: 2.0rem; font-weight: 750; margin-bottom: 0.2rem; }
      .subtle { opacity: 0.75; }
      textarea { direction: rtl !important; font-size: 1.05rem !important; }
      .rtl { direction: rtl; text-align: right; font-size: 1.15rem; line-height: 2.0; }
      .pill { display:inline-block; padding: 0.22rem 0.55rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.15); margin-right: 0.35rem; font-size: 0.85rem; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title"> Arabic Diacritization Studio</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtle">Device: <b>{DEVICE}</b> • AraBERT: <code>{BERT_MODEL_NAME}</code></div>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
      /* Push page content below Streamlit's top header */
      .block-container { padding-top: 4.5rem !important; }

      /* Optional: make the header less “thick” / cleaner */
      [data-testid="stHeader"] { height: 3.25rem; }

      /* If you still see overlap in some themes, force a bit more top spacing */
      .big-title { margin-top: 0.25rem; }

      textarea { direction: rtl !important; font-size: 1.05rem !important; }
      .rtl { direction: rtl; text-align: right; font-size: 1.15rem; line-height: 2.0; }
      .pill { display:inline-block; padding: 0.22rem 0.55rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.15); margin-right: 0.35rem; font-size: 0.85rem; opacity: 0.9; }
      .big-title { font-size: 2.0rem; font-weight: 750; margin-bottom: 0.2rem; }
      .subtle { opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Assets")
    model_path = st.text_input("model.pt path", value=DEFAULT_MODEL_PATH)
    mapping_path = st.text_input("diacritic2id.pickle path", value=DEFAULT_DIACRITIC2ID_PATH)

    st.markdown("---")
    st.markdown("###  UI options")
    show_debug = st.toggle("Show debug tables", value=True)
    show_features = st.toggle("Show 24 extracted features (first line)", value=False)
    max_lines = st.number_input("Max lines per batch", min_value=1, max_value=20000, value=2000, step=100)
    st.caption("Tip: if you upload huge files, start smaller to keep UI responsive.")

# Load assets
try:
    assets = load_assets(model_path, mapping_path)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded: {assets['num_labels']} labels • {assets['vocab_size']} characters")

tab1, tab2 = st.tabs(["✨ Single text", "📦 Batch file (txt)"])

with tab1:
    colL, colR = st.columns([1.05, 0.95], gap="large")

    with colL:
        st.markdown("#### Input (Arabic, no diacritics)")
        sample = "ذهب الولد إلى المدرسة وقرأ كتابا"
        text = st.text_area("",
                            value=sample,
                            height=180,
                            placeholder="اكتب النص هنا...")

        run = st.button("Diacritize", type="primary", use_container_width=True)

    with colR:
        st.markdown("#### Output")
        if run and text.strip():
            out = predict_line(text.strip(), assets)

            st.markdown('<div class="rtl">', unsafe_allow_html=True)
            st.write(out["diacritized"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Quick chips
            n_chars = sum(1 for ch in out["plain"] if is_arabic_letter(ch))
            st.markdown(
                f"""
                <span class="pill">Arabic letters: {n_chars}</span>
                <span class="pill">Words: {len(out["words"])}</span>
                <span class="pill">Chars total: {len(out["plain"])}</span>
                """,
                unsafe_allow_html=True
            )

            if show_debug:
                st.markdown("##### Debug (first-line character predictions)")
                dbg = pd.DataFrame(out["rows"])
                st.dataframe(dbg, use_container_width=True, height=240)

            if show_features:
                st.markdown("##### 24 extracted features (per character, incl. spaces/punct)")
                feat_df = pd.DataFrame(out["enhanced_feats"])
                feat_df.insert(0, "char", list(out["plain"]))
                st.dataframe(feat_df, use_container_width=True, height=260)

            # Downloads
            pred_df, dia_txt, _ = predict_lines([text.strip()], assets)
            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            txt_bytes = dia_txt.encode("utf-8")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download predictions.csv",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d2:
                st.download_button(
                    "Download predictions_diacritized.txt",
                    data=txt_bytes,
                    file_name="predictions_diacritized.txt",
                    mime="text/plain",
                    use_container_width=True
                )

with tab2:
    st.markdown("#### Upload a .txt file (one sentence per line)")
    up = st.file_uploader("dataset_no_diacritics.txt", type=["txt"])

    if up is not None:
        raw = up.read().decode("utf-8", errors="replace").splitlines()
        lines = [ln.rstrip("\n") for ln in raw if ln.strip()]
        if len(lines) > max_lines:
            st.warning(f"File has {len(lines)} non-empty lines; UI limit is {max_lines}. Truncating for this run.")
            lines = lines[:max_lines]

        st.write(f"Lines to process: **{len(lines)}**")

        if st.button("Run batch diacritization", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                pred_df, dia_txt, debug_df = predict_lines(lines, assets)

            st.success(f"Done. Generated **{len(pred_df)}** competition-format predictions.")

            c1, c2 = st.columns([0.55, 0.45], gap="large")
            with c1:
                st.markdown("##### Preview (diacritized)")
                preview = "\n".join(dia_txt.splitlines()[:8])
                st.markdown('<div class="rtl">', unsafe_allow_html=True)
                st.write(preview)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                if show_debug and debug_df is not None:
                    st.markdown("##### Debug (first line only)")
                    st.dataframe(debug_df, use_container_width=True, height=260)

            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            txt_bytes = dia_txt.encode("utf-8")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download predictions.csv",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d2:
                st.download_button(
                    "Download predictions_diacritized.txt",
                    data=txt_bytes,
                    file_name="predictions_diacritized.txt",
                    mime="text/plain",
                    use_container_width=True
                )

st.caption("This UI mirrors the demo.py logic: build per-line batch dict → model() → argmax labels → map to competition IDs → emit predictions.csv + diacritized text. 🪄")
