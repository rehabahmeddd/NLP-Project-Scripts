"""
Competition Prediction Script - Rehab Model (FastText-based)
Generates predictions in competition format from the Rehab trained model
"""

import torch
import torch.nn as nn
import pickle
import pandas as pd
import unicodedata
from typing import List
from transformers import AutoTokenizer, AutoModel

# ==================== CONFIGURATION ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_PATH = "model.pt"
BERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"

# ==================== LOAD CHECKPOINT ====================

print(f"Loading model checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

label2id = checkpoint.get('label2id')
id2label = checkpoint.get('id2label')
char2id = checkpoint.get('char2id')
id2char = checkpoint.get('id2char')
NUM_LABELS = checkpoint.get('num_labels')
VOCAB_SIZE = checkpoint.get('vocab_size')

print(f"Loaded vocabularies: {NUM_LABELS} labels, {VOCAB_SIZE} characters")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

# Load competition mappings
print("Loading competition label mappings...")
diacritic2id = pickle.load(open('diacritic2id.pickle', 'rb'))
id2diacritic = {v: k for k, v in diacritic2id.items()}
print(f"Competition labels: {len(diacritic2id)}")

# ==================== HELPER FUNCTIONS ====================

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

def is_diacritic(ch: str) -> bool:
    return ch in ARABIC_DIACRITICS

def is_arabic_letter(ch: str) -> bool:
    if not ("\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"):
        return False
    if is_diacritic(ch):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("L")

def line_to_struct(line: str):
    base_chars = []
    for ch in line:
        if not is_diacritic(ch):
            base_chars.append(ch)
    
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
        if w_idx >= 0 and w_idx < len(words):
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

# ==================== MODEL ====================

from torchcrf import CRF

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
        
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=char2id.get("<PAD>", 0))
        
        self.feat_proj = nn.Sequential(
            nn.Linear(num_enhanced_feats, feat_hidden_dim),
            nn.LayerNorm(feat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feat_hidden_dim, feat_hidden_dim),
            nn.ReLU()
        )
        
        input_dim = emb_dim + feat_hidden_dim + self.bert_hidden_size
        
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if lstm_layers > 1 else 0)
        
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

    def forward(self, batch):
        char_ids = batch["char_ids"].to(DEVICE)
        enhanced_feats = batch["enhanced_feats"].to(DEVICE)
        plain_text = batch["plain_text"]
        words_list = batch["words"]
        char2word = batch["char2word"].to(DEVICE)
        
        B, T = char_ids.shape
        
        encoding = tokenizer(words_list, is_split_into_words=True, padding=True,
                            truncation=True, return_tensors="pt").to(DEVICE)
        bert_out = self.bert(**encoding)
        token_embeddings = bert_out.last_hidden_state
        
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
        
        binary_logits = self.binary_head(lstm_out).squeeze(-1)
        multi_logits = self.multi_head(lstm_out)
        
        return binary_logits, multi_logits

# ==================== LOAD MODEL ====================

print("Loading model architecture...")
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)

model = EnhancedDiacritizer(
    bert_model=bert_model,
    vocab_size=VOCAB_SIZE,
    num_labels=NUM_LABELS,
    num_enhanced_feats=NUM_ENHANCED_FEATURES,
    emb_dim=300,
    feat_hidden_dim=48,
    lstm_hidden_dim=256,
    lstm_layers=3,
    dropout=0.3,
    freeze_bert=True,
    use_crf=False
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded!")

# ==================== LOAD TEST DATA ====================

print("\nLoading test data...")
with open('dataset_no_diacritics.txt', 'r', encoding='utf-8') as f:
    test_lines = [line.rstrip('\n') for line in f.readlines() if line.strip()]

print(f"Loaded {len(test_lines)} lines")

# ==================== GENERATE PREDICTIONS ====================

print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

all_predictions = []
char_id = 0

for line_idx, text in enumerate(test_lines):
    if (line_idx + 1) % max(1, len(test_lines) // 10) == 0:
        print(f"  Processing line {line_idx + 1}/{len(test_lines)}")
    
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
    
    model.eval()
    with torch.no_grad():
        binary_logits, multi_logits = model(batch)
        pred_ids = multi_logits[0].argmax(dim=-1).tolist()
    
    # Map to competition labels - ONLY for Arabic letters
    for ch, pred_id in zip(plain, pred_ids):
        if is_arabic_letter(ch):
            model_label = id2label.get(pred_id, "NONE")
            
            if model_label == "NONE" or model_label == "":
                comp_id = diacritic2id.get('', 14)
            else:
                comp_id = diacritic2id.get(model_label, 14)
            
            all_predictions.append({
                'id': char_id,
                'label': comp_id
            })
            char_id += 1

print(f"\n[OK] Generated {len(all_predictions)} predictions")

# ==================== SAVE PREDICTIONS ====================

print("\nSaving predictions...")
output_df = pd.DataFrame(all_predictions)
output_df.to_csv('predictions.csv', index=False)

print(f"[OK] Saved to: predictions.csv")
print(f"  Total predictions: {len(output_df)}")

# ==================== GENERATE DIACRITIZED TEXT ====================

print("\nGenerating diacritized text...")

diacritized_text = ""
char_idx = 0

for text in test_lines:
    for ch in text:
        if is_arabic_letter(ch):
            # Find prediction for this character
            if char_idx < len(all_predictions):
                comp_id = all_predictions[char_idx]['label']
                diacritic = id2diacritic.get(comp_id, '')
                diacritized_text += ch + diacritic
            else:
                diacritized_text += ch
            char_idx += 1
        else:
            diacritized_text += ch
    diacritized_text += '\n'

# Save diacritized text
with open('predictions_diacritized.txt', 'w', encoding='utf-8') as f:
    f.write(diacritized_text)

print(f"[OK] Saved diacritized text to: predictions_diacritized.txt")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"Files created:")
print(f"  - predictions.csv (competition format)")
print(f"  - predictions_diacritized.txt (diacritized text)")
