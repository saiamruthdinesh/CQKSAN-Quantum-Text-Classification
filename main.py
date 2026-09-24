"""
CQKSAN: Enhancing Text Classification through Quantum Transfer Learning
A Hybrid Quantum-Classical Approach with Complex Kernel Self-Attention Networks

Single-File Consolidated Implementation:
- Module 1: Configuration & Hardware Settings
- Module 2: Text Preprocessing
- Module 3: Dataset Generation & Stratified Split (No Data Leakage)
- Module 4: BERT Contextual Embeddings (Transfer Learning)
- Module 5: Baseline LSTM Classifier
- Module 6: Parameterized Quantum Circuit & Complex Kernel Layer
- Module 7: Complex Kernel Self-Attention & Proposed CQKSAN Model
- Module 8: End-to-End Training & Checkpointing
- Module 9: Evaluation (Accuracy, MCC, Confusion Matrix, Graphs, JSON Export)
- Module 10: Real-World Demonstration (CLI & Unseen Review Verification)
"""

import os
import sys
import re
import math
import json
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BERT_MODEL_NAME = "distilbert-base-uncased"
    MAX_SEQ_LENGTH = 128
    EMBEDDING_DIM = 768

    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs"
    CHECKPOINT_DIR = "./checkpoints"

    SAMPLE_SIZE = 1000
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # LSTM Settings
    LSTM_HIDDEN_DIM = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    LSTM_BIDIRECTIONAL = True

    # CQKSAN Settings
    N_QUBITS = 4
    Q_LAYERS = 2
    ATTENTION_HEADS = 4
    DROPOUT = 0.2

    # Training Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    EPOCHS = 6
    WEIGHT_DECAY = 1e-4

    LSTM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_lstm_model.pt")
    CQKSAN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_cqksan_model.pt")
    METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    COMPARISON_PLOT = os.path.join(OUTPUT_DIR, "model_comparison.png")
    CURVES_PLOT = os.path.join(OUTPUT_DIR, "training_curves.png")
    CONFUSION_MATRIX_PLOT = os.path.join(OUTPUT_DIR, "confusion_matrices.png")

for path in [Config.DATA_DIR, Config.OUTPUT_DIR, Config.CHECKPOINT_DIR]:
    os.makedirs(path, exist_ok=True)


# ==============================================================================
# 2. TEXT PREPROCESSING MODULE
# ==============================================================================
class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ==============================================================================
# 3. DATASET MODULE (Stratified Split, No Data Leakage)
# ==============================================================================
class MovieReviewDatasetLoader:
    def __init__(self, data_dir=Config.DATA_DIR, sample_size=Config.SAMPLE_SIZE):
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.preprocessor = TextPreprocessor()

    def get_or_create_dataset(self) -> pd.DataFrame:
        csv_path = os.path.join(self.data_dir, "movie_reviews.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded existing dataset from {csv_path} ({len(df)} samples)")
        else:
            print("Generating balanced movie reviews dataset...")
            df = self._generate_balanced_reviews()
            df.to_csv(csv_path, index=False)
            print(f"Created balanced dataset with {len(df)} samples")

        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=Config.SEED).reset_index(drop=True)

        df["cleaned_text"] = df["text"].apply(self.preprocessor.clean_text)
        return df

    def split_data(self, df: pd.DataFrame):
        train_df, temp_df = train_test_split(
            df,
            test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
            random_state=Config.SEED,
            stratify=df["label"]
        )
        rel_test_ratio = Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=rel_test_ratio,
            random_state=Config.SEED,
            stratify=temp_df["label"]
        )
        print(f"Stratified Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _generate_balanced_reviews(self) -> pd.DataFrame:
        pos = [
            "An absolute cinematic masterpiece with outstanding performances and brilliant direction.",
            "Incredible storytelling, deeply emotional narrative, and captivating cinematography.",
            "Brilliant acting and a phenomenal soundtrack that keeps you hooked from start to finish.",
            "One of the greatest films of the decade, truly inspiring and wonderfully executed.",
            "A charming, witty, and delightful movie that brings pure joy and cinematic excellence."
        ]
        neg = [
            "A complete waste of time with a predictable plot, terrible dialogue, and dreadful acting.",
            "Boring, uninspired, and poorly executed from the opening scene to the end.",
            "Dreadful acting, wooden characters, and a nonsensical storyline that falls flat.",
            "A disappointing disaster with terrible pacing and unconvincing visual effects.",
            "Lacks any emotional depth, completely hollow narrative, and painful to sit through."
        ]
        records = []
        for i in range(500):
            records.append({"text": f"{pos[i % len(pos)]} (Variation {i+1}) A truly memorable experience.", "label": 1})
            records.append({"text": f"{neg[i % len(neg)]} (Variation {i+1}) Utterly dissatisfied with this.", "label": 0})
        return pd.DataFrame(records)


# ==============================================================================
# 4. BERT EMBEDDINGS (Transfer Learning)
# ==============================================================================
class BertEmbeddingExtractor:
    def __init__(self, model_name=Config.BERT_MODEL_NAME, max_length=Config.MAX_SEQ_LENGTH, device=Config.DEVICE):
        self.device = device
        self.max_length = max_length
        print(f"Loading BERT Tokenizer & Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.to(self.device)
        self.bert.eval()

    def get_embeddings(self, texts):
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_embeddings = outputs.last_hidden_state
            cls_embeddings = seq_embeddings[:, 0, :]
        return seq_embeddings, cls_embeddings, attention_mask


# ==============================================================================
# 5. BASELINE LSTM MODEL (Existing System)
# ==============================================================================
class BaselineLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.LSTM_HIDDEN_DIM,
        num_layers=Config.LSTM_NUM_LAYERS,
        dropout=Config.LSTM_DROPOUT,
        bidirectional=Config.LSTM_BIDIRECTIONAL,
        num_classes=2
    ):
        super(BaselineLSTMModel, self).__init__()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            feat = hn[-1, :, :]
        return self.fc(feat)


# ==============================================================================
# 6. COMPLEX QUANTUM KERNEL LAYER (PennyLane / Unitary Simulation)
# ==============================================================================
class ComplexQuantumKernelLayer(nn.Module):
    def __init__(self, n_qubits=Config.N_QUBITS, q_layers=Config.Q_LAYERS):
        super(ComplexQuantumKernelLayer, self).__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.feature_proj = nn.Linear(Config.EMBEDDING_DIM, n_qubits)
        self.weights = nn.Parameter(torch.randn(q_layers, n_qubits, 3) * 0.1)

        if PENNYLANE_AVAILABLE:
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
                    qml.RZ(inputs[i], wires=i)
                for l in range(q_layers):
                    for i in range(n_qubits):
                        qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.circuit = circuit
        else:
            self.circuit = None

    def forward(self, x):
        batch_size = x.shape[0]
        angles = torch.tanh(self.feature_proj(x)) * np.pi

        if PENNYLANE_AVAILABLE and self.circuit is not None:
            q_outs = [torch.stack(self.circuit(angles[b], self.weights)) for b in range(batch_size)]
            return torch.stack(q_outs)
        else:
            cos_theta = torch.cos(angles)
            sin_theta = torch.sin(angles)
            entangled = torch.roll(angles, shifts=1, dims=-1)
            phase = torch.sin(angles + entangled)
            return cos_theta * 0.5 + sin_theta * 0.5 + phase * 0.2


# ==============================================================================
# 7. COMPLEX KERNEL SELF-ATTENTION & CQKSAN MODEL (Proposed System)
# ==============================================================================
class ComplexKernelSelfAttention(nn.Module):
    def __init__(self, embed_dim=Config.EMBEDDING_DIM, num_heads=Config.ATTENTION_HEADS, dropout=Config.DROPOUT):
        super(ComplexKernelSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.kernel_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores * torch.sigmoid(self.kernel_scale)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn_probs = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(context)

class CQKSANModel(nn.Module):
    def __init__(self, embed_dim=Config.EMBEDDING_DIM, n_qubits=Config.N_QUBITS, num_classes=2, dropout=Config.DROPOUT):
        super(CQKSANModel, self).__init__()
        self.attention = ComplexKernelSelfAttention(embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim + n_qubits)
        self.quantum_layer = ComplexQuantumKernelLayer(n_qubits=n_qubits)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + n_qubits, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, seq_embeddings, cls_embeddings, attention_mask=None):
        attn_out = self.attention(seq_embeddings, mask=attention_mask)
        h_classical = self.norm1(seq_embeddings + attn_out)

        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).expand_as(h_classical)
            pooled = (h_classical * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = h_classical.mean(dim=1)

        q_features = self.quantum_layer(cls_embeddings)
        q_features = q_features.to(dtype=pooled.dtype)
        fused = self.norm2(torch.cat([pooled, q_features], dim=-1))
        return self.classifier(fused)


# ==============================================================================
# 8. TRAINING & EVALUATION PIPELINE
# ==============================================================================
def extract_all_features(texts, extractor, batch_size=32):
    all_seq, all_cls, all_mask = [], [], []
    for i in range(0, len(texts), batch_size):
        seq, cls_emb, mask = extractor.get_embeddings(list(texts[i:i+batch_size]))
        all_seq.append(seq.cpu())
        all_cls.append(cls_emb.cpu())
        all_mask.append(mask.cpu())
    return (
    torch.cat(all_seq, dim=0),
    torch.cat(all_cls, dim=0),
    torch.cat(all_mask, dim=0)
)

def train_and_evaluate():
    torch.manual_seed(Config.SEED)
    print("=" * 60)
    print("STEP 1: Preparing Dataset & Stratified Splits")
    print("=" * 60)
    loader = MovieReviewDatasetLoader()
    df = loader.get_or_create_dataset()
    train_df, val_df, test_df = loader.split_data(df)

    print("\n" + "=" * 60)
    print("STEP 2: Extracting Pre-trained BERT Embeddings")
    print("=" * 60)
    extractor = BertEmbeddingExtractor()
    tr_seq, tr_cls, tr_mask = extract_all_features(train_df["cleaned_text"], extractor)
    val_seq, val_cls, val_mask = extract_all_features(val_df["cleaned_text"], extractor)
    test_seq, test_cls, test_mask = extract_all_features(test_df["cleaned_text"], extractor)

    tr_y = torch.tensor(train_df["label"].values, dtype=torch.long)
    val_y = torch.tensor(val_df["label"].values, dtype=torch.long)
    test_y = torch.tensor(test_df["label"].values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(tr_seq, tr_cls, tr_mask, tr_y), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_seq, val_cls, val_mask, val_y), batch_size=Config.BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # --- Train LSTM Baseline ---
    print("\n" + "=" * 60)
    print("STEP 3: Training Baseline LSTM Model (Existing System)")
    print("=" * 60)
    lstm = BaselineLSTMModel().to(Config.DEVICE)
    opt_lstm = torch.optim.AdamW(lstm.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    best_lstm_acc = 0.0
    lstm_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, Config.EPOCHS + 1):
        lstm.train()
        l_loss, l_corr, l_tot = 0.0, 0, 0
        for seq, _, _, labels in train_loader:
            seq, labels = seq.to(Config.DEVICE), labels.to(Config.DEVICE)
            opt_lstm.zero_grad()
            out = lstm(seq)
            loss = criterion(out, labels)
            loss.backward()
            opt_lstm.step()
            l_loss += loss.item() * seq.size(0)
            l_corr += (torch.argmax(out, dim=1) == labels).sum().item()
            l_tot += labels.size(0)

        lstm.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for seq, _, _, labels in val_loader:
                seq, labels = seq.to(Config.DEVICE), labels.to(Config.DEVICE)
                out = lstm(seq)
                v_loss += criterion(out, labels).item() * seq.size(0)
                v_corr += (torch.argmax(out, dim=1) == labels).sum().item()
                v_tot += labels.size(0)

        tr_acc, va_acc = l_corr / l_tot, v_corr / v_tot
        lstm_history["train_loss"].append(l_loss / l_tot)
        lstm_history["train_acc"].append(tr_acc)
        lstm_history["val_loss"].append(v_loss / v_tot)
        lstm_history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d}/{Config.EPOCHS:02d} | Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        if va_acc >= best_lstm_acc:
            best_lstm_acc = va_acc
            torch.save(lstm.state_dict(), Config.LSTM_CHECKPOINT)

    # --- Train Proposed CQKSAN ---
    print("\n" + "=" * 60)
    print("STEP 4: Training Proposed CQKSAN Model (Quantum-Classical Hybrid)")
    print("=" * 60)
    cqksan = CQKSANModel().to(Config.DEVICE)
    opt_cqksan = torch.optim.AdamW(cqksan.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    best_cqksan_acc = 0.0
    cqksan_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, Config.EPOCHS + 1):
        cqksan.train()
        c_loss, c_corr, c_tot = 0.0, 0, 0
        for seq, cls_emb, mask, labels in train_loader:
            seq, cls_emb, mask, labels = seq.to(Config.DEVICE), cls_emb.to(Config.DEVICE), mask.to(Config.DEVICE), labels.to(Config.DEVICE)
            opt_cqksan.zero_grad()
            out = cqksan(seq, cls_emb, mask)
            loss = criterion(out, labels)
            loss.backward()
            opt_cqksan.step()
            c_loss += loss.item() * seq.size(0)
            c_corr += (torch.argmax(out, dim=1) == labels).sum().item()
            c_tot += labels.size(0)

        cqksan.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for seq, cls_emb, mask, labels in val_loader:
                seq, cls_emb, mask, labels = seq.to(Config.DEVICE), cls_emb.to(Config.DEVICE), mask.to(Config.DEVICE), labels.to(Config.DEVICE)
                out = cqksan(seq, cls_emb, mask)
                v_loss += criterion(out, labels).item() * seq.size(0)
                v_corr += (torch.argmax(out, dim=1) == labels).sum().item()
                v_tot += labels.size(0)

        tr_acc, va_acc = c_corr / c_tot, v_corr / v_tot
        cqksan_history["train_loss"].append(c_loss / c_tot)
        cqksan_history["train_acc"].append(tr_acc)
        cqksan_history["val_loss"].append(v_loss / v_tot)
        cqksan_history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d}/{Config.EPOCHS:02d} | Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        if va_acc >= best_cqksan_acc:
            best_cqksan_acc = va_acc
            torch.save(cqksan.state_dict(), Config.CQKSAN_CHECKPOINT)

    # --- Step 5: Test Evaluation & Metrics Verification ---
    print("\n" + "=" * 60)
    print("STEP 5: Model Evaluation on Unseen Test Split (Accuracy & MCC)")
    print("=" * 60)
    lstm.load_state_dict(torch.load(Config.LSTM_CHECKPOINT, map_location=Config.DEVICE))
    lstm.eval()

    cqksan.load_state_dict(torch.load(Config.CQKSAN_CHECKPOINT, map_location=Config.DEVICE))
    cqksan.eval()

    with torch.no_grad():
        lstm_preds = torch.argmax(lstm(test_seq.to(Config.DEVICE)), dim=1).cpu().numpy()
        cqksan_preds = torch.argmax(cqksan(test_seq.to(Config.DEVICE), test_cls.to(Config.DEVICE), test_mask.to(Config.DEVICE)), dim=1).cpu().numpy()

    y_true = test_y.numpy()
    lstm_acc = accuracy_score(y_true, lstm_preds)
    lstm_mcc = matthews_corrcoef(y_true, lstm_preds)
    _, _, lstm_f1, _ = precision_recall_fscore_support(y_true, lstm_preds, average="binary")

    cqksan_acc = accuracy_score(y_true, cqksan_preds)
    cqksan_mcc = matthews_corrcoef(y_true, cqksan_preds)
    _, _, cqksan_f1, _ = precision_recall_fscore_support(y_true, cqksan_preds, average="binary")

    acc_boost = ((cqksan_acc - lstm_acc) / max(lstm_acc, 1e-5)) * 100
    mcc_boost = ((cqksan_mcc - lstm_mcc) / max(lstm_mcc, 1e-5)) * 100

    metrics_dict = {
        "Baseline_LSTM": {"Accuracy": round(float(lstm_acc), 4), "MCC": round(float(lstm_mcc), 4), "F1": round(float(lstm_f1), 4)},
        "Proposed_CQKSAN": {"Accuracy": round(float(cqksan_acc), 4), "MCC": round(float(cqksan_mcc), 4), "F1": round(float(cqksan_f1), 4)},
        "Improvements": {"Accuracy_Gain_Percent": round(acc_boost, 2), "MCC_Gain_Percent": round(mcc_boost, 2)}
    }

    print(f"Existing LSTM:     Accuracy = {lstm_acc*100:.2f}%, MCC = {lstm_mcc:.4f}, F1 = {lstm_f1*100:.2f}%")
    print(f"Proposed CQKSAN:   Accuracy = {cqksan_acc*100:.2f}%, MCC = {cqksan_mcc:.4f}, F1 = {cqksan_f1*100:.2f}%")
    print(f"Relative Accuracy Gain: {acc_boost:+.2f}%")
    print(f"Relative MCC Gain:      {mcc_boost:+.2f}%")

    with open(Config.METRICS_JSON, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics summary to {Config.METRICS_JSON}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(confusion_matrix(y_true, lstm_preds), annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Existing System (LSTM) Confusion Matrix")
    sns.heatmap(confusion_matrix(y_true, cqksan_preds), annot=True, fmt="d", cmap="Purples", ax=axes[1])
    axes[1].set_title("Proposed System (CQKSAN) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Config.CONFUSION_MATRIX_PLOT, dpi=300)
    plt.close()

    # Plot Comparison Bar Chart
    plt.figure(figsize=(7, 4))
    x = np.arange(3)
    plt.bar(x - 0.15, [lstm_acc*100, lstm_mcc*100, lstm_f1*100], width=0.3, label="LSTM", color="#e67e22")
    plt.bar(x + 0.15, [cqksan_acc*100, cqksan_mcc*100, cqksan_f1*100], width=0.3, label="CQKSAN", color="#2980b9")
    plt.xticks(x, ["Accuracy (%)", "MCC (x100)", "F1-Score (%)"])
    plt.title("Performance Comparison: LSTM vs CQKSAN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Config.COMPARISON_PLOT, dpi=300)
    plt.close()
    print("Saved confusion matrices and comparison charts to ./outputs/")

    # --- Step 6: Real-World Demonstration on Unseen Reviews ---
    print("\n" + "=" * 60)
    print("STEP 6: Real-World Demonstration on Unseen Reviews")
    print("=" * 60)
    preprocessor = TextPreprocessor()
    unseen_tests = [
        ("An extraordinary cinematic achievement with breathtaking acting, score, and directing.", "Positive"),
        ("A catastrophic disaster with nonsensical plotlines, terrible acting, and boring pacing.", "Negative"),
        ("A genuinely moving and visually inspiring masterpiece that left me speechless.", "Positive"),
        ("Unbearably dull and painful to sit through. Total waste of time and money.", "Negative")
    ]

    labels = ["Negative", "Positive"]
    for text, expected in unseen_tests:
        cleaned = preprocessor.clean_text(text)
        s, c, m = extractor.get_embeddings([cleaned])
        with torch.no_grad():
            cq_p = torch.softmax(cqksan(s.to(Config.DEVICE), c.to(Config.DEVICE), m.to(Config.DEVICE)), dim=-1)[0]
            ls_p = torch.softmax(lstm(s.to(Config.DEVICE)), dim=-1)[0]
        cq_pred = torch.argmax(cq_p).item()
        ls_pred = torch.argmax(ls_p).item()
        print(f"\nReview: \"{text}\"")
        print(f"Ground Truth:      {expected}")
        print(f"CQKSAN Prediction: {labels[cq_pred]} ({cq_p[cq_pred]*100:.1f}% confidence)")
        print(f"LSTM Prediction:   {labels[ls_pred]} ({ls_p[ls_pred]*100:.1f}% confidence)")

if __name__ == "__main__":
    train_and_evaluate()
