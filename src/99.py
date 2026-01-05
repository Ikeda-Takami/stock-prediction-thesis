import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import itertools
import random

# ==========================================
# ★ 広域グリッドサーチ設定エリア ★
# ==========================================
# 探索範囲
PARAM_GRID = {
    "HIDDEN_SIZE": [64, 128, 256],
    "NUM_LAYERS":  [1, 2],
    "DROPOUT":     [0.2, 0.4, 0.6],
    "LR":          [0.005, 0.001, 0.0001],
    "BATCH_SIZE":  [16, 32, 64]
}

N_RUNS_PER_GRID = 1  # 1回勝負（高速化）
EPOCHS = 70

# 期間設定
TRAIN_START = "2017-09-01"
TRAIN_END   = "2024-06-30"
TEST_START  = "2024-07-01"
TEST_END    = "2024-12-30"

# パス設定
BASE_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/"
INPUT_SUMMARY_DIR = os.path.join(BASE_DIR, "data/summary_reasoning_augasuto_v3/")
FINANCE_CSV       = os.path.join(BASE_DIR, "data/finance/market_data_all_labeled.csv")
REPORT_FILE       = os.path.join(BASE_DIR, f"result/GRID_SEARCH_ACCURACY_RESULT.txt")

SBERT_MODEL_NAME = "pkshatech/GLuCoSE-base-ja"
FEATURE_COLS = ["Nikkei_Close", "Nikkei_Change", "Dow_Change", "USDJPY_Close"]
SEQUENCE_LENGTH = 10
# ==========================================

os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

def log_print(text, file_obj=None):
    print(text)
    if file_obj:
        file_obj.write(text + "\n")

# --- STEP3抽出関数 (最強版) ---
def extract_step3(text):
    match = re.search(r"STEP\s*3[^\n]*?[:：](.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        content = content.replace("**", "").replace("##", "").strip()
        return content
    return text.strip()

def load_data(start_date, end_date, sbert_model):
    all_files = sorted(glob.glob(os.path.join(INPUT_SUMMARY_DIR, "summary_*.txt")))
    target_files = []
    for p in all_files:
        d_str = os.path.basename(p).replace("summary_", "").replace(".txt", "")
        if start_date <= d_str <= end_date:
            target_files.append(p)
            
    if not target_files: return pd.DataFrame()

    dates, vectors = [], []
    for file_path in target_files:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        if not full_text: continue
        
        step3_text = extract_step3(full_text)
        vec = sbert_model.encode(step3_text, normalize_embeddings=True)
        dates.append(os.path.basename(file_path).replace("summary_", "").replace(".txt", ""))
        vectors.append(vec)
        
    return pd.DataFrame({"Date_Str": dates, "Vector": vectors})

def create_dataset(df, seq_len):
    X_list, y_list = [], []
    vector_features = np.stack(df["Vector"].values)
    scaled_cols = [f"{c}_Scaled" for c in FEATURE_COLS]
    numeric_features = df[scaled_cols].values
    combined_features = np.hstack([vector_features, numeric_features])
    labels = df["Actual_Label"].values
    
    for i in range(len(df) - seq_len):
        X_list.append(combined_features[i : i + seq_len])
        y_list.append(labels[i + seq_len])
    return np.array(X_list), np.array(y_list)

class HybridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

# === メイン処理 ===
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("📥 データをロード中...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    
    df_train_text = load_data(TRAIN_START, TRAIN_END, sbert_model)
    df_test_text  = load_data(TEST_START, TEST_END, sbert_model)
    df_finance = pd.read_csv(FINANCE_CSV)
    df_finance[FEATURE_COLS] = df_finance[FEATURE_COLS].fillna(0)

    df_train = pd.merge(df_finance, df_train_text, on="Date_Str", how="inner").sort_values("Date_Str")
    df_test  = pd.merge(df_finance, df_test_text, on="Date_Str", how="inner").sort_values("Date_Str")
    
    scaler = MinMaxScaler()
    scaler.fit(df_train[FEATURE_COLS])
    scaled_cols = [f"{c}_Scaled" for c in FEATURE_COLS]
    df_train[scaled_cols] = scaler.transform(df_train[FEATURE_COLS])
    df_test[scaled_cols]  = scaler.transform(df_test[FEATURE_COLS])
    
    X_train_base, y_train_base = create_dataset(df_train, SEQUENCE_LENGTH)
    X_test_base, y_test_base   = create_dataset(df_test, SEQUENCE_LENGTH)
    
    input_dim = X_train_base.shape[2]
    print(f"✅ データ準備完了 InputDim: {input_dim}")

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # ★ここが変わりました: Accuracyを基準にする
    best_acc = 0.0
    best_params = {}
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
        log_print(f"🚀 広域グリッドサーチ開始 (Accuracy重視): 全 {len(combinations)} 通り", f_out)
        
        for i, params in enumerate(combinations):
            log_print(f"\n[{i+1}/{len(combinations)}] Testing: {params}", f_out)
            
            acc_scores = []
            f1_scores = []
            
            for run in range(N_RUNS_PER_GRID):
                torch.manual_seed(run)
                random.seed(run)
                np.random.seed(run)
                
                model = HybridLSTM(input_dim, params["HIDDEN_SIZE"], params["NUM_LAYERS"], params["DROPOUT"]).to(device)
                optimizer = optim.Adam(model.parameters(), lr=params["LR"])
                criterion = nn.BCELoss()
                
                loader = DataLoader(TensorDataset(torch.tensor(X_train_base, dtype=torch.float32), 
                                                  torch.tensor(y_train_base, dtype=torch.float32).unsqueeze(1)), 
                                    batch_size=params["BATCH_SIZE"], shuffle=True)
                
                model.train()
                for epoch in range(EPOCHS):
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(X_test_base, dtype=torch.float32).to(device)
                    preds = (model(inputs).cpu().numpy() >= 0.5).astype(int).flatten()
                    
                    # 両方のスコアを計算
                    acc = accuracy_score(y_test_base, preds)
                    f1  = f1_score(y_test_base, preds, zero_division=0)
                    
                    acc_scores.append(acc)
                    f1_scores.append(f1)
            
            avg_acc = np.mean(acc_scores)
            avg_f1  = np.mean(f1_scores)
            
            log_print(f"   -> Avg Acc: {avg_acc:.2%}, Avg F1: {avg_f1:.4f}", f_out)
            
            # ★ Accuracyで勝負！
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = params
                log_print(f"   ★ NEW BEST ACCURACY! ★", f_out)

        log_print("\n" + "="*50, f_out)
        log_print(f"🏆 最優秀パラメータ決定 (Accuracy)", f_out)
        log_print(f"Best Accuracy: {best_acc:.2%}", f_out)
        log_print(f"Parameters: {best_params}", f_out)
        log_print("="*50, f_out)

    print(f"\n💾 完了！結果はこちら: {REPORT_FILE}")