import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==========================================
# ★ LSSI 最終評価 (Loss, Prec, Recall追加版) ★
# ==========================================
EXPERIMENT_NAME = "LSSI_Final_Full_Metrics"
N_RUNS = 10 

# 期間設定 (augasuto_v3)
TRAIN_START = "2024-06-01"
TRAIN_END   = "2024-08-31"
TEST_START  = "2024-09-01"
TEST_END    = "2024-10-30"

INPUT_SUMMARY_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/summary_reasoning_augasuto_v3/"
FINANCE_CSV       = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/finance/nikkei_225_labeled.csv"
REPORT_FILE       = f"/home/sakulab/workspace/B4_ikeda/graduation_thesis/result/LSSI_Final_Full_{TEST_START}-{TEST_END}.txt"

# ★★★ グリッドサーチの勝者設定 (暫定値: 実行前に書き換えてください！) ★★★
# さっきのグリッドサーチ結果を見て、一番良いやつを入れてください
BEST_PARAMS = {
    "SEQUENCE_LENGTH": 45,     # ← GridSearchのBestを入れてね
    "HIDDEN_SIZE":     64,    # ← GridSearchのBestを入れてね
    "NUM_LAYERS":      1,
    "LEARNING_RATE":   0.005,
    "DROPOUT":         0.2,
    "BATCH_SIZE":      16
}

EPOCHS = 100
PATIENCE = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

def log_print(text, file_obj=None):
    print(text)
    if file_obj: file_obj.write(text + "\n")

def extract_scalar_signal(text):
    match = re.search(r"【スコア】\s*(\d+)", text)
    if match: return int(match.group(1))
    return None

def load_lssi_data_all():
    all_files = sorted(glob.glob(os.path.join(INPUT_SUMMARY_DIR, "summary_*.txt")))
    dates, scores = [], []
    print(f"🔍 全期間データのLSSI信号抽出中...")
    for file_path in tqdm(all_files, desc="Extracting Scalars"):
        filename = os.path.basename(file_path)
        d_str = filename.replace("summary_", "").replace(".txt", "")
        with open(file_path, "r", encoding="utf-8") as f: text = f.read().strip()
        if not text: continue
        score = extract_scalar_signal(text)
        if score is not None:
            dates.append(d_str)
            scores.append(score)
    return pd.DataFrame({"Date_Str": dates, "Scalar_Signal": scores})

def create_sequences_and_split(df, seq_len):
    features = df[["Scalar_Signal_Scaled", "Close_Scaled", "Volume_Scaled"]].values
    labels = df["Actual_Label"].values
    dates = df["Date_Str"].values
    
    X_all, y_all, date_all = [], [], []
    
    for i in range(len(df) - seq_len):
        X_all.append(features[i : i + seq_len])
        y_all.append(labels[i + seq_len])
        date_all.append(dates[i + seq_len])
        
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    date_all = np.array(date_all)
    
    train_mask = (date_all <= TRAIN_END)
    test_mask  = (date_all >= TEST_START)
    
    return (X_all[train_mask], y_all[train_mask]), (X_all[test_mask], y_all[test_mask])

class LSSILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSSILSTM, self).__init__()
        actual_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            dropout=actual_dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

if __name__ == "__main__":
    with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
        log_print(f"🚀 LSSI 最終評価 (Full Metrics)", f_out)
        
        # 1. データ準備
        df_signal = load_lssi_data_all()
        df_finance = pd.read_csv(FINANCE_CSV)
        df = pd.merge(df_finance, df_signal, on="Date_Str", how="inner").sort_values("Date_Str")
        
        scaler = MinMaxScaler()
        cols = ["Close", "Volume", "Scalar_Signal"]
        train_fit_mask = (df["Date_Str"] <= TRAIN_END)
        scaler.fit(df.loc[train_fit_mask, cols])
        df[["Close_Scaled", "Volume_Scaled", "Scalar_Signal_Scaled"]] = scaler.transform(df[cols])
        
        (X_train, y_train), (X_test, y_test) = create_sequences_and_split(df, BEST_PARAMS["SEQUENCE_LENGTH"])
        input_dim = X_train.shape[2]
        
        log_print(f"✅ データセット: Train={len(X_train)}, Test={len(X_test)}", f_out)

        history = {
            "Accuracy": [], "F1": [], "Precision": [], "Recall": [], "Best_Loss": []
        }

        # 2. 10回勝負
        for run in range(1, N_RUNS + 1):
            log_print(f"🔄 Run {run}/{N_RUNS} ...", f_out)
            torch.manual_seed(run)
            np.random.seed(run)
            
            model = LSSILSTM(input_dim, BEST_PARAMS["HIDDEN_SIZE"], BEST_PARAMS["NUM_LAYERS"], BEST_PARAMS["DROPOUT"]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS["LEARNING_RATE"])
            
            loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)), 
                                batch_size=BEST_PARAMS["BATCH_SIZE"], shuffle=True)
            
            best_loss = float('inf')
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            
            model.train()
            for epoch in range(EPOCHS):
                total_loss = 0
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= PATIENCE:
                    break
            
            # ベストモデルで評価
            model.load_state_dict(best_model_wts)
            model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
                preds_prob = model(inputs).cpu().numpy()
            
            y_pred = (preds_prob >= 0.5).astype(int).flatten()
            
            # 指標計算
            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            
            history["Accuracy"].append(acc)
            history["F1"].append(f1)
            history["Precision"].append(prec)
            history["Recall"].append(rec)
            history["Best_Loss"].append(best_loss)
            
            log_print(f"   👉 Acc={acc:.2%}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f} (Loss={best_loss:.4f})", f_out)

        # 3. 最終集計
        log_print("\n" + "="*50, f_out)
        log_print(f"🏆 最終平均結果 (LSSI)", f_out)
        log_print(f"✅ Accuracy : {np.mean(history['Accuracy']):.2%} (Max: {np.max(history['Accuracy']):.2%})", f_out)
        log_print(f"⭐ F1 Score : {np.mean(history['F1']):.4f}", f_out)
        log_print(f"🎯 Precision: {np.mean(history['Precision']):.4f}", f_out)
        log_print(f"📢 Recall   : {np.mean(history['Recall']):.4f}", f_out)
        log_print(f"📉 Best Loss: {np.mean(history['Best_Loss']):.4f}", f_out)
        log_print("="*50, f_out)

    print(f"\n💾 レポート: {REPORT_FILE}")