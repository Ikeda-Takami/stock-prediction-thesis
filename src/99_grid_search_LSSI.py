import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
import time
import copy  # ★追加
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==========================================
# ★ LSSI グリッドサーチ (完全統一版) ★
# ==========================================
SEARCH_GRID = {
    "SEQUENCE_LENGTH": [45,90], 
    "HIDDEN_SIZE":     [32,64,18],
    "NUM_LAYERS":      [1, 2,3],
    "LEARNING_RATE":   [0.01,0.005],
    "DROPOUT":         [0.2,0.3],
    "BATCH_SIZE":      [16, 32,8]
}

EPOCHS = 100
PATIENCE = 15
TRAIN_START = "2024-06-01"
TRAIN_END   = "2024-08-31"
TEST_START  = "2024-09-01"
TEST_END    = "2024-10-31"

INPUT_SUMMARY_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/summary_reasoning_augasuto_v3/"
FINANCE_CSV       = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/finance/nikkei_225_labeled.csv"
RESULT_FILE       = f"/home/sakulab/workspace/B4_ikeda/graduation_thesis/result/GridSearch_LSSI_EXACT_{TEST_START}-{TEST_END}.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_scalar_signal(text):
    match = re.search(r"【スコア】\s*(\d+)", text)
    if match: return int(match.group(1))
    return None

def load_lssi_data_all():
    all_files = sorted(glob.glob(os.path.join(INPUT_SUMMARY_DIR, "summary_*.txt")))
    dates, scores = [], []
    for file_path in all_files:
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
    
    train_mask = (date_all >= TRAIN_START) & (date_all <= TRAIN_END)
    test_mask  = (date_all >= TEST_START) & (date_all <= TEST_END)
    
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

def main():
    print(f"🚀 LSSI グリッドサーチ (完全統一版) 開始")
    
    # データ準備
    df_signal = load_lssi_data_all()
    df_finance = pd.read_csv(FINANCE_CSV)
    df = pd.merge(df_finance, df_signal, on="Date_Str", how="inner").sort_values("Date_Str")
    
    scaler = MinMaxScaler()
    cols = ["Close", "Volume", "Scalar_Signal"]
    train_mask_fit = (df["Date_Str"] >= TRAIN_START) & (df["Date_Str"] <= TRAIN_END)
    scaler.fit(df.loc[train_mask_fit, cols])
    df[["Close_Scaled", "Volume_Scaled", "Scalar_Signal_Scaled"]] = scaler.transform(df[cols])
    
    print(f"✅ データ準備完了: Total {len(df)} rows")

    keys, values = zip(*SEARCH_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    all_results = []
    start_time_all = time.time()
    
    for i, params in enumerate(param_combinations):
        seq_len = params["SEQUENCE_LENGTH"]
        (X_train, y_train), (X_test, y_test) = create_sequences_and_split(df, seq_len)
        
        if len(X_train) == 0: continue
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                                torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
                                  batch_size=params["BATCH_SIZE"], shuffle=True)
        
        model = LSSILSTM(3, params["HIDDEN_SIZE"], params["NUM_LAYERS"], params["DROPOUT"]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])
        
        # ★ Early Stopping ロジック (評価コードと統一)
        best_loss = float('inf')
        patience_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(model.state_dict()) # ★ベストを保存
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                break
        
        # ★ベストモデルをロードして評価 (ここが重要！)
        model.load_state_dict(best_model_wts)
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds_prob = model(inputs).cpu().numpy()
            y_true = y_test
        y_pred = (preds_prob >= 0.5).astype(int).flatten()
        
        res = params.copy()
        res.update({
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1_Score": f1_score(y_true, y_pred, zero_division=0),
            "Test_Size": len(y_true)
        })
        all_results.append(res)
        
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time_all
            print(f"[{i+1}/{len(param_combinations)}] Acc: {res['Accuracy']:.2%} (TestSize={len(y_true)})")

    if all_results:
        df_res = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
        df_res.to_csv(RESULT_FILE, index=False)
        print(f"\n🏆 Best: {df_res.iloc[0]['Accuracy']:.2%}")
        print(df_res.iloc[0])

if __name__ == "__main__":
    main()