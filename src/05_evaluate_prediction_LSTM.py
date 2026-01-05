import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

# ==========================================
# ★ 設定エリア ★
# ==========================================
N_RUNS = 10  # 試行回数

# 期間設定
TRAIN_START = "2017-09-01"
TRAIN_END   = "2024-06-30"  # 修正: 6月は30日まで
TEST_START  = "2024-07-01"
TEST_END    = "2024-12-30"

# パス設定
BASE_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/"
INPUT_SUMMARY_DIR = os.path.join(BASE_DIR, "data/summary_reasoning_augasuto_v3/")
# ★重要: ダウや為替が入っているCSVを指定してください
FINANCE_CSV       = os.path.join(BASE_DIR, "data/finance/market_data_all_labeled.csv") 
REPORT_FILE       = os.path.join(BASE_DIR, f"result/LSTM_Step3_FUSION_{TEST_START}-{TEST_END}.txt")

# モデル設定
SBERT_MODEL_NAME = "pkshatech/GLuCoSE-base-ja"
NORMALIZE_VEC    = True

# ★LSTMに入力する市況データのカラム (Volume除外版)
FEATURE_COLS = [
    "Nikkei_Close",   # 日経終値
    "Nikkei_Change",  # 日経前日比 (重要)
    "Dow_Change",     # ダウ前日比
    "USDJPY_Close"    # ドル円
]

# LSTMハイパーパラメータ
SEQUENCE_LENGTH = 10
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.6
EPOCHS          = 70
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001  # ★修正: 0.00だと学習しないので0.001にしました
# ==========================================

os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

def log_print(text, file_obj=None):
    print(text)
    if file_obj:
        file_obj.write(text + "\n")

# --- STEP3抽出関数 (マークダウン対応・最強版) ---
def extract_step3(text):
    """
    テキストからSTEP3の部分だけを抽出する。
    **STEP 3** : や **STEP3:** などの表記揺れにも対応。
    """
    # STEPの後にスペースがあっても数字の3、その後に改行以外の文字(マークダウン記号など)があってもOKとする
    match = re.search(r"STEP\s*3[^\n]*?[:：](.*)", text, re.IGNORECASE | re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        # 抽出後に残ってしまったマークダウン記号（**や##）を消す掃除処理
        content = content.replace("**", "").replace("##", "").strip()
        return content
    
    # 見つからない場合は全文を返す（エラー回避）
    return text.strip()

# --- データ準備関数 ---
def load_data(start_date, end_date, sbert_model):
    all_files = sorted(glob.glob(os.path.join(INPUT_SUMMARY_DIR, "summary_*.txt")))
    target_files = []
    for p in all_files:
        d_str = os.path.basename(p).replace("summary_", "").replace(".txt", "")
        if start_date <= d_str <= end_date:
            target_files.append(p)
            
    if not target_files: return pd.DataFrame()

    dates = []
    vectors = []
    print(f"   データ読み込み & STEP3ベクトル化中 ({start_date} 〜 {end_date})...")
    
    for file_path in tqdm(target_files, desc="Vectorizing Step3"):
        filename = os.path.basename(file_path)
        date_str = filename.replace("summary_", "").replace(".txt", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        
        if not full_text: continue
        
        # 強化版抽出関数を使用
        step3_text = extract_step3(full_text)
        vec = sbert_model.encode(step3_text, normalize_embeddings=NORMALIZE_VEC)
        
        dates.append(date_str)
        vectors.append(vec)
        
    return pd.DataFrame({"Date_Str": dates, "Vector": vectors})

def create_dataset(df, seq_len):
    X_list, y_list = [], []
    
    # 1. テキストベクトル (768次元)
    vector_features = np.stack(df["Vector"].values)
    
    # 2. 市況データ (4次元: Nikkei, Change, Dow, USD)
    # カラム名に _Scaled をつけたものを使用
    scaled_cols = [f"{c}_Scaled" for c in FEATURE_COLS]
    numeric_features = df[scaled_cols].values
    
    # 3. 結合 (772次元)
    combined_features = np.hstack([vector_features, numeric_features])
    
    labels = df["Actual_Label"].values
    
    for i in range(len(df) - seq_len):
        X_list.append(combined_features[i : i + seq_len])
        y_list.append(labels[i + seq_len])
        
    return np.array(X_list), np.array(y_list)

# --- LSTMモデル定義 ---
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
    with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
        log_print(f"🚀 LSTM (STEP3 + 4大指標統合) 連続実験 ({N_RUNS}回平均)", f_out)
        log_print(f"📊 使用指標: {FEATURE_COLS}", f_out)
        log_print(f"🏋️ 訓練: {TRAIN_START} 〜 {TRAIN_END}", f_out)
        log_print(f"🧪 テスト: {TEST_START} 〜 {TEST_END}", f_out)
        
        # 1. データ準備
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_print(f"📥 SBERTロード: {device}", f_out)
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        
        # ベクトル作成
        df_train_text = load_data(TRAIN_START, TRAIN_END, sbert_model)
        df_test_text  = load_data(TEST_START, TEST_END, sbert_model)
        
        # 市況データ読み込み
        df_finance = pd.read_csv(FINANCE_CSV)
        
        # ★欠損値埋め
        df_finance[FEATURE_COLS] = df_finance[FEATURE_COLS].fillna(0)

        # マージ
        df_train = pd.merge(df_finance, df_train_text, on="Date_Str", how="inner").sort_values("Date_Str")
        df_test  = pd.merge(df_finance, df_test_text, on="Date_Str", how="inner").sort_values("Date_Str")
        
        # スケーリング (0-1に正規化)
        scaler = MinMaxScaler()
        scaler.fit(df_train[FEATURE_COLS]) # Trainでfit
        
        # 変換して _Scaled カラムを作成
        train_scaled = scaler.transform(df_train[FEATURE_COLS])
        test_scaled  = scaler.transform(df_test[FEATURE_COLS])
        
        scaled_col_names = [f"{c}_Scaled" for c in FEATURE_COLS]
        df_train[scaled_col_names] = train_scaled
        df_test[scaled_col_names]  = test_scaled
        
        # データセット作成
        X_train, y_train = create_dataset(df_train, SEQUENCE_LENGTH)
        X_test, y_test   = create_dataset(df_test, SEQUENCE_LENGTH)
        
        log_print(f"✅ データ準備完了: Input Dim={X_train.shape[2]} (768 + {len(FEATURE_COLS)})", f_out)

        # 2. ループ実行
        history = {
            "Accuracy": [], "Precision": [], "Recall": [], "F1": []
        }

        for run in range(1, N_RUNS + 1):
            log_print(f"🔄 Run {run}/{N_RUNS} 開始...", f_out)
            
            torch.manual_seed(run)
            random.seed(run)
            np.random.seed(run)
            
            model = HybridLSTM(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)), 
                                batch_size=BATCH_SIZE, shuffle=True)
            
            model.train()
            for epoch in range(EPOCHS):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
            
            # 評価
            model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
                preds_prob = model(inputs).cpu().numpy()
            
            y_pred = (preds_prob >= 0.5).astype(int).flatten()
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            
            history["Accuracy"].append(acc)
            history["Precision"].append(prec)
            history["Recall"].append(rec)
            history["F1"].append(f1)
            
            log_print(f"   👉 Run {run} Result: Acc={acc:.2%}, F1={f1:.4f}", f_out)

        # 3. 最終結果の集計
        log_print("\n" + "="*50, f_out)
        log_print(f"🏆 {N_RUNS}回試行の平均結果 (Fusion Model)", f_out)
        log_print("="*50, f_out)
        
        for metric, values in history.items():
            avg = np.mean(values)
            std = np.std(values)
            log_print(f"✅ {metric:<10}: 平均 {avg:.2%} (±{std:.2%})", f_out)
            
        log_print("="*50, f_out)

    print(f"\n💾 全実験完了！レポート: {REPORT_FILE}")