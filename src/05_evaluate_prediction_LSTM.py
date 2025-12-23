import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==========================================
# ★ 設定エリア ★
# ==========================================
START_DATE = "2024-08-01"
END_DATE   = "2024-08-31"
# 本番例: "2017-01-01" 〜 "2024-12-31"

INPUT_SUMMARY_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/summary_test_august/"
FINANCE_CSV = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/finance/nikkei_225_labeled.csv"
REPORT_FILE = f"/home/sakulab/workspace/B4_ikeda/graduation_thesis/result/LSTM_{START_DATE}_{END_DATE}/report.txt"

# モデル設定
SBERT_MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
MAX_SEQ_LENGTH = 512  # 文章の読み込み最大長
NORMALIZE_VEC  = True # ベクトルの正規化

# LSTMハイパーパラメータ
SEQUENCE_LENGTH = 3   # 過去何日分を見るか
HIDDEN_SIZE = 64      # AIの思考力
NUM_LAYERS  = 2       # ★2階建てにして複雑なパターンを読めるようにする
DROPOUT     = 0.2     # ★過学習防止（20%の情報をランダムにカット）
EPOCHS      = 50
BATCH_SIZE  = 8
LEARNING_RATE = 0.001
# ==========================================

os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

def log_print(text, file_obj=None):
    print(text)
    if file_obj:
        file_obj.write(text + "\n")

with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
    
    log_print(f"🚀 ハイブリッドLSTM予測（テキスト＋株価＋出来高）", f_out)
    log_print(f"📅 対象期間: {START_DATE} 〜 {END_DATE}", f_out)
    
    log_print("-" * 50, f_out)
    log_print("🛠️ 実験設定 (Hyperparameters)", f_out)
    log_print(f"🔹 SBERT Model      : {SBERT_MODEL_NAME}", f_out)
    log_print(f"🔹 Max Seq Length   : {MAX_SEQ_LENGTH}", f_out)
    log_print(f"🔹 Features         : Text(768) + Close(1) + Volume(1)", f_out)
    log_print(f"🔹 Sequence Length  : {SEQUENCE_LENGTH} days", f_out)
    log_print(f"🔹 LSTM Structure   : Layers={NUM_LAYERS}, Hidden={HIDDEN_SIZE}, Dropout={DROPOUT}", f_out)
    log_print(f"🔹 Training         : Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}", f_out)
    log_print("-" * 50, f_out)

    # ==========================================
    # Phase 1: ベクトル化 (SBERT)
    # ==========================================
    log_print("\n[Phase 1] テキストデータのベクトル化...", f_out)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_print(f"📥 SBERTロード: {SBERT_MODEL_NAME} ({device})", f_out)
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    sbert_model.max_seq_length = MAX_SEQ_LENGTH

    # ベクトルの次元数を自動取得
    text_dim = sbert_model.get_sentence_embedding_dimension()
    log_print(f"📏 テキスト次元数: {text_dim}", f_out)

    all_files = sorted(glob.glob(os.path.join(INPUT_SUMMARY_DIR, "summary_*.txt")))
    target_files = []
    for p in all_files:
        d_str = os.path.basename(p).replace("summary_", "").replace(".txt", "")
        if START_DATE <= d_str <= END_DATE:
            target_files.append(p)

    if len(target_files) == 0:
        log_print("❌ ファイルが見つかりません。", f_out)
        exit()

    dates = []
    vectors = []
    
    print("   ベクトル化処理中...")
    for file_path in tqdm(target_files):
        filename = os.path.basename(file_path)
        date_str = filename.replace("summary_", "").replace(".txt", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if not text: continue

        vec = sbert_model.encode(text, normalize_embeddings=NORMALIZE_VEC)
        dates.append(date_str)
        vectors.append(vec)

    df_vectors = pd.DataFrame({"Date_Str": dates, "Vector": vectors})
    log_print(f"✅ ベクトル化完了: {len(df_vectors)} 日分", f_out)

    # ==========================================
    # Phase 2: データ結合 & 数値データの正規化
    # ==========================================
    log_print("\n[Phase 2] データセット作成（数値データの正規化を含む）...", f_out)

    if not os.path.exists(FINANCE_CSV):
        log_print("❌ 株価CSVが見つかりません。", f_out)
        exit()

    df_finance = pd.read_csv(FINANCE_CSV)
    
    # 結合
    df = pd.merge(df_finance, df_vectors, on="Date_Str", how="inner")
    df = df.sort_values("Date_Str").reset_index(drop=True)
    
    # ★重要: 数値データ（終値・出来高）を 0〜1 の範囲に縮める（正規化）
    # これをやらないと、数値が大きすぎてAIがパニックになります
    # CSVのカラム名が 'Close', 'Volume' であることを想定しています
    scaler = MinMaxScaler()
    numeric_features = df[["Close", "Volume"]].values
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    log_print(f"📊 数値データ正規化完了 (Close, Volume)", f_out)

    # テキストベクトルと数値データを合体させる！
    # [ベクトル(768)] + [終値(1)] + [出来高(1)] = [合計(770)]
    vector_features = np.stack(df["Vector"].values)
    
    # 横に結合 (Horizontal Stack)
    combined_features = np.hstack([vector_features, numeric_features_scaled])
    
    # 入力次元数を更新 (768 + 2 = 770 になるはず)
    INPUT_SIZE = combined_features.shape[1]
    log_print(f"📏 AIへの入力次元数: {INPUT_SIZE} (テキスト{text_dim} + 数値2)", f_out)

    label_array = df["Actual_Label"].values

    # データセット作成
    X_list = []
    y_list = []

    if len(df) <= SEQUENCE_LENGTH:
        log_print("⚠️ データ不足です。", f_out)
        exit()

    for i in range(len(df) - SEQUENCE_LENGTH):
        seq_x = combined_features[i : i + SEQUENCE_LENGTH]
        target_y = label_array[i + SEQUENCE_LENGTH]
        X_list.append(seq_x)
        y_list.append(target_y)

    X_data = np.array(X_list)
    y_data = np.array(y_list)
    log_print(f"📊 データセット形状: X={X_data.shape}, y={y_data.shape}", f_out)

    # ==========================================
    # Phase 3: LSTM学習 (2階建て & Dropout)
    # ==========================================
    log_print("\n[Phase 3] 学習開始...", f_out)

    train_size = int(len(X_data) * 0.8)
    if train_size == 0: train_size = 1

    X_train = torch.tensor(X_data[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y_data[:train_size], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_data[train_size:], dtype=torch.float32)
    y_test = torch.tensor(y_data[train_size:], dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=min(BATCH_SIZE, len(X_train)), shuffle=True)

    # モデル定義 (引数を増やして強化)
    class HybridLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super(HybridLSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers=num_layers, # 層の数
                dropout=dropout,       # ドロップアウト率
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            last_out = out[:, -1, :]
            return self.sigmoid(self.fc(last_out))

    # モデル作成
    model = HybridLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

    log_print("✅ 学習完了", f_out)

    # ==========================================
    # Phase 4: 評価
    # ==========================================
    log_print("\n[Phase 4] 評価...", f_out)

    if len(X_test) == 0:
        log_print("⚠️ テストデータなし", f_out)
    else:
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            preds_prob = model(X_test).cpu().numpy()
            y_true = y_test.numpy().flatten()
        
        y_pred = (preds_prob >= 0.5).astype(int).flatten()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        log_print("=" * 40, f_out)
        log_print(f"📊 最終評価結果", f_out)
        log_print("=" * 40, f_out)
        log_print(f"✅ Accuracy : {acc:.2%}", f_out)
        log_print(f"⭐ F1 Score : {f1:.4f}", f_out)
        log_print("=" * 40, f_out)
        
        log_print("\n--- 詳細レポート ---", f_out)
        log_print(classification_report(y_true, y_pred, target_names=["下落", "上昇"], zero_division=0), f_out)

        log_print("\n--- 予測サンプル ---", f_out)
        for i in range(len(y_true)):
            t = "上昇" if y_true[i] == 1 else "下落"
            p = "上昇" if y_pred[i] == 1 else "下落"
            res = "⭕" if t == p else "❌"
            log_print(f"データ{i}: 正解[{t}] -> 予測[{p}] {res}", f_out)

print(f"\n💾 レポート保存完了: {REPORT_FILE}")