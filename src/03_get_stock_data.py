import yfinance as yf
import pandas as pd
import os

# ==========================================
# ★ 設定エリア ★
# ==========================================
# 保存先ディレクトリ
OUTPUT_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/finance/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存するファイル名
CSV_PATH = os.path.join(OUTPUT_DIR, "nikkei_225_labeled.csv")

# 取得期間 (データがある期間より少し広めに設定)
START_DATE = "2017-01-01"
END_DATE = "2024-12-31"
# ==========================================

print(f"🚀 日経平均株価 (^N225) を取得中... ({START_DATE} ~ {END_DATE})")

# 1. データ取得 (Yahoo Finance API)
ticker = "^N225"
try:
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
except Exception as e:
    print(f"❌ データ取得エラー: {e}")
    exit()

if df.empty:
    print("❌ データが取得できませんでした。インターネット接続を確認してください。")
    exit()

# 2. データの整形
# yfinanceのバージョンによってはカラムが多層(MultiIndex)になるので修正
if isinstance(df.columns, pd.MultiIndex):
    try:
        # Tickerレベルを削除してシンプルなカラム名にする
        df.columns = df.columns.droplevel(1)
    except:
        pass

# 不要な行を削除
df = df.dropna()

# 3. 正解ラベルの作成
# ロジック: 今日のデータ(T)に対して、翌日(T+1)の終値が上がったかどうかを知りたい
# shift(-1) を使うと「1行下のデータ（未来）」を今の行に持ってこれる

df["Next_Close"] = df["Close"].shift(-1) # 翌日の終値

# 翌日のデータがない日（最新の日付など）は正解が作れないので削除
df = df.dropna(subset=["Next_Close"])

# 正解ラベル: (翌日終値 > 当日終値) なら 1 (上昇), それ以外は 0 (下落)
df["Actual_Label"] = (df["Next_Close"] > df["Close"]).astype(int)
df["Actual_Diff"] = df["Next_Close"] - df["Close"]  # 具体的にいくら動いたか

# 日付フォーマットを文字列にしておく（後で結合しやすくするため）
df["Date_Str"] = df.index.strftime("%Y-%m-%d")

# ★ここが変更点！ "Volume" を追加しました ★
# 必要なカラムだけ選んで保存
result_df = df[["Date_Str", "Close", "Volume", "Next_Close", "Actual_Diff", "Actual_Label"]]

# CSV出力
result_df.to_csv(CSV_PATH, index=False)

print(f"✅ 正解データ（出来高付き）を作成しました: {CSV_PATH}")
print("-" * 30)
print("--- 作成されたデータ（最初の5行） ---")
print(result_df.head())
print("-" * 30)
print(f"📊 総データ数: {len(result_df)} 日分")