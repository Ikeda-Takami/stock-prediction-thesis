import yfinance as yf
import pandas as pd
import os

# ==========================================
# ★ 設定エリア ★
# ==========================================
# 保存先ディレクトリ
OUTPUT_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/finance/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存するファイル名（全てのデータが入るので名前を豪華にしました）
CSV_PATH = os.path.join(OUTPUT_DIR, "market_data_all_labeled.csv")

# 取得期間
START_DATE = "2017-01-01"
END_DATE = "2024-12-31"
# ==========================================

print(f"🚀 市況データ（日経・ダウ・ドル円）を一括取得中... ({START_DATE} ~ {END_DATE})")

# 1. 各データを個別に取得（マルチインデックス事故を防ぐため個別取得が安全）
tickers = {
    "^N225": "Nikkei",
    "^DJI": "Dow",
    "JPY=X": "USDJPY"
}

dfs = []
for ticker, name in tickers.items():
    print(f" - {name} ({ticker}) をダウンロード中...")
    try:
        # 必要なカラムだけ取得
        d = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        d = d[['Close', 'Volume']] if 'Volume' in d.columns else d[['Close']]
        
        # カラム名を変更 (例: Close -> Nikkei_Close)
        # yfinanceのバージョンによるMultiIndex対応
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)
            
        d = d.rename(columns={
            'Close': f'{name}_Close',
            'Volume': f'{name}_Volume'
        })
        
        # タイムゾーン情報があると結合時にエラーになるので削除
        d.index = d.index.tz_localize(None)
        dfs.append(d)
        
    except Exception as e:
        print(f"❌ {name} の取得エラー: {e}")

# 2. データの結合 (Merge)
print("🔄 データを結合しています...")
# 日経平均の日付を基準（Left Join）にします
df_merged = dfs[0].join(dfs[1:], how='left')

# 3. 欠損値の処理 (Forward Fill)
# 例: 日本が祝日でデータがない行はそもそも存在しません(Left Joinのため)
# 例: アメリカが祝日でDowがない日は、前日のDowをコピーして埋めます
df_merged = df_merged.ffill()

# まだNaNがある（開始日直後など）場合は削除
df_merged = df_merged.dropna()

# 4. 特徴量の作成（LLM推論 & LSTM学習用）
# 前日比（Change）を計算
df_merged["Nikkei_Change"] = df_merged["Nikkei_Close"].diff()
df_merged["Dow_Change"] = df_merged["Dow_Close"].diff()
df_merged["USDJPY_Change"] = df_merged["USDJPY_Close"].diff()

# 5. 正解ラベルの作成
# 翌日(T+1)の日経平均終値
df_merged["Next_Close"] = df_merged["Nikkei_Close"].shift(-1)

# 翌日のデータがない行は削除
df_merged = df_merged.dropna(subset=["Next_Close"])

# ラベル: 上昇=1, 下落=0
df_merged["Actual_Label"] = (df_merged["Next_Close"] > df_merged["Nikkei_Close"]).astype(int)
df_merged["Actual_Diff"] = df_merged["Next_Close"] - df_merged["Nikkei_Close"]

# 日付カラム作成
df_merged["Date_Str"] = df_merged.index.strftime("%Y-%m-%d")

# 並び替え（見やすいように）
cols = [
    "Date_Str", 
    "Nikkei_Close", "Nikkei_Change", "Nikkei_Volume",
    "Dow_Close", "Dow_Change",
    "USDJPY_Close", "USDJPY_Change",
    "Next_Close", "Actual_Diff", "Actual_Label"
]
# 存在しないカラム（Dow_Volumeなど）を除外して選択
final_cols = [c for c in cols if c in df_merged.columns]
result_df = df_merged[final_cols]

# CSV出力
result_df.to_csv(CSV_PATH, index=False)

print(f"✅ 完了！最強の市況データを作成しました: {CSV_PATH}")
print("-" * 30)
print(result_df.tail()) # 最新のデータを確認
print(f"📊 総データ数: {len(result_df)} 日分")