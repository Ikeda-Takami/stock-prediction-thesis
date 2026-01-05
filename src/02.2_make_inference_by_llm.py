import pandas as pd
import os
import glob
import ollama
import time
import math
from datetime import datetime

# ==========================================
# ★ 設定エリア ★
# ==========================================
# 期間
START_DATE = "2017-09-01" 
END_DATE   = "2024-12-31"

# LLM設定
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11500"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
MODEL_NAME = "gpt-oss:120b"
CONTEXT_SIZE = 131072 
SAFE_CHAR_LIMIT = 100000 

# パス設定
BASE_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/"
INPUT_CORPUS_DIR = os.path.join(BASE_DIR, "corpus_15/") 
# 出力先（バージョン管理推奨）
OUTPUT_DIR = os.path.join(BASE_DIR, "summary_reasoning_v1/")
MARKET_DATA_PATH = os.path.join(BASE_DIR, "finance/market_data_all_labeled.csv")

# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------
# 1. 市況データの読み込み
# ------------------------------------------
print("📊 市況データをロード中...")
df_market = pd.read_csv(MARKET_DATA_PATH)
df_market['Date_Str'] = pd.to_datetime(df_market['Date_Str']).dt.date
df_market.set_index('Date_Str', inplace=True)
print(" -> ロード完了！")

# ------------------------------------------
# 2. 推論ループ (週末統合ロジック)
# ------------------------------------------
all_files = sorted(glob.glob(os.path.join(INPUT_CORPUS_DIR, "corpus_*.txt")))
target_files = []

for f in all_files:
    d_str = os.path.basename(f).replace("corpus_", "").replace(".txt", "")
    if START_DATE <= d_str <= END_DATE:
        target_files.append(f)

print(f"🔥 LLM推論モード起動！")
print(f"🧠 モデル: {MODEL_NAME} | 📅 対象: {len(target_files)}ファイル")
print("-" * 50)

# ★ 週末データを溜め込むための変数
weekend_buffer = ""
weekend_dates = []

for i, file_path in enumerate(target_files):
    filename = os.path.basename(file_path)
    date_str = filename.replace("corpus_", "").replace(".txt", "")
    
    # 日付オブジェクト変換
    current_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # ファイル読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        daily_text = f.read()

    # --- 判定ロジック ---
    # 市況データがあるか確認（平日か？）
    is_trading_day = current_date in df_market.index

    if not is_trading_day:
        # ■ 休日（土日祝）の場合
        # バッファに溜め込むだけで、推論はしない
        print(f"[{i+1}/{len(target_files)}] 💤 {date_str} (休日) -> バッファに追加")
        weekend_buffer += f"\n--- {date_str} のツイート ---\n{daily_text}"
        weekend_dates.append(date_str)
        continue  # 次のループへ（スキップ）

    else:
        # ■ 平日（市場が開いている日）の場合
        # 推論を実行する
        save_path = os.path.join(OUTPUT_DIR, f"summary_{date_str}.txt")
        
        # 生成済みならスキップ
        if os.path.exists(save_path):
            # バッファもクリアしておく（次のサイクルのため）
            weekend_buffer = "" 
            weekend_dates = []
            continue

        print(f"[{i+1}/{len(target_files)}] 🚀 {date_str} (平日) 推論開始 ", end="", flush=True)
        start_time = time.time()

        # テキストの結合（週末分 + 当日分）
        if weekend_buffer:
            print(f" [週末分({len(weekend_dates)}日)を統合]", end="")
            full_text = weekend_buffer + f"\n--- {date_str} (当日) のツイート ---\n{daily_text}"
            # 使い終わったのでクリア
            weekend_buffer = ""
            weekend_dates = []
        else:
            full_text = daily_text

        # 文字数制限（間引き）
        total_chars = len(full_text)
        if total_chars > SAFE_CHAR_LIMIT:
            step = math.ceil(total_chars / SAFE_CHAR_LIMIT)
            # 行ごとに分割して間引く
            lines = full_text.splitlines()
            text_data = "\n".join(lines[::step])
            status = f"(間引1/{step})"
        else:
            text_data = full_text
            status = "(全量)"

        # 市況データの取得
        try:
            row = df_market.loc[current_date]
            n_close = row['Nikkei_Close']
            n_change = row['Nikkei_Change'] if not pd.isna(row['Nikkei_Change']) else 0
            d_change = row['Dow_Change'] if not pd.isna(row['Dow_Change']) else 0
            u_close = row['USDJPY_Close']
            
            n_sign = "+" if n_close > 0 else ""
            n_chg_sign = "+" if n_change > 0 else ""
            d_chg_sign = "+" if d_change > 0 else ""

            market_text = (
                f"■日経平均株価: {n_close:,.0f}円 (前日比: {n_chg_sign}{n_change:,.0f}円)\n"
                f"■NYダウ(前日): 前日比 {d_chg_sign}{d_change:,.0f}ドル\n"
                f"■ドル円レート: {u_close:.2f}円"
            )
        except Exception as e:
            market_text = f"（データ取得エラー: {e}）"

        # --- プロンプト構築 (0-9スコア版) ---
        # --- プロンプト構築 (各STEP 100文字記述版) ---
        my_prompt = f"""
あなたは世界最高のヘッジファンドで運用される「Alpha-Generative AI」です。
あなたの目的は、**「機会損失（儲け損ない）」を「実際の損失」と同じくらい重大な罪**と捉え、上昇トレンドや反発の兆候を絶対に見逃さないことです。

以下の3つの高度な推論フレームワークを順に実行し、明日の日経平均株価を**「予測（上昇/下落）」と「0から9のスコア」**で評価してください。

**【重要ルール：逃げ場なし】**
- **偶数個（10段階）のスケール**を採用しているため、「中立」という選択肢はありません。
- どんなに迷っても、**「売り目線（0-4）」か「買い目線（5-9）」か、必ずどちらかのサイド**に立ってください。

【入力データ】
▼ 市況データ（事実・トレンド）
{market_text}

▼ SNS投資家の発言（感情・群集心理）
※週末（土日）のデータが含まれる場合は、週明けへの期待や不安として解釈してください。
{text_data}

---

【推論プロトコル】

### STEP 1: FinCoT (事実に基づく大枠の方向決定)
**役割: テクニカル・ストラテジスト**
- 市況データ（日経平均前日比、NYダウ、ドル円）のみを見て、トレンドが「強気」か「弱気」か判定します。
- **ルール**: 「トレンドは友」。前日比プラスやダウ堅調なら、基本は**「買い目線（5-9）」**のゾーンを選択します。小さな悪材料でトレンドに逆らってはいけません。

### STEP 2: DualGAT (群集心理によるスコア精緻化)
**役割: 行動ファイナンス分析官**
- ここでSNSを見ます。選んだゾーンの中でスコアを確定させます。
- **乖離の発見**: STEP 1で「買い目線」としたのにSNSが悲観的なら、それは「絶好の押し目（逆張り機）」です。自信を持ってスコアを **8** や **9** に引き上げてください。
- **ノイズ除去**: SNSの「暴落だ」という悲鳴はノイズ、あるいは「セリングクライマックス」の合図です。これに同調してスコアを下げる愚を犯さないでください。

### STEP 3: SEP (自己反省による境界線チェック)
**役割: リスク管理オーディター**
- **特に「4（迷い売り）」と「5（迷い買い）」の境界にいる場合**、自問自答してください。
  - 「私は単に慎重になりすぎて、4を選ぼうとしていないか？」
  - 「ダウが少し下がっただけで、上昇トレンド（5以上）を否定していないか？」
- 「機会損失は罪」であることを思い出し、迷ったら**「5（買い）」**側に倒す勇気を持ってください。

---

【出力フォーマット】
**以下の形式・制約を厳守してください。**
1. **アスタリスク（**）やシャープ（##）などのMarkdown装飾は絶対に使用しないこと。**
2. **【理由】は必ず「STEP1」「STEP2」「STEP3」の3行だけで構成すること。**
3. **重要：各STEPは「200文字程度の濃密な文章」で記述すること。改行はせず、具体的な数値（円、ドル、％）やSNSのキーワードを引用して論理を肉付けすること。**

[出力テンプレート]
【予測】 上昇 または 下落
【スコア】 0 〜 9 の整数
【理由】
STEP1: (トレンド分析の内容。前日比やダウの数値を引用し、なぜそのトレンド判断に至ったか200文字程度で詳しく記述)
STEP2: (SNS分析の内容。具体的なキーワードを挙げ、群集心理がノイズかシグナルかを200文字程度で深く分析)
STEP3: (自己反省と結論。迷いやリスク要因を挙げた上で、なぜ最終的にそのスコアを選んだのか200文字程度で論理的に完結)
        """

        # --- LLM実行 ---
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": my_prompt}],
                options={
                    "temperature": 0.1, 
                    "num_ctx": CONTEXT_SIZE,
                    "seed": 42
                }
            )

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response["message"]["content"])

            elapsed = time.time() - start_time
            print(f" ✅{status} {elapsed:.1f}秒")

        except Exception as e:
            print(f" ❌ エラー: {e}")

print("-" * 50)
print(f"✨ 推論完了！保存先: {OUTPUT_DIR}")