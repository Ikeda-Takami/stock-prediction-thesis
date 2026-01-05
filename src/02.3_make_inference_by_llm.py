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
OUTPUT_DIR = os.path.join(BASE_DIR, "summary_reasoning_augasuto_v3/")
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
        # --- プロンプト構築 (3段階ゾーン制・改良版) ---
        # --- プロンプト構築 (決定論的推論・高密度ベクトル化対応版) ---
        my_prompt = f"""
SYSTEM ROLE definition:
あなたは「SENTINEL-X」です。感情を持たない高頻度取引アルゴリズムのデータ処理ユニットです。
あなたの目的は、入力データを下流の予測モデル（LSTM）のために「数値ベクトル化可能な高密度シグナル」に変換することです。
金融アドバイスのような「ヘッジ（曖昧な表現）」は禁止されています。あなたは「計算機」として、0か1かの明確な判断を下してください。

【入力データ】
▼ 市況データ（事実・トレンド）
{market_text}

▼ SNS投資家の発言（感情・群集心理）
※【重要】週末（土日）データが含まれる場合の解釈プロトコル：
1. **「エネルギーの充填」として扱う**: 土日の投稿量は、月曜日の「窓開け（Gap Open）」の方向と強さを示唆します。
2. **「織り込み（Discounting）」の判定**: 土日の悪材料が月曜の寄り付きで織り込まれ、日中は「材料出尽くし」で反発する逆説的な動きを考慮してください。
3. **「夜明けの補正」**: 直近（月曜朝）の投稿ほど重みを置いて解釈してください。
{text_data}

---

【推論プロトコル】

### STEP 1: FinCoT (市況データと高度な金融知識によるゾーン判定)
**役割: テクニカル・ストラテジスト**
- 入力された市況データを、あなたの持つ**「膨大かつ多角的な金融・経済知識」**と照合し、トレンドを以下の3つのゾーンのいずれかに分類します。
  1. **【下降ゾーン (0-2)】**: ファンダメンタルズ悪化やテクニカル崩壊が明確。
  2. **【保合いゾーン (3-6)】**: 材料拮抗、または方向感が欠如している。
  3. **【上昇ゾーン (7-9)】**: 明確なトレンド、またはマクロ経済的な追い風がある。
- **制約**: 「様子見」は許されません。微細なシグナルからでも、必ずどれかのゾーンに割り振ってください。

### STEP 2: DualGAT (シグナル分離と乖離判定)
**役割: 行動ファイナンス分析官**
- SNSデータを以下の2つに厳密に分離・評価し、最終スコア（0〜9）を確定させます。
  A. **Hard Signals (検証可能な事実)**: 決算、提携、数値データ。
  B. **Soft Signals (主観的ノイズ)**: 恐怖、希望、皮肉、根拠のない煽り。
- **乖離（Divergence）チェック**:
  - STEP 1のトレンドや「Hard Signals」に対し、「Soft Signals」が逆行している場合（例：トレンドは上なのにSNSがパニック）、それは**「ノイズ」**あるいは**「逆張り機会」**です。
  - **判定ロジック**: 市場が過熱している状態でSNSが極度の楽観にある場合は「反転下落（売り）」、総悲観だが事実に変化がない場合は「押し目買い（買い）」と判定し、スコアを修正してください。

### STEP 3: Chain of Density (高密度ベクトル生成)
**役割: Embedding Encoder (前処理モジュール)**
- あなたの出力は、直接ベクトル化（Embedding）されてLSTMに入力されます。人間が読むための文章ではなく、**「機械が読むための高密度データ」**を作成してください。
- **生成プロセス**:
  1. ここまでの分析を統合する。
  2. 冗長な接続詞、挨拶、ヘッジ表現（"～と思われる", "～の可能性がある"）を**全て削除**する。
  3. 断定的な表現（"示唆する", "確度が高い"）のみを使用し、具体的な数値や専門用語を詰め込む「電文スタイル（Telegraphic Style）」で記述する。
- **必須要素**: トレンド方向、センチメントの強度、乖離の有無、そして「なぜそのスコアなのか」の決定的な根拠。

---

### STEP 3: Chain of Density (高密度ベクトル生成)
**役割: Embedding Encoder (前処理モジュール)**
- あなたの出力は、直接ベクトル化（Embedding）されてLSTMに入力されます。人間が読むための文章ではなく、**「機械が読むための高密度データ」**を作成してください。
- **生成プロセス**:
  1. ここまでの分析を統合し、結論を導出する。
  2. 冗長な接続詞、挨拶、ヘッジ表現（"～と思われる", "～の可能性がある"）を**全て削除**する。
  3. 断定的な表現（"示唆する", "確度が高い"）のみを使用し、具体的な数値や専門用語を詰め込む「電文スタイル（Telegraphic Style）」で記述する。
  4. **【最重要】スコア決定ロジックの埋め込み**: 最終的に算出したスコア（0-9）について、**「なぜその数字なのか（例：なぜ8で止まり、9ではないのか）」**という決定的な理由を結論として記述する。
- **必須要素**: トレンド方向、センチメント強度、乖離の有無、および**スコアを特定した最終的な決定打（Conclusion Basis）**。

[出力テンプレート]
【予測】 上昇 または 下落
【スコア】 0 〜 9 の整数
【理由】
STEP1: (トレンド分析。市況データと金融知識を統合し、なぜそのゾーンを選んだか。断定的な口調で200文字程度記述)
STEP2: (SNS分析。Hard/Softシグナルの分離と乖離判定を行い、スコア決定のロジックを記述。「～のためスコアXとする」と明記。200文字程度)
STEP3: (ベクトル化用要約。ヘッジ表現を排除し、リスク要因と勝機を高密度に凝縮した電文スタイルのテキスト。LSTMへの入力として最適な200文字程度)
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