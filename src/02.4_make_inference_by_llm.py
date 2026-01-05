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
START_DATE = "2024-06-01" 
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
OUTPUT_DIR = os.path.join(BASE_DIR, "summary_reasoning_augasuto_v4/")
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
# ★重要: 過去10日分を正しく取るために日付でソートする
df_market.sort_index(inplace=True)
print(" -> ロード完了！")

# ------------------------------------------
# 2. 推論ループ
# ------------------------------------------
all_files = sorted(glob.glob(os.path.join(INPUT_CORPUS_DIR, "corpus_*.txt")))
target_files = []

for f in all_files:
    d_str = os.path.basename(f).replace("corpus_", "").replace(".txt", "")
    if START_DATE <= d_str <= END_DATE:
        target_files.append(f)

print(f"🔥 LLM推論モード起動！ (Trend 10-days ver)")
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
        print(f"[{i+1}/{len(target_files)}] 💤 {date_str} (休日) -> バッファに追加")
        weekend_buffer += f"\n--- {date_str} のツイート ---\n{daily_text}"
        weekend_dates.append(date_str)
        continue 

    else:
        # ■ 平日（市場が開いている日）の場合
        save_path = os.path.join(OUTPUT_DIR, f"summary_{date_str}.txt")
        
        # 生成済みならスキップ
        if os.path.exists(save_path):
            weekend_buffer = "" 
            weekend_dates = []
            continue

        print(f"[{i+1}/{len(target_files)}] 🚀 {date_str} (平日) 推論開始 ", end="", flush=True)
        start_time = time.time()

        # テキストの結合（週末分 + 当日分）
        if weekend_buffer:
            print(f" [週末分({len(weekend_dates)}日)を統合]", end="")
            full_text = weekend_buffer + f"\n--- {date_str} (当日) のツイート ---\n{daily_text}"
            weekend_buffer = ""
            weekend_dates = []
        else:
            full_text = daily_text

        # 文字数制限（間引き）
        total_chars = len(full_text)
        if total_chars > SAFE_CHAR_LIMIT:
            step = math.ceil(total_chars / SAFE_CHAR_LIMIT)
            lines = full_text.splitlines()
            text_data = "\n".join(lines[::step])
            status = f"(間引1/{step})"
        else:
            text_data = full_text
            status = "(全量)"

        # --- ★ここを変更: 過去10日分の市況データを取得 ---
        market_text = ""
        try:
            # 現在の日付の行番号(index location)を取得
            curr_idx = df_market.index.get_loc(current_date)
            
            # 過去10日分（現在含む）のスライス範囲を計算
            # 例: indexが100なら、91〜100 (計10個) を取る
            start_idx = max(0, curr_idx - 9)
            recent_data = df_market.iloc[start_idx : curr_idx + 1]
            
            # テキスト化
            lines = []
            lines.append("日付        | 日経終値 | 日経前日比 | ダウ変化 | ドル円")
            lines.append("-" * 50)
            
            for dt, row in recent_data.iterrows():
                # データ取り出し
                n_close  = row['Nikkei_Close']
                n_change = row['Nikkei_Change'] if not pd.isna(row['Nikkei_Change']) else 0
                d_change = row['Dow_Change']    if not pd.isna(row['Dow_Change']) else 0
                u_close  = row['USDJPY_Close']
                
                # 符号付け
                n_sign = "+" if n_change > 0 else ""
                d_sign = "+" if d_change > 0 else ""
                
                # 1行の文字列を作成 (例: 2024-10-01 | 38000 | +100 | +50 | 145.00)
                line = f"{dt} | {n_close:,.0f} | {n_sign}{n_change:,.0f} | {d_sign}{d_change:,.0f} | {u_close:.2f}"
                lines.append(line)
            
            market_text = "\n".join(lines)
            
        except Exception as e:
            market_text = f"（データ取得エラー: {e}）"

        # --- プロンプト構築 ---
        my_prompt = f"""
SYSTEM ROLE definition:
あなたは「SENTINEL-X」です。感情を持たない高頻度取引アルゴリズムのデータ処理ユニットです。
あなたの目的は、入力データを下流の予測モデル（LSTM）のために「数値ベクトル化可能な高密度シグナル」に変換することです。
金融アドバイスのような「ヘッジ（曖昧な表現）」は禁止されています。あなたは「計算機」として、0か1かの明確な判断を下してください。

【入力データ】
▼ 市況データ（直近10営業日のトレンド推移）
※一番下の行が「本日（予測対象日の前日）」の確定値です。この流れ（モメンタム）を読み取ってください。
{market_text}

▼ SNS投資家の発言（本日分のみ）
※【重要】週末（土日）データが含まれる場合の解釈プロトコル：
1. **「エネルギーの充填」として扱う**: 土日の投稿量は、月曜日の「窓開け（Gap Open）」の方向と強さを示唆します。
2. **「織り込み（Discounting）」の判定**: 土日の悪材料が月曜の寄り付きで織り込まれ、日中は「材料出尽くし」で反発する逆説的な動きを考慮してください。
3. **「夜明けの補正」**: 直近（月曜朝）の投稿ほど重みを置いて解釈してください。
{text_data}

---

【推論プロトコル】

### STEP 1: FinCoT (市況データと高度な金融知識によるゾーン判定)
**役割: テクニカル・ストラテジスト**
- **10日間のトレンド分析**: 入力された過去10日間の価格推移から、現在の相場が「上昇トレンド」「下降トレンド」「保合い」のどの局面にあり、勢い（モメンタム）が加速しているか減速しているかを分析してください。
- 以下の3つのゾーンのいずれかに分類します。
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
  - STEP 1で分析した「数値トレンド」に対し、「Soft Signals」が逆行している場合（例：数値は上昇トレンドなのにSNSがパニック）、それは**「ノイズ」**あるいは**「逆張り機会」**です。
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
STEP1: (トレンド分析。直近10日間の市況データの推移に基づき、現在のトレンドとゾーン判定を行う。断定的な口調で200文字程度記述)
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