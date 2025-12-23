import os
import glob
import ollama
import time
import math

# ==========================================
# ★ 期間設定（ここを書き換えて範囲を調整！） ★
# ==========================================
# 2017-08-31 から 2024-12-31 までの全日程を対象にします
START_DATE = "2017-08-31" 
END_DATE   = "2024-03-31"

# ==========================================
# ★ LLM推論設定（Dual RTX 120Bモデル専用） ★
# ==========================================
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11500"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

MODEL_NAME = "gpt-oss:120b"
CONTEXT_SIZE = 131072  # 8月の実験で成功した限界値
SAFE_CHAR_LIMIT = 100000 # コンテキストに収めるための文字数上限

# フォルダパス
INPUT_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/corpus/"
OUTPUT_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/summary_full_v1/"

# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全ファイルを取得して期間内でフィルタリング
all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "corpus_*.txt")))
target_files = []
for f in all_files:
    date_part = os.path.basename(f).replace("corpus_", "").replace(".txt", "")
    if START_DATE <= date_part <= END_DATE:
        target_files.append(f)

print(f"🔥 LLM推論モード起動！")
print(f"📅 期間: {START_DATE} 〜 {END_DATE} ({len(target_files)}日間)")
print(f"🧠 使用モデル: {MODEL_NAME} (num_ctx: {CONTEXT_SIZE})")
print("-" * 50)

for i, file_path in enumerate(target_files):
    filename = os.path.basename(file_path)
    date_str = filename.replace("corpus_", "").replace(".txt", "")
    save_path = os.path.join(OUTPUT_DIR, f"summary_{date_str}.txt")

    # 完了済みはスキップ（中断してもここから再開できる）
    if os.path.exists(save_path):
        continue

    print(f"[{i+1}/{len(target_files)}] 🚀 {date_str} を推論中...", end="", flush=True)
    start_time = time.time()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        total_chars = sum(len(line) for line in lines)
        
        # 間引き（サンプリング）
        if total_chars > SAFE_CHAR_LIMIT:
            step = math.ceil(total_chars / SAFE_CHAR_LIMIT)
            text_data = "".join(lines[::step])
            status = f" (間引1/{step})"
        else:
            text_data = "".join(lines)
            status = " (全量)"

        # 記者の魂を込めたプロンプト
        my_prompt = f"""
        あなたは金融市場に精通したベテラン経済記者です。
        以下のテキストは、ある一日の市場参加者のツイートを抽出したものです。

        【タスク1：明日の予測】
        センチメントから翌営業日の日経平均株価が「上がる」か「下がる」か予測し、
        冒頭に【上昇】または【下落】と記してください。

        【タスク2：市況の要約】
        その予測の根拠を、200文字程度の自然な日本語でまとめてください。
        ※箇条書き、見出し、体言止めは禁止。

        --- 入力データ ---
        {text_data}
        --- 入力データ終了 ---

        【再確認】
        必ず日本語で回答してください。
        最初に【上昇】か【下落】、その直後に要約文を記述してください。
        """

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": my_prompt}],
            options={
                "temperature": 0.1,
                "num_ctx": CONTEXT_SIZE,
                "seed": 42
            }
        )

        # 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response["message"]["content"])

        elapsed = time.time() - start_time
        print(f" ✅{status} {elapsed:.1f}秒")

    except Exception as e:
        print(f" ❌ エラー: {e}")
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a") as f:
            f.write(f"{date_str}: {str(e)}\n")

print("-" * 50)
print(f"✨ 全日程の推論が完了しました！")
print(f"📁 保存先: {OUTPUT_DIR}")