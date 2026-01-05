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
END_DATE   = "2024-12-31"

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
OUTPUT_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/summary_full_v2/"

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
        あなたは世界トップクラスのヘッジファンドで運用責任者を務める、伝説的な金融アナリストです。
以下のテキストは、市場参加者の発言（SNS/ニュース）です。
これに基づき、翌日の日経平均株価の動きを論理的に推論してください。

【指示：以下の思考プロセス（Blueprint）に従って分析せよ】

Step 1: ノイズの除去と解釈
- テキストに含まれる感情的な叫びや、根拠のない買い煽り/売り煽り（ノイズ）を無視してください。
- スラングや皮肉が含まれる場合、その裏にある「真の意図」を文脈から読み取ってください。

Step 2: 構造化された推論 (Structured Reasoning)
- 以下の3つの観点から市場心理を分析し、400文字程度の論理的な文章を作成してください。
  1. 【センチメント】: 投資家は恐怖しているか、楽観しているか？
  2. 【外部要因】: 為替（ドル円）や米国市場（ダウ・ナスダック）への言及はあるか？
  3. 【需給・材料】: 特定のセクターへの資金流入や、材料出尽くし感はあるか？

Step 3: 結論と自己検証
- 分析に基づき、明日は「上昇」か「下落」かを断定してください。
- その結論に至った「決定的な理由」を明記してください。

--- 入力データ ---
{text_data}
--- 入力データ終了 ---

【出力形式】
必ず以下のフォーマットのみを出力してください。
【出力形式の絶対ルール】
1. **「センチメント：～」や「外部要因：～」といった見出しや箇条書きは禁止です。**
2. まるで新聞のコラムのように、すべての要素が一つの論理的な流れ（ストーリー）になるように書いてください。
3. 必ず以下の形式のみを出力してください（結論と本文の間には必ず改行を入れてください）。

【上昇】
ここに本文...

または

【下落】
ここに本文...

【上昇】 または 【下落】
（ここにStep 2で構築した、論理的かつ専門的な分析レポートを記述）
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