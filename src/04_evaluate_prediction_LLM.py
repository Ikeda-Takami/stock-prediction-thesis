import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==========================================
# ★ 設定エリア（ここを変えるだけでOK！） ★
# ==========================================
# 解析したい期間
START_DATE = "2024-08-01"
END_DATE   = "2024-08-31" 

# ★追加：使用したモデル名（レポートに記載されます）★
MODEL_NAME = "GPT-OSS-120B" 
# ※ここを "Ollama(Llama3-70B)" や "Gemma-2-27B" など書き換える

# 入力フォルダ（推論結果のテキストがある場所）
PREDICTION_DIR = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/summary_test_august/"

# 正解データのCSVパス
FINANCE_CSV = "/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/finance/nikkei_225_labeled.csv"

# 結果レポートの保存先
REPORT_FILE = f"/home/sakulab/workspace/B4_ikeda/graduation_thesis/result/LLM_{START_DATE}_{END_DATE}/evaluation_report_{START_DATE}_{END_DATE}.txt"
# ==========================================

# 保存先のフォルダパスを取り出す
report_dir = os.path.dirname(REPORT_FILE)
os.makedirs(report_dir, exist_ok=True)

# ログ出力用の関数
def log_print(text, file_obj):
    print(text)
    file_obj.write(text + "\n")

# レポートファイルを開く
with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
    
    log_print(f"🚀 評価パイプライン起動！", f_out)
    # ★追加：ここでモデル名を記録★
    log_print(f"🤖 使用モデル: {MODEL_NAME}", f_out)
    log_print("-" * 30, f_out)
    log_print(f"📅 対象期間: {START_DATE} 〜 {END_DATE}", f_out)
    log_print(f"📂 読み込み先: {PREDICTION_DIR}", f_out)

    # 1. 正解データの読み込み
    if not os.path.exists(FINANCE_CSV):
        log_print("❌ 正解データCSVがありません。03.1 を先に実行してください。", f_out)
        exit()

    df_finance = pd.read_csv(FINANCE_CSV)
    finance_map = df_finance.set_index("Date_Str")["Actual_Label"].to_dict()

    # 2. LLMの予測を収集
    all_files = sorted(glob.glob(os.path.join(PREDICTION_DIR, "summary_*.txt")))
    
    y_true = []     # 正解
    y_pred = []     # 予測
    valid_dates = [] 

    count_target = 0 # 期間内のファイル数

    for file_path in all_files:
        filename = os.path.basename(file_path)
        date_str = filename.replace("summary_", "").replace(".txt", "")

        # 期間フィルタリング
        if not (START_DATE <= date_str <= END_DATE):
            continue
        
        count_target += 1

        # 正解データがない日（土日祝）はスキップ
        if date_str not in finance_map:
            continue

        # テキスト読み込み
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # --- 判定ロジック ---
        prediction = None
        header_text = content[:50] 
        
        if "【上昇】" in header_text:
            prediction = 1
        elif "【下落】" in header_text:
            prediction = 0
        
        if prediction is not None:
            actual = finance_map[date_str]
            y_true.append(actual)
            y_pred.append(prediction)
            valid_dates.append(date_str)
        else:
            log_print(f"⚠️ {date_str}: 判定不能（【上昇/下落】が見つかりません）", f_out)

    log_print(f"📄 期間内のファイル数: {count_target} 件", f_out)
    log_print(f"✅ 有効な評価ペア数: {len(y_true)} 件 (土日祝除く)", f_out)

    # 3. スコア計算
    if len(y_true) == 0:
        log_print("⚠️ 有効なデータペアが見つかりませんでした。", f_out)
        exit()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    log_print("\n" + "="*40, f_out)
    log_print(f"📊 評価レポート: {MODEL_NAME}", f_out) # ★ここにもモデル名を表示
    log_print("="*40, f_out)
    log_print(f"✅ Accuracy  (正解率): {acc:.2%}", f_out)
    log_print(f"🎯 Precision (適合率): {prec:.2%}", f_out)
    log_print(f"📢 Recall    (再現率): {rec:.2%}", f_out)
    log_print(f"⭐ F1 Score  (F値)   : {f1:.4f}", f_out)
    log_print("="*40, f_out)

    # 詳細レポート
    report_str = classification_report(y_true, y_pred, target_names=["下落", "上昇"])
    log_print("\n--- 統計詳細 ---", f_out)
    log_print(report_str, f_out)

    # 日別の勝敗
    log_print("\n--- 日別詳細結果 ---", f_out)
    log_print("日付        | 正解 | 予測 | 判定", f_out)
    log_print("-" * 35, f_out)
    
    for i in range(len(valid_dates)):
        d = valid_dates[i]
        t = "上昇" if y_true[i] == 1 else "下落"
        p = "上昇" if y_pred[i] == 1 else "下落"
        res = "⭕" if y_true[i] == y_pred[i] else "❌"
        log_print(f"{d} | {t} | {p} | {res}", f_out)

print(f"\n💾 レポートを保存しました: {REPORT_FILE}")