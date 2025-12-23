import pandas as pd
import os
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
INPUT_CSV = '/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/raw/Nikkei_filtered.csv'
OUTPUT_DIR = '/home/sakulab/workspace/B4_ikeda/graduation_thesis/data/corpus/'

def create_corpus():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("データを読み込んでいます...")
    df = pd.read_csv(INPUT_CSV)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['utc_date'] = df['created_at'].dt.date
    
    grouped = df.groupby('utc_date')
    
    print(f"合計 {len(grouped)} 日分のコーパスを作成します。")

    for date_value, group_df in tqdm(grouped):
        date_str = date_value.strftime('%Y-%m-%d')
        filename = f"corpus_{date_str}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # 時系列順にソート
        group_df = group_df.sort_values('created_at')

        # === 【ここが修正ポイント】 ===
        # 1. dropna(): 空データを消す
        # 2. astype(str): 文字列にする
        # 3. リスト内包表記で加工:
        #    - ツイート内の改行(\n)をスペースに置換（1ツイート1行にするため）
        #    - 頭に "- " をつける
        formatted_tweets = [
            "- " + t.replace('\n', ' ') 
            for t in group_df['text'].dropna().astype(str).tolist()
        ]
        
        # 改行で結合
        daily_text = '\n'.join(formatted_tweets)
        # ============================
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(daily_text)

    print("完了しました！")
    print(f"保存先: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_corpus()