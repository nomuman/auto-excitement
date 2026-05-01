# Auto Excitement

短い動画をブラウザにアップロードすると、Meta の `facebook/tribev2` モデルが
fsaverage5 表面 (20484 頂点) の BOLD 反応を予測 → Yeo7 ネットワーク平均と
多次元ブレインステート軸 + PCA latent → 動画再生に同期して時系列・脳画像・
タイムラインを表示し、退屈区間の ffmpeg カット script まで吐き出す
シングルファイル GUI です。

## できること

- 動画 (mp4/webm/mov…) のアップロードと進捗バー（実バイト数 + 転送速度）
- WhisperX で音声→単語タイムスタンプ。**日本語**（そのまま、または英訳して推論）も対応
- TRIBEv2 で 20484 頂点 × N TR の BOLD 予測
- 結果可視化:
  - 動画プレーヤ＋再生位置に追従する縦カーソル
  - Chart.js: Yeo7 ネットワーク平均 + 4 軸（Excitement / Valence / Cognitive Load / Novelty）
  - サムネイル × 脳活動マップ（fsaverage5 上の予測活動を fire colormap で）の filmstrip
  - 音声波形 + 文字起こし + 4 軸線グラフ + PCA top-3 のスタックドタイムライン
  - 4 軸スコアのリアルタイム数値表示
- Excitement 閾値で**退屈シーンを自動カット**:
  - 閾値・最低継続秒数のスライダ、保持区間のグレーアウトプレビュー
  - ブラウザ内**プレビュー再生**（カット区間を即時にスキップ）
  - `ffmpeg` の `select` フィルタを組んだ `.sh` ファイル / コマンドの書き出し

## ディレクトリ構成

| パス | 役割 |
|---|---|
| `server.py` | FastAPI サーバ。TribeModel・Yeo7 アトラス・matplotlib プロッタを起動時に常駐させ、SSE で進捗を流す |
| `build_atlas.py` | Yeo 2011 7-network MNI152 ボリュームを fsaverage5 表面へ投影 |
| `smoke_predict.py` | モデル単体のスモークテスト |
| `static/index.html` | 単一ページ GUI（Chart.js を CDN から読み込み） |
| `patches/` | 上流 `facebookresearch/tribev2` と `neuralset` への必須パッチ（後述） |
| `cache/` *(.gitignore)* | モデル中間ファイル + アトラス |
| `static/videos/` *(.gitignore)* | アップロード済み動画（静的配信） |
| `static/thumbs/`, `static/brain/` *(.gitignore)* | 生成サムネ・脳活動 PNG |
| `tribev2-src/` *(.gitignore)* | 上流 tribev2 クローン。別途自分で clone する |

## セットアップ

```bash
git clone <THIS REPO>
cd tribev2-viewer

# 1) tribev2 上流を別途 clone（このリポジトリには含まれない）
git clone https://github.com/facebookresearch/tribev2.git tribev2-src

# 2) venv と依存関係
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ./tribev2-src

# 3) 必須パッチ（日本語対応 + --task translate のため）
patches/apply.sh

# 4) Yeo7 アトラスを fsaverage5 へ投影
python build_atlas.py

# 5) 日本語 spaCy モデル（japanese / japanese_translate モードで必要）
python -m spacy download ja_core_news_lg
```

`uvx whisperx` がパス上にあること（音声→単語タイムスタンプ）。
GPU は CUDA 対応推奨。CPU でも動くが V-JEPA2 が桁違いに遅い。

### 上流に当てているパッチ

`patches/apply.sh` が以下を当てます（再実行は no-op）:

- `patches/tribev2.patch` — `ExtractWordsFromAudio` に `task` (transcribe/translate) を追加、`japanese="ja"` を WhisperX 言語マップへ追加、`--align_model` の空文字バグ修正、`get_audio_and_text_events`/`get_events_dataframe` に `language` 引数を追加、`japanese_translate` 擬似言語の経路を追加
- `patches/neuralset.patch` — `neuralset/utils.py` の spaCy 言語マップに `japanese="ja_core_news_lg"` と ISO `"ja"→"japanese"` を追加

## 起動

```bash
source venv/bin/activate
python server.py            # http://localhost:8000
```

`mp4` などを選択して **予測する** を押すと、

1. アップロード（バイト数進捗）
2. サーバが `model.get_events_dataframe(video_path=..., language=...)` でイベント DF 構築
3. `model.predict(events=df)` が 20484 頂点 × N セグメントを出す
4. Yeo7 ネットワーク平均化 + 4 軸 + PCA top-3 を計算、サムネ + 脳画像 + 波形を生成
5. SSE 経由でフロントへ、Chart.js + filmstrip + タイムライン + 退屈カットツールが現れる

性能の目安: 52 秒の Sintel クリップで end-to-end 約 4–5 分（動画 extractor が支配的）。

## 4 軸の定義（経験則プロキシ + PCA latent）

```
Excitement     = (z(VIS+SMN+DAN+VAN) − z(DMN)) / 2     # 感覚・注意の関与
Valence        = z(Limbic)                              # 感情価のプロキシ
Cognitive Load = z(mean(FrontoParietal, DorsalAttn))    # 制御要求／作業記憶
Novelty        = z(VAN − mean(全 Yeo7))                 # Salience の baseline 超過分（≈予測誤差）
PC1–PC3        = numpy.linalg.svd(preds_centered) の上位 3 成分（z-score）
```

> **注**：軸の名前は事後解釈。`Cognitive Load` の絶対値や、`Novelty` を主観的「驚き」と
> 直結させるのは過大解釈です。1 次元の「興奮↔退屈」では捉えきれない部分を、独立軸 +
> 純データ駆動の latent 空間で補っているという位置付け。

## API

`POST /predict` (multipart):
- `video` (file) — 動画
- `language` (str, default `english`) — `english` / `japanese` / `japanese_translate` / `french` / `spanish` / `dutch` / `chinese`

レスポンス: `{job_id, video_url, language}`

`GET /events/{job_id}` — Server-Sent Events:
- `data: {"type":"progress","phase":"...","percent":N}`
- `data: {"type":"log","message":"..."}`
- `data: {"type":"done"}` → 続いて `event: result\ndata: {...}`
- 失敗時: `data: {"type":"error","message":"..."}` および `event: error`

`result` ペイロード:

```jsonc
{
  "n_segments": 53, "tr": 1.0,
  "times": [...], "durations": [...],
  "networks": { "Visual": [...], "Somatomotor": [...], ... },
  "axes": {
    "excitement": [...], "valence": [...],
    "cognitive_load": [...], "novelty": [...]
  },
  "pca": [[...], [...], [...]],   // 上位 3 PC、長さ n_segments
  "pca_var": [0.73, 0.13, 0.08],  // 寄与率
  "thumbs": [{"t": 0.0, "url": "/static/thumbs/<job>/0000.jpg"}, ...],
  "brain_urls": ["/static/brain/<job>/0000.png", ...],
  "waveform": [0.0, 0.12, ...],   // 1500 ビンの正規化エンベロープ
  "words": [{"t": 12.21, "d": 0.12, "text": "What"}, ...],
  "elapsed_sec": 270.5
}
```

## 既知の制限

- セグメント数は動画長 ÷ TR(=1s) × kept_ratio。`predict()` 内のフィルタで一部落ちる。`times[i]` は実時刻
- 軸スコアは z-score なのでクリップ間で絶対値比較に意味は無い
- Yeo アトラスは `vol_to_surf(interpolation="nearest_most_frequent")` の MNI→fsaverage5 投影で若干の境界誤差。厳密な解析用途には FreeSurfer の `?h.Yeo2011_7Networks_N1000.annot` を直接使用推奨
- `japanese_translate` モードは WhisperX の `--task translate` 単一パス。タイムスタンプは segment-level → 単語均等分割。原語の日本語 transcript は UI に出ない
- TRIBEv2 自体は英語の自然刺激（映画/オーディオブック）で訓練されたモデル。非英語言語を `japanese` モード（そのまま）で投入すると、視覚・聴覚ネットワークは反応するが言語ネットワークの予測精度は明確に落ちる

## トラブルシュート

- `requests.exceptions.RequestException: timed out` — `build_atlas.py` は `urllib` で直接 DL する実装に切替済み、再実行で通る
- ディスクが満杯 → `~/.cache/uv` (再生成可) を削除。HuggingFace の他モデルキャッシュは慎重に
- ポート 8000 が衝突 → `python server.py` を編集するか `uvicorn server:app --port 9000`
- WhisperX が `argument --align_model: expected one argument` で落ちる → `patches/apply.sh` 未適用。空文字を渡すバグの修正が当たっていない

## ライセンス

このリポジトリのコードは MIT。`tribev2-src/` 以下と `patches/` の対象部分は
`facebookresearch/tribev2` のライセンス（MIT）に従います。Yeo 2011 アトラスは
[Yeo et al. 2011 J. Neurophysiol.](https://doi.org/10.1152/jn.00338.2011) のクレジット表示が必要です。
