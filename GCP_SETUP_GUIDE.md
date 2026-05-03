# GCP GCE プリエンプティブル T4 での TRIBEv2 Viewer セットアップ手順

## 概要

このドキュメントでは、Google Cloud Platform（GCP）の Compute Engine（GCE）で **NVIDIA T4 GPU** を搭載した **プリエンプティブル（スポット）インスタンス** を使用して、`TRIBEv2 Viewer` を動作させる手順を解説します。

**なぜ GCE プリエンプティブル T4 か**

- **料金**: 約 **$0.164/時間（約25円/時間）** と非常に安価
- **処理時間**: CPU の約 170 倍速（58 秒動画で **約 10〜15 分**）
- **新規アカウント特典**: **$300 無料クレジット**（90 日間有効）

---

## Step 1: GCP アカウント作成

1. [https://console.cloud.google.com/freetrial](https://console.cloud.google.com/freetrial) にアクセス
2. Google アカウントでログイン（新規でも既存でも可）
3. **「無料トライアルを開始」** をクリック
4. 請求先（クレジットカード）を登録

> 登録後、**$300 の無料クレジットが 90 日間有効** になります。

---

## Step 2: プロジェクト作成

1. GCP コンソール上部のプロジェクト選択ドロップダウンをクリック
2. **「新しいプロジェクト」**
3. プロジェクト名: `tribev2-viewer`（任意）
4. **「作成」**

---

## Step 3: GPU クォータの申請

**デフォルトでは GPU 割り当てが 0 のため、申請が必須です。**

1. コンソール左メニュー → **「IAM と管理」→「クォータ」**
2. フィルタで **「 NVIDIA T4 GPUs 」** を検索
3. 対象のリージョン（例: `us-west1`）を選択
4. **「クォータの編集」** をクリック
5. 新しい上限値に **1** を入力
6. **「送信」**
7. 申請理由に以下を入力（例）:
   ```
   Running a small machine-learning inference workload for personal research.
   Need 1x NVIDIA T4 GPU on preemptible/spot instances only.
   ```

> 承認まで通常 **数分〜数時間** かかります。

---

## Step 4: GCE インスタンス作成

### 4-1. VM インスタンス作成画面へ

1. コンソールメニュー → **「Compute Engine」→「VM インスタンス」**
2. **「作成」** をクリック

### 4-2. 基本設定

| 項目 | 設定値 |
|------|--------|
| 名前 | `tribev2-gpu` |
| リージョン | `us-west1`（オレゴン）または `asia-east1`（台湾） |
| ゾーン | `us-west1-b` |
| マシン構成 | **「GPU」タブを選択** |

### 4-3. GPU 設定

| 項目 | 設定値 |
|------|--------|
| GPU タイプ | **NVIDIA T4** |
| GPU 数 | 1 |
| マシンタイプ | `n1-standard-4`（4 vCPU, 15 GB） |

### 4-4. プロビジョニングモデル（最重要）

| 項目 | 設定値 |
|------|--------|
| プロビジョニングモデル | **「スポット」**（または「プリエンプティブル」） |
| 自動再作成 | **オフ**（1回きりの処理ならオフ推奨） |

> スポットインスタンスは通常価格の **1/3〜1/5** で利用できます。

### 4-5. ブートディスク

| 項目 | 設定値 |
|------|--------|
| オペレーティングシステム | **「Deep Learning on Linux」** |
| バージョン | `Debian 11 based Deep Learning VM for PyTorch 2.1 with CUDA 12.1`（最新を選択） |
| ブートディスクタイプ | **SSD 永続ディスク** |
| サイズ | **100 GB** |

> Deep Learning VM イメージには **PyTorch, CUDA, NVIDIA ドライバー** が事前インストール済みです。

### 4-6. ファイアウォール

| 項目 | 設定値 |
|------|--------|
| ネットワークタグ | `tribev2-server` |
| ファイアウォール | ☑ **HTTP トラフィックを許可** |
| | ☑ **HTTPS トラフィックを許可** |

### 4-7. 作成

下部の **「作成」** をクリック

> インスタンスが起動するまで約 **1〜2 分** かかります。

---

## Step 5: ファイアウォールルール追加（ポート 8000 開放）

HTTP/HTTPS だけではポート 8000 が開いていないため、追加が必要です。

1. コンソールメニュー → **「VPC ネットワーク」→「ファイアウォール」**
2. **「ファイアウォールルールを作成」**
3. 以下を入力:

| 項目 | 設定値 |
|------|--------|
| 名前 | `allow-tribev2-port8000` |
| ネットワーク | `default` |
| ターゲット | **「指定されたターゲットタグ」** |
| ターゲットタグ | `tribev2-server` |
| ソース IP の範囲 | `0.0.0.0/0` |
| プロトコルとポート | **「指定されたポートとプロトコル」→ TCP → `8000`** |

4. **「作成」**

---

## Step 6: SSH で接続

### 方法 A: ブラウザ内 SSH（最も手軽）

1. VM インスタンス一覧に戻る
2. `tribev2-gpu` の行で **「SSH」** 列の **「ブラウザ内で開く」** をクリック
3. 新しいブラウザウィンドウでターミナルが開く

### 方法 B: ローカルターミナルから

```bash
# gcloud CLI をインストール済みの場合
gcloud compute ssh tribev2-gpu --zone=us-west1-b
```

---

## Step 7: サーバー環境構築

SSH 接続後、以下を順番に実行します。

### 7-1. 依存関係インストール

```bash
sudo apt-get update && sudo apt-get install -y git ffmpeg

# Python 3.11 の確認
python3 --version

# pip アップグレード
python3 -m pip install --upgrade pip
```

### 7-2. リポジトリをクローン

```bash
cd ~

# このリポジトリをクローン
git clone <YOUR_REPO_URL> auto-excitement
cd auto-excitement

# tribev2 上流をクローン
git clone https://github.com/facebookresearch/tribev2.git tribev2-src
```

### 7-3. Python 依存関係をインストール

```bash
# requirements.txt の torch を CUDA 版に修正
sed -i 's/torch==2.11.0+cu128/torch==2.6.0/' requirements.txt

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt
pip install -e ./tribev2-src

# WhisperX インストール
pip install whisperx

# accelerate インストール（device_map 用）
pip install accelerate
```

### 7-4. パッチ適用

```bash
bash patches/apply.sh
```

### 7-5. アトラス構築

```bash
python build_atlas.py
```

### 7-6. 日本語 spaCy モデル

```bash
python -m spacy download ja_core_news_lg
```

### 7-7. HuggingFace ログイン

```bash
huggingface-cli login
# プロンプトにトークンを貼り付け
```

> HuggingFace トークンは [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) で生成し、
> [https://huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) でライセンス同意が必要です。

---

## Step 8: サーバー起動

```bash
source venv/bin/activate
nohup python server.py > server.log 2>&1 &
```

### 外部からアクセス

VM は外部 IP を持っているため、以下の URL でアクセス可能です:

```
http://<VMの外部IP>:8000
```

**外部 IP の確認方法**:
- GCP コンソール → VM インスタンス → `tribev2-gpu` の **「外部 IP」** 列

---

## Step 9: 処理完了後のインスタンス停止

料金を抑えるため、使い終わったら必ず停止または削除してください。

### 一時停止（ディスク料金のみ発生）

```bash
# VM 内から
sudo shutdown -h now
```

### 完全削除（課金ゼロ）

**GCP コンソールから**:
1. VM インスタンス一覧
2. `tribev2-gpu` を選択
3. **「削除」**

**または gcloud CLI**:

```bash
gcloud compute instances delete tribev2-gpu --zone=us-west1-b
```

> 削除するとブートディスクも消えるため、次回は環境構築からやり直しになります。
> 環境を保存したい場合は、停止（`stop`）のままにするか、**カスタムイメージ**を作成してください。

---

## 料金目安（プリエンプティブル T4）

| 項目 | 料金 |
|------|------|
| n1-standard-4（CPU/メモリ） | ~$0.05/時間 |
| NVIDIA T4 GPU | ~$0.10/時間 |
| SSD 永続ディスク 100GB | ~$0.014/時間 |
| **合計** | **~$0.164/時間（約25円/時間）** |
| 58秒動画の処理 | **約15分 = 約6円** |

新規アカウントの **$300 クレジット** があれば、**約1,800時間** 分（約75日間の連続使用）無料で使えます。

---

## トラブルシューティング

### GPU クォータの申請が承認されない

- 申請理由をより具体的に記載してください
- ビジネスメールアドレス（@gmail.com ではなく会社ドメイン）だと承認されやすい傾向があります
- 承認まで数日かかることもあります

### `AssertionError: Torch not compiled with CUDA enabled`

- PyTorch が CUDA 対応版ではない可能性があります
- Deep Learning VM イメージを使用している場合は通常発生しません
- 発生した場合は `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` を試してください

### プリエンプティブルインスタンスが途中で停止した

- プリエンプティブルインスタンスは GCP の空きリソースを利用するため、**24時間以内に確実に停止**されます
- 長時間の処理は途中で止まるリスクがあります
- 重要な処理には **通常インスタンス**（$0.35/時間程度）を使用してください

### ディスク容量不足

- モデルファイル（V-JEPA2, Llama 3.2, WhisperX 等）で **30〜50GB** 消費します
- 足りなくなったらブートディスクを拡張してください

---

## 参考リンク

- [GCP 無料トライアル](https://console.cloud.google.com/freetrial)
- [Compute Engine GPU ドキュメント](https://cloud.google.com/compute/docs/gpus)
- [スポット VM について](https://cloud.google.com/compute/docs/instances/spot)
- [Deep Learning VM イメージ](https://cloud.google.com/deep-learning-vm/docs)
- [HuggingFace Llama 3.2 モデルページ](https://huggingface.co/meta-llama/Llama-3.2-3B)

---

*作成日: 2026-05-03*
*対象リポジトリ: auto-excitement (TRIBEv2 Viewer)*
