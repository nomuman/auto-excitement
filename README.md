# Auto Excitement

動画からTRIBEv2モデルで脳活動を予測し、動画と同期してグラフや脳表面に表示する実験用Webツールです。[shi3z/auto-excitement](https://github.com/shi3z/auto-excitement) のフォークです。

## 実装している機能

- ブラウザからの動画アップロードと処理進捗の表示
- WhisperXを使う文字起こしと、日本語を英訳して推論する経路
- 7つの脳ネットワーク、4つの指標、主成分の時系列表示
- Three.jsによる脳表面の表示と、動画・波形・字幕・サムネイルの同期
- 指標のしきい値に基づく区間選択、スキップ再生、ffmpeg用コマンドの出力

指標はモデルの予測から計算した代理指標です。実際の視聴者の脳活動や感情を計測しているわけではありません。Excitement、Valence、Cognitive Load、Noveltyという名前も、主観的な興奮・感情・認知負荷・新奇性を直接測る尺度ではありません。

## フォークの位置づけ

このリポジトリは上流のアプリを引き継いでいます。機能一覧は現在含まれるコードの説明であり、nomumanがすべてを独自開発したという意味ではありません。変更履歴は [コミット一覧](https://github.com/nomuman/auto-excitement/commits/main/) で確認できます。

## 起動の準備

Python環境に加え、ffmpeg、`uvx whisperx`、TRIBEv2のソースとモデルが必要です。上流READMEではPython 3.11とCUDA GPUを推奨しています。現在の依存関係での再現性は今回確認していません。

```sh
git clone https://github.com/nomuman/auto-excitement.git
cd auto-excitement
git clone https://github.com/facebookresearch/tribev2.git tribev2-src
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ./tribev2-src
patches/apply.sh
python build_atlas.py
```

日本語モードを使う場合は、追加で次を実行します。

```sh
python -m spacy download ja_core_news_lg
```

WhisperXとuvは別途利用できる状態にしてください。モデルやアトラスの取得には外部通信とディスク容量が必要です。

## 実行

```sh
python server.py
```

`http://localhost:8000` をブラウザで開きます。現在のサーバーは `0.0.0.0:8000` で待ち受けます。利用者認証を前提とした公開サービスの構成ではないため、アクセスを制限できる実験環境で使ってください。

## 制約と確認範囲

指標は動画内で標準化しており、別の動画との絶対値比較には使えません。日本語を英訳する経路の字幕時刻は区間単位をもとにしており、元の日本語字幕を表示するものではありません。

今回のREADME更新ではサーバー、画面、依存関係とパッチを照合しました。モデルの取得・推論、GPU上での速度、文字起こし、動画の書き出しは実行していません。過去の速度の記述を、このフォークで再現した性能としては掲載していません。

## コードとライセンス

| 場所 | 内容 |
|---|---|
| [server.py](server.py) | FastAPI、推論、指標の計算、進捗の配信 |
| [static/index.html](static/index.html) | グラフ、脳表面、動画カット用UI |
| [build_atlas.py](build_atlas.py) | 脳ネットワークのアトラス生成 |
| [patches](patches) | TRIBEv2・neuralset向けの変更 |
| [smoke_predict.py](smoke_predict.py) | 単体の推論確認用スクリプト |

元READMEはコードをMITとしていますが、このリポジトリにはLICENSEファイルがありません。再配布時は上流の著作権表示と利用条件を確認してください。TRIBEv2とアトラスも、それぞれの利用条件・引用条件に従います。
