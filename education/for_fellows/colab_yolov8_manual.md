
# YOLOv8動画処理のためのGoogle Colabマニュアル

このガイドは、初心者がGoogle Colab上でコードを実行し、必要なファイルをダウンロードして設定する方法を説明します。これにより、YOLOv8を使用して動画内の人物を検出し、カウントすることができます。

## 1. Google Colabを開く
- [Google Colab](https://colab.research.google.com)にアクセスし、新しいノートブックを作成します。

## 2. 必要なライブラリのインストール
コードを実行するために、必要なライブラリをインストールします。以下のコードをColabのセルにコピーして実行してください。

```python
!pip install ultralytics opencv-python-headless
```

これにより、YOLOライブラリ（`ultralytics`）とOpenCVがインストールされます。

## 3. YOLOモデルのダウンロード
YOLOv8のモデル（`yolov8n.pt`）をダウンロードする必要があります。以下のコマンドを実行してモデルをダウンロードしてください。

```python
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt
```

このコマンドにより、YOLOv8のnanoバージョンのモデルがColabの現在のディレクトリに保存されます。

## 4. 動画ファイルのアップロード
次に、動画ファイル（`710560399.045479.MP4`などのMP4ファイル）をGoogle Colabにアップロードします。以下のコードを実行すると、ファイル選択ウィンドウが開き、ローカルの動画ファイルをアップロードできます。

```python
from google.colab import files
uploaded = files.upload()
```

アップロードが完了したら、ファイル名を確認してください。`video_path`の値をアップロードしたファイルの名前に変更する必要があります。

## 5. パスの設定とコードの実行
動画ファイルのパスとモデルのパスを設定します。アップロードした動画ファイルの名前を確認し、それを`video_path`に指定してください。

```python
video_path = "710560399.045479.MP4"  # アップロードした動画ファイルの名前に置き換えてください
yolo_model_path = "yolov8n.pt"  # ダウンロードしたYOLOモデルのファイル名
```

次に、元のコードをColabにコピーして貼り付けてください。パスが正しく設定されていることを確認し、実行します。

## 6. コードの実行
コード全体を実行すると、動画内の人物をカウントし、その情報を`person_count_data.csv`というCSVファイルに保存します。処理が完了すると、CSVファイルが生成されます。

## 7. 結果のダウンロード
最後に、生成されたCSVファイルをダウンロードします。以下のコードを使用してファイルをダウンロードできます。

```python
from google.colab import files
files.download("person_count_data.csv")
```

これにより、CSVファイルがローカルにダウンロードされ、分析に使用できます。

## 注意点
- 動画ファイルが大きい場合、処理に時間がかかることがあります。
- `display_frames`が`True`に設定されている場合、フレームを表示しようとしますが、ColabではGUIウィンドウを表示できないため、このオプションを`False`に設定することをお勧めします。

これで、初心者向けにGoogle Colabでこのコードを実行するための手順が完成しました。
