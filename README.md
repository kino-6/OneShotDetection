# OneShotDetection

**OneShotDetection** は、効率的で軽量な物体検出フレームワークであり、YOLO (You Only Look Once) シリーズ、特に **YOLOv5.6.2** に強く触発されています。本プロジェクトは、GPLの回避とコードベースのリファクタリングを通じて、よりモジュール化された学習可能なフレームワークを目指しています。

なお、trainは未実装であり、学習済みモデルを動作可能な部分まで作成しています。

**OneShotDetection** is an efficient and lightweight object detection framework, strongly inspired by the YOLO (You Only Look Once) series, especially **YOLOv5.6.2**. This project aims to be a more modular and learnable framework through GPL circumvention and refactoring of the code base.

Note that TRAIN has not yet been implemented and the trained model has been created to the point where it is operational.

* [yolov5.6.2](https://github.com/ultralytics/yolov5/tree/v6.2)

## プロジェクトの背景

YOLOシリーズは、物体検出の分野で革命をもたらし、非常に効率的で直感的な設計を提供してきました。一方で、YOLOv5.6.2 のコードベースはそのライセンスの特性上、特定の用途での制約が存在する場合があります。

**OneShotDetection** は以下を目的として設計されています：

1. **GPL回避**: YOLOv5.6.2 のライセンス制約を回避し、オープンで柔軟な利用を可能にする。
2. **リファクタリングと学びなおし**: オリジナルのコードベースを深く理解しつつ、独自の設計と実装を追加。
3. **モジュール化と拡張性**: 開発者が機能を拡張しやすいように、より明確で管理しやすいコード構造を提供。

The YOLO series has revolutionized the field of object detection, providing a highly efficient and intuitive design. On the other hand, due to the nature of its license, the YOLOv5.6.2 codebase may have limitations in certain applications.

**OneShotDetection** is designed to: 1:

1. **GPL Avoidance**: Avoid the license restrictions of YOLOv5.6.2 and allow open and flexible use.
2. **Refactoring and relearning**: add your own design and implementation while maintaining a deep understanding of the original code base.
3. **Modularization and Extensibility**: Provides a clearer, more manageable code structure that makes it easier for developers to extend functionality.

Translated with DeepL.com (free version)

## 特徴

- **YOLOの思想を継承**:
  - 一回の処理で複数の物体を効率的に検出。
  - YOLOと互換性のあるモデル設計。
- **軽量かつモジュール化**:
  - コードベースを分離・再構築し、簡潔な設計を追求。
- **ライセンスフリー**:
  - GPLを回避した実装で商業利用や研究利用の幅を拡大。
- **学習と再設計**:
  - YOLOv5.6.2の構造を深く学びながら、より理解しやすいコードへ。

## 謝辞

このプロジェクトは、**YOLOv5** のアイデアと設計に深く依存しており、YOLOシリーズの貢献者全員に心からの敬意を表します。

YOLOは、物体検出の世界における画期的なフレームワークであり、**OneShotDetection** はその効率性と直感的な設計を受け継ぎつつ、独自の工夫を加えています。本プロジェクトが存在するのは、YOLOが築いた強固な基盤のおかげです。

## インストールと使い方

### 1. インストール

```bash
git clone https://github.com/username/oneshotdetection.git
cd oneshotdetection
pip install -r requirements.txt
```

### 2-1. モデルのビルド

```python
from oneshotdetection import Model

cfg = "./models/yolov5s.yaml"
model = Model(cfg, ch=3, nc=80, anchors=None)
```

### 2-2. モデルの読み込み

```python
from oneshotdetection import Utils

pt_file = "./yolov5s.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Utils.load_model_safely(pt_file, device)
```

### 3. 推論実行

* [zidane.jpg](https://github.com/ultralytics/yolov5/blob/v6.2/data/images/zidane.jpg)
* [yolov5s_weights.pt](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt)

```bash
python3 -m oneshotdetection.main \
    --image data/images/zidane.jpg \
    --model bin/yolov5s_weights.pt \
    --config config/size_s.yaml \
    --input-shape 640 640 \
    --output-dir runs \
    --output-file result.jpg
```

### result

* zidane.jpgを推論した結果
  * ターミナルに推論結果を表示
  * runs/<--output-dir>/<--output-file> に結果を重畳表示した画像を出力

```bash
====================================================================================================
0: {'box': [742.4283447265625, 48.47918701171875, 1140.6806640625, 719.4008178710938], 'confidence': 0.8838440179824829, 'class_id': 0, 'class_name': 'person'}
1: {'box': [441.4966735839844, 437.0728759765625, 496.1723327636719, 711.253662109375], 'confidence': 0.6632153987884521, 'class_id': 27, 'class_name': 'tie'}
2: {'box': [124.39743041992188, 194.07125854492188, 713.905517578125, 716.0787353515625], 'confidence': 0.6595578193664551, 'class_id': 0, 'class_name': 'person'}
3: {'box': [982.9000244140625, 308.60101318359375, 1027.2591552734375, 420.32452392578125], 'confidence': 0.2592342495918274, 'class_id': 27, 'class_name': 'tie'}
Visualization saved to: runs/runs_00000/result.jpg
====================================================================================================
```
