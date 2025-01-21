# 顔学習モデルを作成するための、十徳ナイフの使い方

顔認証システムの核、顔学習モデルの作成には苦労が伴います。

お金に物を言わせれば計算資源は確保できます。またわたしの記事では「どのようなバックボーンと損失関数を使用するか」をコードとして紹介していますし、昨今なら大規模LLMにコードを書かせることもできるでしょう。

苦労するポイントは「**良質な顔データセットを作成**」するところにあります。

昔はスクレイピングひとつにしてもオライリーの本が出ていたほどですが、これも大規模LLMがコードを書いてくれます。[^1]
[^1]: 個人の経験値に基づいた、秘伝のタレ的コードも、そのうち出来るようになるのでしょうね。

**しかしそこから先、何百GBも画像を集めて、どうすればいいですか？**

この記事では「そこから先」をご案内し、その上で使える十徳ナイフの使い方をご紹介します。

顔学習モデル作成に限りませんが、**深層学習はデータセットが9割**です。

ここでは拙作の顔学習フレームワークFACE01を使って、半自動的に良質な顔データセットを作成する方法について共有します。

![](https://raw.githubusercontent.com/yKesamaru/FACE01_DEV/master/assets/images/eye-catch.png)

## 前提
:::details
- ホストマシンにGPUが搭載されているとします。（VRAM4GBでも構いません）
- Linuxマシンで実行することを想定しています。
- x11環境を想定しています。Wayland環境の方は読み替えてください
- 記事に載せる顔画像は記事用に用意したものです
:::

## 環境
:::details
```bash
$ inxi -SG --filter
System:
  Kernel: 6.8.0-51-generic x86_64 bits: 64 Desktop: GNOME 42.9
    Distro: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
Graphics:
  Device-1: NVIDIA TU116 [GeForce GTX 1660 Ti] driver: nvidia v: 555.42.06
  Display: x11 server: X.Org v: 1.21.1.4 driver: X: loaded: nouveau
    unloaded: fbdev,modesetting,vesa failed: nvidia gpu: nvidia
    resolution: 2560x1440~60Hz
  OpenGL: renderer: NVIDIA GeForce GTX 1660 Ti/PCIe/SSE2
    v: 4.6.0 NVIDIA 555.42.06
```
:::

## FACE01の準備
:::details
### docker pull
まず最初にFACE01が実行できるよう準備をします。
以下のようにDockerイメージをプルしてください。

```bash
docker pull tokaikaoninsho/face01_gpu
```
### Xサーバーのローカルホスト接続許可
```bash
xhost +local:
```
### コンテナ起動してアタッチ
```bash
# 例:
docker run -it \
    --gpus all -e DISPLAY=$DISPLAY --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v <path/to/your/dir>:/home/docker/test \
    <Image ID>
```
### Python仮想環境をアクティベート
```bash
. bin/activate
```
![スクショ](assets/スクショ.png)
:::

## make_find_and_move_same_faces.py
まず、顔データセットに重要なことは、それぞれのIDごとに多様性が必要であることです。たとえばインタビュー映像を入力してそこからその人の顔データセットを作成できますが、その場合は同一光源であったり同一のレンズであったりするため、多様性が乏しいと言えます。個人的な経験上、このようなデータセットで学習されたモデルは汎用性が著しく劣ります。しかしデータセットを拡充する上でこの方法は避けて通れません。そこでFACE01にはどれくらいに通った顔がデータセットにあるかを判断し、同じような顔であれば別フォルダに移動するスクリプトがあります。→ make_find_and_move_same_faces.py

## make_noKnown_for_subdir_noGUI.py
顔画像はそのままでは処理できません。顔画像一つ一つを512次元ベクトルデータに変換し、任意の順序で格納する仕組みが必要です。
まず考えられるのがデータベースの活用ですが、pythonコードから読み込む場合、numpy配列であるのならばnpzとしてバイナリデータとして保存するのが簡便であり、読み込みが速いです。
このコードはそのフォルダ（ID・クラス）に存在するすべての顔画像に対して512次元ベクトルデータ（テンソル？）に変換し、バイナリデータとしてnpzとして固めます。
これにより他のユーティリティが計算処理することを可能にします。

## aligned_crop_face.py
顔を検出し、任意のパディングで顔画像を切り抜きます。このとき顔の方向を整えます。

## display_GUI_window_JAPANESE_FACE_V1
実際に入力映像を確認しながら切り抜くべき顔を検出して表示します。また、任意のフレーム間隔で顔画像を切り抜きます。このとき、任意の大きさを指定できます。もし入力映像を確認する必要がなければsimple.pyにて同じことが出来ます。こちらは映像を表示する機能がない分、高速に動作します。

## faiss_combination_similarity.py
クラス数が膨大になると同じ人物が別クラスに紛れることがあります。これを目視で確認するのは不可能です。また組み合わせ総数が膨大になるためすべての計算を順列に行えば現実的な時間で処理できなくなります。
このスクリプトではfaissを用いて組み合わせ爆発を防ぎ、非常に短い時間で、すべてのクラスに対して類似度を計算し、任意の類似度以上の組み合わせを出力します。
![](assets/2025-01-13-18-07-25.png)
![](assets/2025-01-13-18-08-00.png)
![](assets/2025-01-13-18-08-38.png)
高橋一生_h8NU.jpg.png.png_0_align_resize.png	中村倫也_ZZ1P.jpg.png.png_0_align_resize.png	0.971019446849823


## distort_barrel.py
[Illustration of transforms](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py)
例えば上記のtorchvisionではデータ拡張のための色々な機能がありますが、レンズにまつわるデータ拡張機能はありません。
このスクリプトを使用すると様々なレンズ歪みをシミュレートすることができます。それによって学習データの汎用性を向上させます。

## data_augmentation.py
樽型歪みに加えてjitterも加えたデータ拡張を提供します。マルチプロセス版のdata_augmentation_mp.pyもあります。


## 顔データの取捨選択
### 年齢

### 修正

### そら似

## リンク
- [Docker Hubリポジトリ](https://hub.docker.com/r/tokaikaoninsho/face01_gpu)
- [FACE01リポジトリ](https://github.com/yKesamaru/FACE01_DEV)