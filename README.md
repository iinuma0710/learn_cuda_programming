# Learn CUDA Programming
Learn CUDA Programming を読んで CUDA の勉強をする．
購入はここから([Amazon](https://www.amazon.co.jp/gp/product/B07PTBDWV4?pf_rd_r=TQ981MVTRRJCW6PJX971&pf_rd_p=7392bae8-7129-4d1a-96a9-1cfe0aa13ab3))

## ハードウェア情報
- CPU : Intel Core-i3 10100T
- MEM : DDR4-2666 8GB x 2
- GPU : NVIDIA GeForce GT1030 (384 CUDA Cores)
- PSU : TFX 300W 電源 (ケース付属品)

## ドライバのインストール
PPA を追加し推奨ドライバを確認する．
```bash
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:03.1/0000:0b:00.0 ==
modalias : pci:v000010DEd00001B06sv000010DEsd0000120Fbc03sc00i00
vendor   : NVIDIA Corporation
model    : GP102 [GeForce GTX 1080 Ti]
driver   : nvidia-driver-390 - distro non-free
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-440 - distro non-free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin
```

ここでは，バージョン 440 が推奨されているので，これをインストールする．
```bash
$ sudo apt install nvidia-driver-440
```

## CUDA のインストール
色々ファイルをいじらなくても，一発で入るようになった．
2020年8月26日現在，CUDA 10.1 が入る．
```bash
$ sudo apt install nvidia-cuda-toolkit
```

## サンプルコードのダウンロード
[Packt のサポートページ](https://www.packtpub.com/support)からユーザ登録すると，
[Packt のホームページ](https://www.packtpub.com)からサンプルコードをダウンロードできる．

1. [Packt のホームページ](https://www.packtpub.com)にログイン
2. Support タブを選択
3. Code Download をクリック
4. 検索ボックスにこの本のタイトルを入力し画面の支持に従う

ダウンロードした zip ファイルを 7-Zip などで解凍する．  
また，[GitHub のリポジトリ](https://github.com/PacktPublishing/Learn-CUDA-Programming)からもダウンロードできる．
コードに更新があれば，こちらを更新していく．
他にも Packt の書籍やビデオの情報が載っている．

## カラー図版の配布
[ここから](https://static.packt-cdn.com/downloads/9781788996242_ColorImages.pdf)カラー図版のみを集めた
PDFをダウンロードできる．