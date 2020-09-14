# 3 CUDA Thread Programming
CUDA には，グループ内の CUDA スレッドを制御できるように，階層的なスレッドアーキテクチャを採用している．
スレッドがどのように並列的に動作するのか理解しておくことは，並列プログラムを組んだり，
パフォーマンスを向上させたりするのに役立つ．
この章では，CUDA スレッドの操作と GPU の計算資源との関係性について扱う．
実践的な実験として，並列的なデータ削減アルゴリズムを調査し，
最適化手法を用いて CUDA のコードを最適化する方法を見ていく．

本章では，スレッドの並列・同時実行やワープの実行，メモリ帯域の問題，オーバーヘッドの制御，SIMD 操作など，
GPU での CUDA スレッドの操作の方法を学ぶ．

本性の内容：

- Hierarchical CUDA thread operations
- Understanding CUDA occupancy
- Data sharing across multiple CUDA threads
- Identifying an application's performance limiter
- Minimize the CUDA warp divergence effect
- Increasing memory utilization and grid-stride loops
- Cooperative Groups for flexible thread handling
- Warp synchronous programming
- Low-/mixed-precision operations

## Technical requirements
本章で使用する NVIDIA GPU は Pascal アーキテクチャ以降を推奨している．
つまり，computer capability は60以降でなければならない．
GPU のアーキテクチャが不明な場合は，[NVIDIA のサイト](https://developer.nvidia.com/cuda-gpus)から確認できる．

この章のサンプルコードは CUDA 10.1 で開発・検証を行っている．

この章では，コードのプロファイリングによって CUDA プログラミングを実行する．
GPU アーキテクチャが Turing ならば，プロファイリングのために [Nsight Compute](https://developer.nvidia.com/nsight-compute) をインストールしておくことをおすすめする．
本書執筆中には，プロファイラが移行段階であった．
Nsight Compute を用いたカーネルのプロファイリングの基本的な使い方は第5章で扱う．