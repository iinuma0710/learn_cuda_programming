# 1, Introduction to CUDA Programming
2007年の最初のリリース以来，**Compute Unified Device Architecture (CUDA)** は，
**Graphic Computing Units (GPUs)** を一般の計算，すなわちグラフィック以外の用途に用いる場合の
デ・ファクト・スタンダードとして使われるようになってきた．
それでは，何を以て CUDA というのだろうか？
ある人はこんなことを聞いてくるだろう．

- それはプログラミング言語であるか？
- それはコンパイラなのか？
- それは新しいコンピューティング・パラダイムなのか？

この章では，GPU と CUDA のいくつかの「神話」を解き明かしていく．
**High-Performance Computing (HPC)** の歴史を知り，かつて(そして今も)半導体業界を動かし，それゆえプロセッサの
アーキテクチャそのものとも言える，ムーアの法則やデナード・スケーリングのような法則によってそれを確認することで，
ヘテロジニアス・コンピューティングの基礎を学ぶ．
また，CUDA のプログラミングモデルを紹介し，CPU と GPU の根本的な違いも学ぶ．
この章の最後には，C 言語による CUDA プログラミングで ”Hello World” を表示できるようになっていることだろう．

この章では次の内容を扱う．

- [The history of high-performance computing](./01_The_history_of_high-performance_computing.md)
- [Hello World from CUDA](./03_Hello_World_from_CUDA.md)
- [Vector addition using CUDA](./04_Vector_addition_using_CUDA.md)
- [Error reporting with CUDA](./05_Error_reporting_in_CUDA.md)
- [Data type support in CUDA](./06_Data_type_support_in_CUDA.md)

## 技術要件
この章で扱う内容は，次の条件を満たすことを前提とする．

- OS : Linux or Windows
- GPU : Pascal アーキテクチャ以降の NVIDIA GPU
- CUDA : CUDA 10.0 以降

GPU のアーキテクチャが不明な場合は，[NVIDIA のサイト](https://developer.nvidia.com/cuda-gpus)から確認できる．

この章のサンプルコードは CUDA 10.1 で開発・検証を行っている．