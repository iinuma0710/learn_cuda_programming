# 1.5 Error reporting in CUDA
CUDA ではホスト側のコードでエラーの管理を行う．
ほとんどの CUDA 関数は列挙型の ```cudaError_t``` を呼び出す．
```cudaSuccess``` は値が0で，エラーがないことを表す．
また，```cudaGetErrorString()``` 関数を用いると，次のように文字列でエラー状態の説明を取得できる．

```c
cudaError_t e;
e = cudaMemcpy(...);
if (e) {
    printf("Error : %s\n", cudaGetErrorString(e));
}
```

カーネルの立ち上げは返り値を持たない．
そこで，カーネルの立ち上げを含む最後の CUDA 関数のエラーコードを返す，```cudaGetLastError()``` のような関数を用いる．
複数のエラーがある場合には，最後の1つだけが取得される．

```c
MyKernel<<<...>>>(...);
cudaDeviceSynchronize();
e = cudaGetLastError();
```

プロダクション・コードでは，もし GPU カーネルがクラッシュしたり，不正な結果が得られたときに，CPU コードの通常実行を
継続するかを判定する論理的なチェックポイントで，エラーチェックを行うとよいだろう．

次節では，CUDA プログラミングモデルでサポートされているデータ型を紹介する．