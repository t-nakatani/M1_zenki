# （課題）行優先行列乗算のAVX並列化 (Assignment) Parallelizing Row-Major Matrix Multiplication with AVX

* 行優先の1次元配列で表現された行列（要素はfloat型）A, B, Cに対して、C = ABをAVX組み込み関数を用いて並列計算するプログラム


* コンパイル $ ```gcc -O3 -march=native matmulAVX-e.cpp swatch.cpp```