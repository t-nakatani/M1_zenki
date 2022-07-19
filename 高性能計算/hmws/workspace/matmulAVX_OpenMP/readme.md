# （課題）行優先行列乗算のOpenMPとAVXを用いた並列化 (Assignment) Parallelizing Row-Major Matrix Multiplication with OpenMP and AVX

* OpenMPのみを用いる並列計算用関数

* OpenMPとAVX組み込み関数の両方を用いる並列計算用関数

コンパイル $ ```g++ -fopenmp -O3 -march=native matmulAVX-j_withOpenMP.cpp swatch.cpp```

