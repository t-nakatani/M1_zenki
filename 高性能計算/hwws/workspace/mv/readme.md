# 2022前 高性能計算論 【火4】

## dockerにて作業

<!-- * image作成: ```docker build -t linux-prog:test .``` -->

* container作成: ```docker run --gpus all -it -v <src_dir>:/work nvidia/cuda:11.3.0-devel-ubuntu20.04 /bin/bash```
* コンパイル: ```nvcc mv-j.cu swatch.cpp```