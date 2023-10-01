# herod

## 安装

herod 使用 milvus 作为后端，需要安装 milvus。此处使用 docker 部署 milvus 单机实例

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.3.1/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker-compose up -d
```

herod 依赖 opencv 中的 SURF 特征提取算法，这个算法在新版 opencv 中默认不包含，需要手动编译。
以下以 pdm 为例展示在虚拟环境中编译 opencv 的方法。

```bash
CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" PDM_NO_BINARY=opencv-contrib-python pdm sync
```

执行完上述命令后，可以使用 `. .venv/bin/activate` 激活虚拟环境，此时可以执行 `herod --version` 确认是否正常工作

## 使用


