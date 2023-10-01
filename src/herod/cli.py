import typer
import cv2

from pathlib import Path
from herod.feature import FeatureExtractor
from herod import helpers, server
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    Collection,
)

app = typer.Typer()
connections.connect(host="localhost", port="19530")


@app.command()
def show_feature(filename: str, count: int = 500):
    """展示图片的特征点提取结果"""
    extractor = FeatureExtractor("SURF")

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    keys = extractor.detect(img, count)

    img = cv2.drawKeypoints(img, keys, None)
    cv2.imshow("show", img)
    cv2.waitKey(0)


@app.command()
def create_collection(
    name: str,
    force: bool = False,
    description: str = "",
    partition: bool = False,
):
    """建立一个集合"""
    if utility.has_collection(name):
        if not force:
            typer.echo(f"集合 {name} 已存在，如需覆盖请使用 --force 参数")
            raise typer.Exit(1)
        else:
            utility.drop_collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image", dtype=DataType.INT64, description="图片 ID"),
        FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64, description="图片特征向量"
        ),
    ]
    if partition:
        fields.append(
            FieldSchema(
                name="partition",
                dtype=DataType.VARCHAR,
                description="额外分区字段",
                is_partition=True,
            )
        )
    schema = CollectionSchema(fields=fields, description=description)
    Collection(name=name, schema=schema)


@app.command()
def drop_collection(name: str):
    """删除一个集合"""
    utility.drop_collection(name)


@app.command()
def create_index(collection: str, index_type: str = "DISKANN"):
    """为集合建立索引"""
    collection = Collection(name=collection)
    index_params = {
        "metric_type": "L2",
        "index_type": index_type,
    }
    collection.create_index(field_name="embedding", index_params=index_params)


@app.command()
def drop_index(collection: str):
    """删除集合的索引"""
    collection = Collection(name=collection)
    collection.drop_index()


@app.command()
def add_image(collection: str, path: str, count: int = 500, partition: str = None):
    """往集合中增加一张图片或递归添加一个文件夹中的图片"""
    path = Path(path)
    if path.is_dir():
        for file in path.rglob("**/*.*"):
            try:
                helpers.add_image(collection, str(file), count, partition)
                typer.echo(f"处理 {file} 完成")
            except Exception as e:
                typer.echo(f"处理 {file} 时出现错误：{e}")
    else:
        helpers.add_image(collection, str(path), count, partition)
        typer.echo(f"处理 {path} 完成")


@app.command()
def search_image(
    collection: str, filename: str, search_list: int = 16, limit: int = 100
):
    """在集合中搜索一张图片"""
    result = helpers.search_image(collection, filename, search_list, limit)
    for filename, image_id, distance in result:
        typer.echo(f"{filename} {image_id} {distance}")


@app.command()
def start_server():
    """启动服务器"""
    server.start_server()


def main():
    app()
