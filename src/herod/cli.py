import typer
import cv2

from pathlib import Path
from herod.database import record_image_id
from herod.feature import FeatureExtractor
from herod import helpers
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
    schema = CollectionSchema(fields=fields, description=description)
    collection = Collection(name=name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "DISKANN",
    }
    collection.create_index(field_name="embedding", index_params=index_params)


@app.command()
def add_image(collection: str, path: str):
    """往集合中增加一张图片或递归添加一个文件夹中的图片"""
    path = Path(path)
    if path.is_dir():
        for file in path.rglob("**/*.*"):
            try:
                helpers.add_image(collection, str(file))
                typer.echo(f"处理 {file} 完成")
            except Exception as e:
                typer.echo(f"处理 {file} 时出现错误：{e}")
    else:
        helpers.add_image(collection, str(path))
        typer.echo(f"处理 {path} 完成")


@app.command()
def search_image(collection: str, filename: str):
    """在集合中搜索一张图片"""
    helpers.search_image(collection, filename)


def main():
    app()
