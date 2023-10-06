import typer
import cv2
from typing_extensions import Annotated
from pathlib import Path

from herod.database import Lmdb
from herod.feature import FeatureExtractor, Filter, Extractor
from herod.indexer import Indexer
from herod.config import config
from herod import server
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    Collection,
)

app = typer.Typer()
connections.connect(host=config.milvus.host, port=config.milvus.port)


@app.command()
def show_feature(
    filename: Annotated[str, typer.Argument(help="图片路径")],
    limit: Annotated[int, typer.Option(help="限制特征点数量，如果为 0 表示不限制")] = 500,
    brief: Annotated[bool, typer.Option(help="使用简洁版的绘图")] = True,
    extractor: Annotated[Extractor, typer.Option(help="特征点提取算法")] = Extractor.SURF,
    filter: Annotated[Filter, typer.Option(help="特征点均匀化算法")] = Filter.FUFP,
):
    """展示图片的特征点提取结果"""
    extractor = FeatureExtractor(extractor, filter)

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    keys = extractor.detect(img, limit if limit > 0 else None)

    if not brief:
        img = cv2.drawKeypoints(
            img, keys, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    else:
        img = cv2.drawKeypoints(img, keys, None)
    cv2.imshow("show", img)
    while True:
        if cv2.waitKey() == -1:
            break


@app.command()
def create_collection(
    name: Annotated[str, typer.Argument(help="集合名称")],
    description: Annotated[str, typer.Option(help="集合描述")] = "",
):
    """建立一个集合"""
    if utility.has_collection(name):
        typer.echo(f"集合 {name} 已存在")
        raise typer.Exit(1)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image", dtype=DataType.INT64, description="图片 ID"),
        FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64, description="图片特征向量"
        ),
    ]
    schema = CollectionSchema(fields=fields, description=description)
    Collection(name=name, schema=schema)


@app.command()
def drop_collection(name: str):
    """删除一个集合"""
    utility.drop_collection(name)
    Lmdb(name).delete()


@app.command()
def release_collection(name: str):
    """释放一个集合的资源"""
    Collection(name=name).release()


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
def add_image(
    collection: str,
    path: str,
    limit: int = 500,
    extractor: Annotated[Extractor, typer.Option(help="特征点提取算法")] = Extractor.SURF,
    filter: Annotated[Filter, typer.Option(help="特征点均匀化算法")] = Filter.FUFP,
    glob: str = "**/*.*",
):
    """往集合中增加一张图片或递归添加一个文件夹中的图片"""
    path = Path(path)
    indexer = Indexer(collection, extractor=extractor, filter=filter)
    if path.is_dir():
        for file in path.rglob(glob):
            try:
                indexer.add_image(str(file), limit)
                typer.echo(f"处理 {file} 完成")
            except Exception as e:
                typer.echo(f"处理 {file} 时出现错误：{e}")
    else:
        indexer.add_image(str(path), limit)
        typer.echo(f"处理 {path} 完成")


@app.command()
def search_image(
    collection: Annotated[str, typer.Argument(help="集合名称")],
    filename: Annotated[str, typer.Argument(help="图片路径")],
    search_list: Annotated[int, typer.Option(help="搜索列表大小，越大越准确，但是速度越慢")] = 32,
    search_limit: Annotated[int, typer.Option(help="被搜索图片的采样点数量")] = 100,
    limit: Annotated[int, typer.Option(help="每个向量的匹配数量")] = 50,
):
    """在集合中搜索一张图片"""
    indexer = Indexer(collection, search=True)
    result = indexer.search_image(filename, search_list, search_limit, limit)
    for filename, image_id, distance in result[:20]:
        typer.echo(f"{filename}\t{image_id}\t{distance}")


@app.command()
def start_server(host: str = "0.0.0.0", port: int = 8080):
    """启动服务器"""
    server.start_server(host, port)


def main():
    app()
