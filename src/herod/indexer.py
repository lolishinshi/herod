import cv2
import numpy as np
import typer

from herod.database import Lmdb, get_image_hash
from herod.feature import FeatureExtractor, Extractor, Filter
from pymilvus import Collection
from collections import defaultdict
from datetime import datetime


# https://www.jianshu.com/p/4d2b45918958
def wilson_score(values: list[float], p_z: float = 2.326):
    values = 1 - np.array(values)
    mean = np.mean(values)
    var = np.var(values)
    total = len(values)

    score = (
        mean
        + (np.square(p_z) / (2.0 * total))
        - ((p_z / (2.0 * total)) * np.sqrt(4.0 * total * var + np.square(p_z)))
    ) / (1 + np.square(p_z) / total)
    return score


class Indexer:
    def __init__(
        self,
        collection: str,
        search: bool = False,
        extractor: Extractor = Extractor.SURF,
        filter: Filter = Filter.FUFP,
    ):
        self.collection = Collection(name=collection)
        if search:
            typer.echo(f"正在加载集合 {collection} 的索引")
            self.collection.load()
        self.extractor = FeatureExtractor(extractor, filter)
        self.mdb = Lmdb(collection)

    def add_image(self, filename: str, limit: int = 500):
        """
        往集合中增加一张图片
        :param filename: 文件名
        :param limit: 特征点数量
        :return:
        """
        image_id = get_image_hash(filename)
        if self.mdb.get_image_by_id(image_id) is None:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            kps, des = self.extractor.detect_and_compute(img, limit)
            # 可能会有空白图片，没有特征点
            if not kps:
                print(f"图片 {filename} 没有特征点")
                return
            data = [[image_id] * len(des), des]
            self.collection.insert(data)
        self.mdb.record_image_id(image_id, filename)

    def add_image_raw(self, data: bytes, name: str, limit: int = 500):
        """
        往集合中增加一张图片
        :param data: 图片数据
        :param name: 文件名
        :param limit: 特征点数量
        :return:
        """
        image_id = get_image_hash(data)
        if self.mdb.get_image_by_id(image_id) is None:
            img = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
            kps, des = self.extractor.detect_and_compute(img, limit)
            # 可能会有空白图片，没有特征点
            if not kps:
                print(f"图片 {name} 没有特征点")
                return
            data = [[image_id] * len(des), des]
            self.collection.insert(data)
        self.mdb.record_image_id(image_id, name)

    def search_image(
        self,
        image: str | cv2.typing.MatLike,
        search_list: int = 16,
        search_limit: int = 100,
        limit: int = 100,
    ) -> tuple[float, list[tuple[str, int, float]]]:
        """
        在集合中搜索图片
        :param image: 图片
        :param search_list: 搜索列表大小，越大越准确，但是速度越慢
        :param search_limit: 被搜索图片的采样点数量
        :param limit: 返回结果数量
        :return:
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            img = image
        _, des = self.extractor.detect_and_compute(img, search_limit)

        now = datetime.now()
        results = self.collection.search(
            data=des,
            anns_field="embedding",
            param={"search_list": search_list},
            limit=limit,
            output_fields=["image"],
        )
        elapsed = (datetime.now() - now).total_seconds()

        d = defaultdict(list)

        for result in results:
            for image in result:
                d[image.entity.get("image")].append(image.distance)

        d = [
            (
                self.mdb.get_image_by_id(mid).decode(),
                wilson_score(distances),
            )
            for mid, distances in d.items()
        ]
        d.sort(key=lambda x: x[1], reverse=True)

        return elapsed, d

    # def __del__(self):
    #     self.collection.release()
