import cv2
from herod.database import record_image_id, get_image_by_id
from herod.feature import FeatureExtractor
from pymilvus import Collection
from collections import defaultdict

extractor = FeatureExtractor("SURF")


def add_image(collection: str, filename: str):
    """往集合中增加一张图片"""
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, des = extractor.detect_and_compute(img, 500)
    image_id = record_image_id(filename)
    collection = Collection(name=collection)
    collection.insert(
        [
            [image_id] * len(des),
            des,
        ]
    )


def search_image(
    collection: str, filename: str, search_list: int = 16, limit: int = 100
):
    """
    在集合中搜索图片
    :param collection: 集合名称
    :param filename: 文件名
    :param search_list: 搜索列表大小，越大越准确，但是速度越慢
    :param limit: 返回结果数量
    :return:
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, des = extractor.detect_and_compute(img, 500)

    collection = Collection(name=collection)
    collection.load()
    results = collection.search(
        data=des,
        anns_field="embedding",
        param={"search_list": search_list},
        limit=100,
        output_fields=["image"],
    )

    d = defaultdict(int)

    for result in results:
        for image in result:
            d[image.entity.get("image")] += 1

    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    for i in d[:10]:
        print(get_image_by_id(i[0]), i[1])
