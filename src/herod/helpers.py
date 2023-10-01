import cv2
from herod.database import record_image_id, get_image_by_id, get_image_hash
from herod.feature import FeatureExtractor
from pymilvus import Collection
from collections import defaultdict
from datetime import datetime

extractor = FeatureExtractor("SURF")


def add_image(collection: str, filename: str, count: int = 500, partition: str = None):
    """
    往集合中增加一张图片
    :param collection: 集合名称
    :param filename: 文件名
    :param count: 特征点数量
    :param partition: 分区名称
    :return:
    """
    image_id = get_image_hash(filename)
    if get_image_by_id(image_id) is None:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        _, des = extractor.detect_and_compute(img, count)
        collection = Collection(name=collection)
        data = [[image_id] * len(des), des]
        if partition:
            data.append([partition] * len(des))
        collection.insert(data)
    record_image_id(image_id, filename)


def search_image(
    collection: str,
    image: str | cv2.typing.MatLike,
    search_list: int = 16,
    limit: int = 100,
) -> list[tuple[str, int, float]]:
    """
    在集合中搜索图片
    :param collection: 集合名称
    :param image: 图片
    :param search_list: 搜索列表大小，越大越准确，但是速度越慢
    :param limit: 返回结果数量
    :return:
    """
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image
    _, des = extractor.detect_and_compute(img, 500)

    collection = Collection(name=collection)
    collection.load()
    now = datetime.now()
    results = collection.search(
        data=des,
        anns_field="embedding",
        param={"search_list": search_list},
        limit=limit,
        output_fields=["image"],
    )
    print(f"搜索耗时：{(datetime.now() - now).total_seconds()}s")

    d = defaultdict(list)

    for result in results:
        for image in result:
            d[image.entity.get("image")].append(image.distance)

    d = sorted(d.items(), key=lambda x: len(x[1]), reverse=True)
    d = [
        (get_image_by_id(i[0]).decode(), len(i[1]), sum(i[1]) / len(i[1]))
        for i in d[:20]
    ]

    return d
