import math
import typing

import cv2


class FeatureExtractor:
    def __init__(self, name: str):
        if name == "SURF":
            # TODO: upRight 设置为 True，忽略方向来获得更快的计算速度？
            self.extractor = cv2.xfeatures2d.SURF.create(hessianThreshold=500)
        elif name == "SIFT":
            self.extractor = cv2.SIFT.create()
        else:
            raise ValueError(f"Unknown feature extractor: {name}")

    def detect(
        self, img: cv2.typing.MatLike, count: int | None = None
    ) -> typing.Sequence[cv2.KeyPoint]:
        if count is None:
            return self.extractor.detect(img)
        else:
            keys = self.extractor.detect(img)
            return fufp_extract(keys, img.shape[0], img.shape[1], count)

    def compute(
        self, img: cv2.typing.MatLike, kp: typing.Sequence[cv2.KeyPoint]
    ) -> tuple[typing.Sequence[cv2.KeyPoint], cv2.typing.MatLike]:
        return self.extractor.compute(img, kp)

    def detect_and_compute(
        self, img: cv2.typing.MatLike, count: int | None = None, resize: bool = True
    ) -> tuple[typing.Sequence[cv2.KeyPoint], cv2.typing.MatLike]:
        # TODO: 对于超长图片，缩放可能会严重影响分辨率
        if resize:
            img = adjust_image_size(img)
        keys = self.detect(img, count)
        return self.compute(img, keys)


def fufp_extract(
    keys: typing.Sequence[cv2.KeyPoint], height: int, width: int, count: int
) -> list[cv2.KeyPoint]:
    """
    使用 FUFP 算法均匀地从图像出提取出若干个特征点
    最终结果不保证刚好为 count 个

    参考：宋霄罡，张元培，梁莉,等.面向视觉SLAM的快速均匀特征点提取方法[J]. 导航定位与授时, 2022, 9(4): 41-50.

    :param keys: 候选特征点
    :param height: 图像高度
    :param width: 图像宽度
    :param count: 需要提取的特征点个数
    :return: 返回提取出的特征点
    """

    # 计算垂直和水平方向上的网格划分数量
    # 此处进行双倍划分，后续再进行合并
    y_num = round(math.sqrt(height / width * count)) * 2
    x_num = round(math.sqrt(width / height * count)) * 2

    # 将特征点放入网格中
    boxes: list[list[list[cv2.KeyPoint]]] = [
        [[] for _ in range(x_num)] for _ in range(y_num)
    ]

    for key in keys:
        x, y = key.pt
        x = math.floor(x / (width / x_num))
        y = math.floor(y / (height / y_num))
        boxes[y][x].append(key)

    # 每四个一组合并网格，如果遇到四个空网格，则下一组不进行合并
    result = []
    empty_box = 0
    for y in range(0, y_num - 1, 2):
        for x in range(0, x_num - 1, 2):
            if empty_box == 0:
                _box = []
                _box.extend(boxes[y][x])
                _box.extend(boxes[y][x + 1])
                _box.extend(boxes[y + 1][x])
                _box.extend(boxes[y + 1][x + 1])
                if len(_box) == 0:
                    empty_box += 1
                else:
                    result.append(max(_box, key=lambda k: k.response))
            else:
                empty_box -= 1
                if boxes[y][x]:
                    result.append(max(boxes[y][x], key=lambda k: k.response))
                if boxes[y][x + 1]:
                    result.append(max(boxes[y][x + 1], key=lambda k: k.response))
                if boxes[y + 1][x]:
                    result.append(max(boxes[y + 1][x], key=lambda k: k.response))
                if boxes[y + 1][x + 1]:
                    result.append(max(boxes[y + 1][x + 1], key=lambda k: k.response))

    return result


def adjust_image_size(img: cv2.typing.MatLike, width: int = 1920, height: int = 1080):
    if img.shape[0] > height or img.shape[1] > width:
        scale = min(height / img.shape[0], width / img.shape[1])
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return img
