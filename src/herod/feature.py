import math
import typing
from enum import Enum
import cv2


class Extractor(str, Enum):
    SURF = "SURF"
    SIFT = "SIFT"


class Filter(str, Enum):
    FUFP = "FUFP"
    QUAD = "QUAD"


class FeatureExtractor:
    def __init__(self, name: Extractor = Extractor.SURF, filter: Filter = Filter.QUAD):
        match name:
            case Extractor.SURF:
                # TODO: upRight 设置为 True，忽略方向来获得更快的计算速度？
                self.extractor = cv2.xfeatures2d.SURF.create(hessianThreshold=500)
            case Extractor.SIFT:
                self.extractor = cv2.SIFT.create()

        match filter:
            case Filter.FUFP:
                self.filter = fufp_extract
            case Filter.QUAD:
                self.filter = quad_filter

    def detect(
        self, img: cv2.typing.MatLike, count: int | None = None
    ) -> typing.Sequence[cv2.KeyPoint]:
        if count is None:
            return self.extractor.detect(img)
        else:
            keys = self.extractor.detect(img)
            return self.filter(keys, img.shape[0], img.shape[1], count)

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


class QuadNode:
    def __init__(
        self, keys: list[cv2.KeyPoint], x: float, y: float, width: float, height: float
    ):
        self.keys = keys
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def split(self) -> list["QuadNode"]:
        dx = self.x + self.width / 2
        dy = self.y + self.height / 2
        ul, ur, dl, dr = [], [], [], []

        for key in self.keys:
            if key.pt[0] < dx:
                if key.pt[1] < dy:
                    ul.append(key)
                else:
                    dl.append(key)
            else:
                if key.pt[1] < dy:
                    ur.append(key)
                else:
                    dr.append(key)

        result = [
            QuadNode(ul, self.x, self.y, self.width / 2, self.height / 2),
            QuadNode(ur, dx, self.y, self.width / 2, self.height / 2),
            QuadNode(dl, self.x, dy, self.width / 2, self.height / 2),
            QuadNode(dr, dx, dy, self.width / 2, self.height / 2),
        ]
        return [node for node in result if node.keys]

    @property
    def can_split(self):
        return len(self.keys) > 1


def quad_filter(
    keys: typing.Sequence[cv2.KeyPoint], height: int, width: int, limit: int
) -> list[cv2.KeyPoint]:
    """
    使用四叉树均匀地从图像出提取出若干个特征点
    :param keys: 候选特征点
    :param height: 图像高度
    :param width: 图像宽度
    :param limit: 需要提取的特征点个数
    :return:
    """
    root = QuadNode(list(keys), 0, 0, width, height)
    nodes = root.split()
    end = False
    while not end:
        nodes.sort(key=lambda n: len(n.keys), reverse=True)
        tmp = []
        no_split = True
        while nodes:
            node = nodes.pop()
            if node.can_split:
                tmp.extend(node.split())
                no_split = False
            else:
                tmp.append(node)
            if len(tmp) + len(nodes) >= limit:
                end = True
                break
        if no_split:
            end = True
        nodes.extend(tmp)

    return [max(node.keys, key=lambda key: key.response) for node in nodes]
