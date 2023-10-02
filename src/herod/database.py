import os

import lmdb
import blake3


class Lmdb:
    _env: dict[str, lmdb.Environment] = {}

    def __init__(self, collection: str):
        if Lmdb._env.get(collection) is None:
            Lmdb._env[collection] = lmdb.open(
                f"./lmdb/{collection}.mdb", map_size=1024**3, subdir=False
            )
        self.env = Lmdb._env[collection]

    def record_image_id(self, image_id: int, filename: str):
        """记录并返回图片的 ID，如果图片已经存在则返回 None"""
        with self.env.begin(write=True) as txn:
            digest = image_id.to_bytes(5, "big")
            txn.put(digest[:5], filename.encode())

    def get_image_by_id(self, image_id: int) -> bytes | None:
        with self.env.begin() as txn:
            return txn.get(image_id.to_bytes(5, "big"))

    def delete(self):
        """删除集合"""
        os.remove(self.env.path())
        del Lmdb._env[self.env.path()]


def get_image_hash(file: str | bytes) -> int:
    if isinstance(file, str):
        with open(file, "rb") as f:
            data = f.read()
            digest = blake3.blake3(data).digest()
            return int.from_bytes(digest[:5], "big")
    else:
        digest = blake3.blake3(file).digest()
        return int.from_bytes(digest[:5], "big")
