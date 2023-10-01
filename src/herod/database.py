import lmdb
import blake3

# 最大 1TiB
env = lmdb.open("herod.mdb", map_size=1024**3, subdir=False)


def get_image_hash(filename: str) -> int:
    with open(filename, "rb") as f:
        data = f.read()
        digest = blake3.blake3(data).digest()
        return int.from_bytes(digest[:5], "big")


def record_image_id(image_id: int, filename: str):
    """记录并返回图片的 ID，如果图片已经存在则返回 None"""
    with env.begin(write=True) as txn:
        digest = image_id.to_bytes(5, "big")
        txn.put(digest[:5], filename.encode())


def get_image_by_id(image_id: int) -> bytes | None:
    with env.begin() as txn:
        return txn.get(image_id.to_bytes(5, "big"))
