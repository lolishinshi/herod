import lmdb
import blake3

# 最大 1TiB
env = lmdb.open("herod.mdb", map_size=1024**3, subdir=False)


def record_image_id(filename: str) -> int:
    with open(filename, "rb") as f:
        data = f.read()
        digest = blake3.blake3(data).digest()
    with env.begin(write=True) as txn:
        txn.put(digest[:5], filename.encode())
    return int.from_bytes(digest[:5], "big")


def get_image_by_id(image_id: int) -> bytes:
    with env.begin() as txn:
        return txn.get(image_id.to_bytes(5, "big"))
