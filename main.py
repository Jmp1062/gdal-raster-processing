import boto3
import numpy as np  # noqa: F401 -- top-level import so Burla installs numpy on workers
import rasterio  # noqa: F401 -- top-level import so Burla installs rasterio (bundles GDAL) on workers
from burla import remote_parallel_map

SRC_BUCKET = "sentinel-s2-l2a"
DST_BUCKET = "my-ndvi-outputs"

tile_ids = []
with open("sentinel_tiles.txt") as f:
    tile_ids = [line.strip() for line in f if line.strip()]

print(f"processing {len(tile_ids)} Sentinel-2 tiles")


def compute_ndvi(tile_id: str) -> dict:
    import boto3
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile

    s3 = boto3.client("s3", region_name="eu-central-1")

    def read_band(band: str) -> tuple[np.ndarray, dict]:
        key = f"tiles/{tile_id}/{band}.jp2"
        body = s3.get_object(Bucket=SRC_BUCKET, Key=key, RequestPayer="requester")["Body"].read()
        with MemoryFile(body) as mem, mem.open() as src:
            return src.read(1).astype("float32"), src.profile

    red, profile = read_band("B04")
    nir, _ = read_band("B08")

    ndvi = (nir - red) / (nir + red + 1e-6)

    profile.update(driver="GTiff", dtype="float32", count=1, compress="DEFLATE", tiled=True)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(ndvi.astype("float32"), 1)
        s3.put_object(Bucket=DST_BUCKET, Key=f"ndvi/{tile_id}.tif", Body=mem.read())

    return {"tile_id": tile_id, "mean_ndvi": float(ndvi.mean()), "pixels": int(ndvi.size)}


# 2,000 tiles -> 2,000 workers running in parallel, each with 2 CPUs and 8 GB RAM
results = remote_parallel_map(compute_ndvi, tile_ids, func_cpu=2, func_ram=8)

import pandas as pd
pd.DataFrame(results).to_csv("ndvi_report.csv", index=False)
