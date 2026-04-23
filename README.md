# Run GDAL Raster Processing Across 1,000+ CPUs in Python

Process Sentinel-2, Landsat, or NAIP tiles at the same time on thousands of cloud machines. One tile per worker. One function call.

## Try it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Burla-Cloud/gdal-raster-processing/blob/main/Burla_GDALRaster_Demo.ipynb)

Follow along in a notebook - compute NDVI on 6 public Sentinel-2 tiles across 6 cloud workers in about 4 minutes. No prior Burla knowledge needed.

## The Problem

You have 2,000 Sentinel-2 tiles on S3 and you want to compute NDVI, reproject, or clip them. A single-core `gdalwarp` or `rasterio` loop takes days. GDAL is hard to containerize cleanly - native deps, PROJ versions, GDAL_DATA paths. Running it on 1,000 EC2 instances yourself means AMIs, user-data scripts, and a queue.

Most "big data" tools don't help. Dask arrays assume one logical grid. Spark doesn't speak GeoTIFF. AWS Batch works but takes a day to wire up.

## The Solution (Burla)

Write a normal function that processes one tile with `rasterio` (or shells out to `gdalwarp`). Hand Burla the list of tile IDs. It runs the function on 2,000 workers at the same time, with GDAL/PROJ already installed.

Local-first, no cluster to manage. Same code on your laptop and on 2,000 CPUs.

## Example

```python
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


# 2,000 tiles -> Burla grows the cluster on demand and processes them in parallel
results = remote_parallel_map(compute_ndvi, tile_ids, func_cpu=2, func_ram=8, grow=True)

import pandas as pd
pd.DataFrame(results).to_csv("ndvi_report.csv", index=False)
```

## Why This Is Better

**vs Ray** - no head node, no actors, no GDAL-in-Ray headaches. You don't install GDAL on every node yourself.

**vs Dask** - `dask.array` assumes a contiguous grid. Per-tile GDAL operations don't fit that model without `delayed`, and `delayed` plus a 2,000-task graph is slow to schedule.

**vs AWS Batch** - no custom Docker image with GDAL, PROJ, and your code. No job definition, no compute environment, no IAM dance. Burla starts tiles in seconds; Batch takes minutes to cold start.

## How It Works

You pass a list of tile IDs and a function that processes one tile. Burla runs `compute_ndvi(tile_id)` on 2,000 cloud workers in parallel. Each worker has GDAL/rasterio available. Results come back as a list.

## When To Use This

- Per-tile processing: NDVI, NDWI, cloud masking, reprojection, cropping.
- Batch COG conversion for thousands of GeoTIFFs.
- Re-running an analysis over every tile in a continent.
- Chipping large scenes into ML training patches.

