"""
Extract Anny body model LBS data to binary format for anny-js.

Bakes a fixed body shape from phenotype params, then exports:
  - Shaped T-pose vertices  (V×3 float32)
  - Triangle faces          (F×3 int32)
  - Skinning weights        (V×8 float32)
  - Skinning bone indices   (V×8 int32)
  - Rest bone poses         (B×4×4 float32)  world-space 4×4 transforms
  - Bone parents            (B int32)
  - Bone labels             JSON list

All packed into a single `anny_model.bin` with a `anny_model.json` manifest
describing array shapes and byte offsets.

Usage (from anny-js root, with body-tuning venv active):
    source ../clad/vton-exp/hmr/body-tuning/venv/bin/activate
    python tools/extract_anny.py [--out assets/] [--params '{"gender":0.5,...}']
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# Default: neutral average body.
DEFAULT_PARAMS = {
    "gender":       0.5,   # 0=male, 1=female
    "age":          0.3,
    "muscle":       0.4,
    "weight":       0.4,
    "height":       0.5,
    "proportions":  0.5,
    "cupsize":      0.5,
    "firmness":     0.5,
    "african":      0.0,
    "asian":        0.0,
    "caucasian":    1.0,
}


def build_blendshape_coeffs(model, params: dict):
    import torch
    # model.get_phenotype_blendshape_coefficients takes keyword scalar/tensor args.
    pheno_kwargs = {k: float(v) for k, v in params.items() if k != "_local_changes"}
    with torch.no_grad():
        coeffs = model.get_phenotype_blendshape_coefficients(**pheno_kwargs)  # (1, C)
    return coeffs


def extract(params: dict, out_dir: Path):
    import anny
    import torch

    print("Creating Anny fullbody model …", flush=True)
    model = anny.create_fullbody_model(
        all_phenotypes=True, triangulate_faces=True, local_changes=False
    ).to(dtype=torch.float32)

    print(f"  bones={model.bone_count}  verts={model.template_vertices.shape[0]}"
          f"  faces={model.faces.shape[0]}  blendshapes={model.blendshapes.shape[0]}")

    print("Computing blendshape coefficients …", flush=True)
    coeffs = build_blendshape_coeffs(model, params)  # (1, C)

    with torch.no_grad():
        # Shaped rest vertices (T-pose), shape (1, V, 3)
        rest_verts = model.get_rest_vertices(coeffs)  # (1, V, 3)
        # Shaped bone positions → rest bone poses (1, B, 4, 4)
        _, _, rest_bone_poses = model.get_rest_bone_poses(coeffs)

    rest_verts_np = rest_verts[0].cpu().numpy().astype(np.float32)   # (V, 3)
    rest_bone_poses_np = rest_bone_poses[0].cpu().numpy().astype(np.float32)  # (B, 4, 4)
    faces_np = model.faces.cpu().numpy().astype(np.int32)            # (F, 3)
    weights_np = model.vertex_bone_weights.cpu().numpy().astype(np.float32)  # (V, 8)
    indices_np = model.vertex_bone_indices.cpu().numpy().astype(np.int32)     # (V, 8)
    bone_parents = list(model.bone_parents)                          # list[int]
    bone_labels = list(model.bone_labels)                            # list[str]

    # ------------------------------------------------------------------ pack
    arrays = {
        "rest_vertices":    rest_verts_np,       # (V,3)  float32
        "faces":            faces_np,            # (F,3)  int32
        "bone_weights":     weights_np,          # (V,8)  float32
        "bone_indices":     indices_np,          # (V,8)  int32
        "rest_bone_poses":  rest_bone_poses_np,  # (B,4,4) float32
    }

    bin_path = out_dir / "anny_model.bin"
    manifest_path = out_dir / "anny_model.json"

    offset = 0
    manifest_arrays = {}
    chunks = []
    for name, arr in arrays.items():
        raw = arr.tobytes()
        manifest_arrays[name] = {
            "dtype":  arr.dtype.str,   # e.g. "<f4"
            "shape":  list(arr.shape),
            "offset": offset,
            "bytes":  len(raw),
        }
        chunks.append(raw)
        offset += len(raw)

    with open(bin_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    manifest = {
        "version":     1,
        "bone_count":  model.bone_count,
        "vert_count":  int(rest_verts_np.shape[0]),
        "face_count":  int(faces_np.shape[0]),
        "max_bones_per_vert": int(weights_np.shape[1]),
        "arrays":      manifest_arrays,
        "bone_parents": bone_parents,
        "bone_labels":  bone_labels,
        "baked_params": params,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_mb = offset / 1024 / 1024
    print(f"\nWrote {bin_path}  ({total_mb:.1f} MB)")
    print(f"Wrote {manifest_path}")
    print(f"\nArrays:")
    for name, info in manifest_arrays.items():
        print(f"  {name:20s} {str(info['shape']):20s} {info['dtype']}  "
              f"{info['bytes']//1024} KB  @ offset {info['offset']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets", help="Output directory")
    ap.add_argument("--params", default=None,
                    help="JSON phenotype params (overrides defaults)")
    args = ap.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.params:
        params.update(json.loads(args.params))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extract(params, out_dir)


if __name__ == "__main__":
    main()
