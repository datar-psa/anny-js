#!/usr/bin/env python3
"""
Generate FK + LBS parity fixtures by running the reference Python anny package
(Apache 2.0) on a fixed set of canonical poses, then dumping deltas + expected
outputs as JSON for the JS parity test.

Run with the body-tuning venv active (or any env with `anny` + `torch` + numpy):

    source /home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/venv/bin/activate
    python tools/gen_parity_fixtures.py

The script loads the SAME `assets/anny_model.{json,bin}` the JS library ships
so the model state is bit-identical on both sides — only the FK + LBS impls
differ.

Fixture format (per pose, JSON):
    {
      "pose": "<name>",
      "deltas": [ [9 floats row-major 3x3] | null  ×B ],
      "expected_bone_xforms": [ 16 floats row-major 4x4  ×B ],
      "expected_vertices":    [ 3 floats  ×V ]
    }
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from anny.utils.kinematics import forward_kinematic
from anny.skinning.skinning import linear_blend_skinning


# ── Model loading from the anny-js asset bundle ────────────────────────────

def _np_dtype(dt: str) -> np.dtype:
    return np.dtype(dt)


def load_anny_assets(assets_dir: Path):
    """Return (rest_vertices, faces, bone_weights, bone_indices, rest_bone_poses,
    bone_parents, bone_labels) from the JS asset bundle."""
    manifest = json.loads((assets_dir / "anny_model.json").read_text())
    buf = (assets_dir / "anny_model.bin").read_bytes()

    def view(name: str):
        a = manifest["arrays"][name]
        dt = _np_dtype(a["dtype"])
        arr = np.frombuffer(buf, dtype=dt, count=a["bytes"] // dt.itemsize, offset=a["offset"])
        return arr.reshape(a["shape"]).copy()  # decouple from the buffer

    rest_vertices   = view("rest_vertices")            # (V,3) float32
    faces           = view("faces")                    # (F,3) int32
    bone_weights    = view("bone_weights")             # (V,K) float32
    bone_indices    = view("bone_indices")             # (V,K) int32
    rest_bone_poses = view("rest_bone_poses")          # (B,4,4) float32
    bone_parents    = np.asarray(manifest["bone_parents"], dtype=np.int32)
    bone_labels     = list(manifest["bone_labels"])
    return (rest_vertices, faces, bone_weights, bone_indices, rest_bone_poses,
            bone_parents, bone_labels)


# ── Pose construction helpers ──────────────────────────────────────────────

def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)


def make_deltas(boneCount: int, bone_labels: List[str], named) -> List[Optional[np.ndarray]]:
    """`named` is a list of (bone_name, 3x3_rotation) pairs. Returns a list of
    length boneCount where unset entries are None."""
    lookup = {n: i for i, n in enumerate(bone_labels)}
    out: List[Optional[np.ndarray]] = [None] * boneCount
    for name, r in named:
        idx = lookup.get(name)
        if idx is None:
            print(f"warning: bone {name!r} not in rig; skipping", file=sys.stderr)
            continue
        out[idx] = r.astype(np.float32)
    return out


def canonical_poses(boneCount: int, bone_labels: List[str]):
    """A small but representative set of poses for parity testing."""
    return [
        # 1. Rest pose — every delta is identity → every transform should be identity.
        ("rest", make_deltas(boneCount, bone_labels, [])),
        # 2. Both arms straight up (90° around Z on the upperarm bones).
        ("arms_up", make_deltas(boneCount, bone_labels, [
            ("upperarm01.L", rotation_z(+math.pi / 2)),
            ("upperarm01.R", rotation_z(-math.pi / 2)),
        ])),
        # 3. Right knee bent (90° around X on the lower leg).
        ("knee_bend_R", make_deltas(boneCount, bone_labels, [
            ("lowerleg01.R", rotation_x(math.pi / 2)),
        ])),
        # 4. Torso twist (45° around Z on each spine bone — accumulates up the chain).
        ("torso_twist", make_deltas(boneCount, bone_labels, [
            ("spine01", rotation_z(math.pi / 16)),
            ("spine02", rotation_z(math.pi / 16)),
            ("spine03", rotation_z(math.pi / 16)),
        ])),
        # 5. Combined: arms back + spine bend forward + head tilted (compound test).
        ("compound", make_deltas(boneCount, bone_labels, [
            ("upperarm01.L", rotation_y(+math.pi / 4)),
            ("upperarm01.R", rotation_y(-math.pi / 4)),
            ("spine03",      rotation_x(math.pi / 8)),
            ("neck01",       rotation_x(-math.pi / 12)),
        ])),
    ]


# ── Run FK + LBS via the reference Python anny ─────────────────────────────

def run_fk_lbs(rest_vertices, bone_weights, bone_indices, rest_bone_poses,
               bone_parents, deltas: List[Optional[np.ndarray]]):
    boneCount = rest_bone_poses.shape[0]

    # Build delta_transforms (B,4,4): identity where delta is None.
    deltas_4x4 = np.tile(np.eye(4, dtype=np.float32), (boneCount, 1, 1))
    for b, d in enumerate(deltas):
        if d is not None:
            deltas_4x4[b, :3, :3] = d
    delta_t = torch.from_numpy(deltas_4x4)[None]                        # (1,B,4,4)
    rest_t  = torch.from_numpy(rest_bone_poses)[None]                   # (1,B,4,4)
    parents_t = torch.from_numpy(bone_parents.astype(np.int64))         # (B,)

    _, transforms = forward_kinematic(parents_t, rest_t, delta_t)       # (1,B,4,4)

    verts_t = torch.from_numpy(rest_vertices)[None]                     # (1,V,3)
    weights_t = torch.from_numpy(bone_weights)[None]                    # (1,V,K)
    idx_t = torch.from_numpy(bone_indices.astype(np.int64))[None]       # (1,V,K)

    skinned = linear_blend_skinning(verts_t, weights_t, idx_t, transforms)  # (1,V,3)
    return transforms[0].cpu().numpy(), skinned[0].cpu().numpy()


# ── Serialize ──────────────────────────────────────────────────────────────

def to_jsonable_deltas(deltas):
    out = []
    for d in deltas:
        if d is None:
            out.append(None)
        else:
            # Row-major 3x3 → flat 9 floats. anny's deltas are float32; cast to
            # Python float for JSON.
            out.append([float(v) for v in d.flatten().tolist()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets", help="anny-js assets/ directory")
    ap.add_argument("--out", default="tests/parity/fixtures", help="Output directory")
    args = ap.parse_args()

    assets_dir = Path(args.assets).resolve()
    out_dir    = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model assets from {assets_dir}…", flush=True)
    (rest_vertices, faces, bone_weights, bone_indices, rest_bone_poses,
     bone_parents, bone_labels) = load_anny_assets(assets_dir)

    boneCount = rest_bone_poses.shape[0]
    vertCount = rest_vertices.shape[0]
    print(f"  bones={boneCount}  verts={vertCount}")

    poses = canonical_poses(boneCount, bone_labels)

    for name, deltas in poses:
        print(f"\nPose '{name}'…", flush=True)
        bone_xforms, skinned = run_fk_lbs(
            rest_vertices, bone_weights, bone_indices, rest_bone_poses,
            bone_parents, deltas,
        )

        # Flatten 4x4 row-major (numpy default is C-order — already row-major).
        xforms_flat = bone_xforms.reshape(boneCount, 16).astype(np.float32).tolist()
        verts_flat  = skinned.reshape(-1).astype(np.float32).tolist()

        out_path = out_dir / f"{name}.json"
        with out_path.open("w") as f:
            json.dump({
                "pose": name,
                "boneCount": boneCount,
                "vertCount": vertCount,
                "deltas": to_jsonable_deltas(deltas),
                "expected_bone_xforms": xforms_flat,
                "expected_vertices": verts_flat,
            }, f)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  → {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
