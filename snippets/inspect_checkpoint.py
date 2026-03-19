import argparse
import json
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_dir: str):
    ckpt_path = Path(checkpoint_dir)
    print(f"Inspecting checkpoint directory: {ckpt_path}\n")

    # 1. Inspect Recipe State
    # Note: In ssi/checkpoint.py, recipe_state.pt is saved in the base output directory,
    # and the global_step_X subfolder contains the HF model shards.
    recipe_paths = [
        ckpt_path / "recipe_state.pt",
        ckpt_path.parents[1] / "recipe_state.pt", # if ckpt_path is epoch_X/global_step_Y
        ckpt_path.parent / "recipe_state.pt"
    ]

    recipe_found = False
    for p in recipe_paths:
        if p.exists():
            print(f"--- Found Recipe State: {p} ---")
            # Usually torchtune recipe state uses map_location="cpu"
            state = torch.load(p, map_location="cpu", weights_only=False)
            print(f"Keys in recipe_state.pt: {list(state.keys())}")
            for k, v in state.items():
                if isinstance(v, dict):
                    print(f"  {k}: dict with {len(v)} elements (keys: {list(v.keys())[:5]}...)")
                elif isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")
                else:
                    print(f"  {k}: {type(v)}")
            recipe_found = True
            break

    if not recipe_found:
        print("--- No recipe_state.pt found near the provided directory ---")

    # 2. Inspect Model Weights Structure (Index file)
    print("\n")
    weight_map = {}
    index_files = list(ckpt_path.glob("*.index.json"))
    if index_files:
        print(f"--- Found Model Index: {index_files[0]} ---")
        with open(index_files[0]) as f:
            index_data = json.load(f)
            metadata = index_data.get('metadata', {})
            total_size_gb = metadata.get('total_size', 0) / (1024 ** 3)
            print(f"Total size: {total_size_gb:.2f} GiB")
            weight_map = index_data.get("weight_map", {})
            print(f"Number of weight tensors mapped: {len(weight_map)}")

            # Print a few examples
            examples = list(weight_map.items())[:5]
            print("Example weight mappings:")
            for k, v in examples:
                print(f"  {k} -> {v}")
            if len(weight_map) > 5:
                print(f"  ... and {len(weight_map) - 5} more")
    else:
        print(f"--- No model index JSON found in {ckpt_path} ---")

    # 3. Inspect Model Shards
    if weight_map:
        print("\n--- Inspecting Model Shards on Disk ---")
        unique_shards = sorted(list(set(weight_map.values())))
        for shard_name in unique_shards:
            shard_path = ckpt_path / shard_name
            if not shard_path.exists():
                print(f"  [MISSING] Shard not found on disk: {shard_path}")
                continue

            # Read the shard
            shard_size_gb = shard_path.stat().st_size / (1024 ** 3)
            print(f"  [OK] {shard_name} ({shard_size_gb:.2f} GiB)")

            try:
                num_params = 0
                dtypes = set()

                if shard_name.endswith(".safetensors"):
                    try:
                        from safetensors import safe_open
                        with safe_open(shard_path, framework="pt", device="cpu") as f:
                            # Safetensors allows us to check shape/dtype without loading everything into memory
                            keys = f.keys()
                            for k in keys:
                                tensor_slice = f.get_slice(k)
                                shape = tensor_slice.get_shape()
                                params = 1
                                for dim in shape:
                                    params *= dim
                                num_params += params

                            # Get dtype directly from the first tensor
                            if keys:
                                first_key = keys[0]
                                dtype = f.get_tensor(first_key).dtype
                                dtypes.add(str(dtype))

                    except ImportError:
                        print("       [WARNING] 'safetensors' package not installed. Cannot read contents.")
                        continue

                elif shard_name.endswith(".bin") or shard_name.endswith(".pt"):
                    # Use mmap=True to avoid loading the whole file into active memory immediately
                    state_dict = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
                    for k, v in state_dict.items():
                        num_params += v.numel()
                        dtypes.add(str(v.dtype))
                else:
                    print(f"       [SKIPPED] Unrecognized shard extension: {shard_name}")
                    continue

                print(f"       Parameters: {num_params:,}")
                print(f"       DTypes: {', '.join(dtypes)}")

            except Exception as e:
                print(f"       [ERROR] Failed to read shard: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a training checkpoint.")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the checkpoint directory (e.g. checkpointer_output/epoch_0/global_step_100)")
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_dir)
