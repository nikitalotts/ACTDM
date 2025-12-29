# print_models_on_gpu.py
import torch
import argparse
import os

def to_device(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [to_device(x, device) for x in obj]
        return type(obj)(seq)
    return obj

def print_state_dict_shapes(sd, n_keys=200):
    keys = sorted(sd.keys())
    total = len(keys)
    print(f"  state_dict keys: {total}")
    for k in keys[:min(total, n_keys)]:
        v = sd[k]
        if isinstance(v, torch.Tensor):
            print(f"    {k:80s} {tuple(v.shape)}  device={v.device}")
        else:
            print(f"    {k:80s} type={type(v)}")
    if total > n_keys:
        print(f"    ... (+{total - n_keys} keys)")

def handle_loaded(obj, path, device, max_print_keys=200):
    print("=" * 80)
    print(f"File: {path}")
    print(f"Loaded type: {type(obj)}")
    print(f"Moving objects to device: {device}")
    
    if isinstance(obj, torch.nn.Module):
        obj = obj.to(device)
        print("=== torch.nn.Module ===")
        print(obj)
        return

    if isinstance(obj, dict):
        found_module = False
        for k, v in obj.items():
            if isinstance(v, torch.nn.Module):
                found_module = True
                print(f"\n--- Module at top-level key: '{k}' ---")
                v = v.to(device)
                print(v)

        candidates = [k for k, v in obj.items() if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values())]
        top_is_state_dict = any(isinstance(v, torch.Tensor) for v in obj.values())

        if top_is_state_dict and not found_module:
            print("\nTop-level looks like state_dict (key->tensor). Moving tensors to device and printing shapes...")
            sd_on_dev = to_device(obj, device)
            print_state_dict_shapes(sd_on_dev, n_keys=max_print_keys)
            return

        for cand in candidates:
            print(f"\n--- state_dict-like nested key: '{cand}' ---")
            sd = obj[cand]
            sd_on_dev = to_device(sd, device)
            print_state_dict_shapes(sd_on_dev, n_keys=max_print_keys)
            return

        if not found_module and not candidates:
            print("\nNo nn.Module objects or nested state_dicts detected. Top-level keys:")
            for k in list(obj.keys())[:200]:
                print(f"  {k:60s} -> {type(obj[k])}")

    else:
        print("Object repr:")
        print(repr(obj)[:2000])

    print("=" * 80 + "\n")

def load_on_device(path, device):
    print(f"Loading checkpoint '{path}' to device {device} ...")
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print("Error with map_location=device:", e)
        ckpt = torch.load(path, map_location="cpu")
    return ckpt

def main():
    default_files = [
        "datasets/rocstories/decoder-bert-base-cased-80-transformer.pth",
        "checkpoints/1tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0/last.pth"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=default_files, help="paths to .pth files")
    parser.add_argument("--device-mode", choices=["auto", "cpu", "cuda"], default="auto", help="Device mode: auto, cpu, cuda")
    parser.add_argument("--max-print-keys", type=int, default=200, help="how many state_dict keys to print")
    args = parser.parse_args()

    # Выбираем устройство
    if args.device_mode == "cpu":
        device = "cpu"
    elif args.device_mode == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda:0"
    else:  # auto
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\n>>> Using device: {device}\n")

    for p in args.files:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        try:
            loaded = load_on_device(p, device)
            handle_loaded(loaded, p, device, max_print_keys=args.max_print_keys)
        except RuntimeError as e:
            print(f"RuntimeError while processing {p}: {e}")
        except Exception as e:
            print(f"Error while processing {p}: {e}")

if __name__ == "__main__":
    main()
    print('ALL OK')
