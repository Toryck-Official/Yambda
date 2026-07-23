from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train an ensemble of independent future predictors.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "predictor_ensemble"))
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_train_rows", type=int, default=5000)
    parser.add_argument("--max_val_rows", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--response_pos_weight", default="")
    parser.add_argument("--regret_class_weight", default="")
    parser.add_argument("--device", default="auto")
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts: list[str] = []
    for idx in range(int(args.ensemble_size)):
        member_dir = out_dir / f"member_{idx:02d}"
        seed = int(args.base_seed) + idx
        cmd = [
            sys.executable,
            str(ROOT / "03_train" / "train_predictor.py"),
            "--data_dir", str(args.data_dir),
            "--embed_store", str(args.embed_store),
            "--out_dir", str(member_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--max_train_rows", str(args.max_train_rows),
            "--max_val_rows", str(args.max_val_rows),
            "--d_model", str(args.d_model),
            "--n_layer", str(args.n_layer),
            "--n_head", str(args.n_head),
            "--dropout", str(args.dropout),
            "--seed", str(seed),
            "--device", str(args.device),
        ]
        if args.response_pos_weight:
            cmd.extend(["--response_pos_weight", str(args.response_pos_weight)])
        if args.regret_class_weight:
            cmd.extend(["--regret_class_weight", str(args.regret_class_weight)])
        cmd.extend(extra)
        print(f"[ensemble] member={idx} seed={seed} out_dir={member_dir}", flush=True)
        subprocess.run(cmd, check=True)
        ckpts.append(str(member_dir / "future_predictor.pt"))

    manifest = {
        "ensemble_size": int(args.ensemble_size),
        "base_seed": int(args.base_seed),
        "predictor_ckpts": ckpts,
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
