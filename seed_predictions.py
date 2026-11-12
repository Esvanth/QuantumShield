# """
# QuantumShield - send real test flows to the deployed endpoint.

# Populates the detection history with genuine CICIDS2017 rows rather than
# made-up vectors, so the dashboard's live feed and the demo video show a
# realistic mix of attack and benign traffic.

# Run from the project root:
#     python seed_predictions.py
#     python seed_predictions.py --attacks 10 --benign 15
#     python seed_predictions.py --url http://3.254.138.3:5000
# """

# import json
# import time
# import argparse

# import numpy as np
# import requests

# DEFAULT_URL = "http://3.254.138.3:5000"
# DATA_DIR = "data/processed/cicids2017"
# N_FEATURES = 16


# def load_rows(n_attacks, n_benign, seed=42):
#     X = np.load(f"{DATA_DIR}/X_test.npy")[:, :N_FEATURES]
#     y = np.load(f"{DATA_DIR}/yb_test.npy")

#     rng = np.random.default_rng(seed)
#     attack_idx = rng.choice(np.where(y == 1)[0], n_attacks, replace=False)
#     benign_idx = rng.choice(np.where(y == 0)[0], n_benign, replace=False)

#     rows = [(X[i], 1) for i in attack_idx] + [(X[i], 0) for i in benign_idx]
#     rng.shuffle(rows)
#     return rows


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--url", default=DEFAULT_URL)
#     ap.add_argument("--attacks", type=int, default=8)
#     ap.add_argument("--benign", type=int, default=12)
#     ap.add_argument("--delay", type=float, default=0.5,
#                     help="seconds between requests, so timestamps spread out")
#     args = ap.parse_args()

#     rows = load_rows(args.attacks, args.benign)
#     print(f"sending {len(rows)} flows to {args.url}/predict\n")

#     correct = 0
#     for i, (features, true_label) in enumerate(rows, 1):
#         body = {"features": [round(float(v), 6) for v in features]}
#         try:
#             r = requests.post(f"{args.url}/predict", json=body, timeout=30)
#             r.raise_for_status()
#             out = r.json()
#         except Exception as exc:
#             print(f"{i:3d}. request failed: {exc}")
#             continue

#         truth = "ATTACK" if true_label == 1 else "BENIGN"
#         hit = out["prediction"] == truth
#         correct += hit
#         print(f"{i:3d}. actual {truth:6s}  predicted {out['prediction']:6s}  "
#               f"p={out['attack_probability']:.6f}  {'ok' if hit else 'MISS'}")
#         time.sleep(args.delay)

#     print(f"\n{correct} of {len(rows)} correct")
#     print(f"check the history: {args.url}/recent")


# if __name__ == "__main__":
#     main()