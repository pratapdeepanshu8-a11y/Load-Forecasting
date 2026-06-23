import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

features = [
    "hour",
    "lag_1",
    "month",
    "lag_24"
]

importance = [
    0.35,
    0.30,
    0.20,
    0.15
]

assets_dir = BASE_DIR / "assets"
assets_dir.mkdir(exist_ok=True)

plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.savefig(
    assets_dir / "feature_importance.png"
)

plt.close()

print("Feature Importance Graph Saved!")