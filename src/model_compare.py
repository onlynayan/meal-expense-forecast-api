import json, os

def compare_models():
    model_files = {
        "RandomForest_Daily": "./models/metadata_rf.json",
        "RandomForest_Monthly": "./models/metadata_rf_monthly.json",
        "Prophet": "./models/metadata_prophet.json"
    }

    results = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                results[name] = json.load(f)

    # Rank models by R2 and MAPE
    ranking = sorted(
        results.items(),
        key=lambda x: (-x[1].get("R2", 0), x[1].get("MAPE", 999))
    )

    best_model = ranking[0][0] if ranking else None

    summary = {
        "best_model": best_model,
        "comparison": {name: metrics for name, metrics in results.items()}
    }

    os.makedirs("./models", exist_ok=True)
    with open("./models/model_comparison.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Model comparison completed.")
    print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    compare_models()
