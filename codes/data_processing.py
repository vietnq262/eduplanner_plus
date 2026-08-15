import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import os

# Optional (for HuggingFace arrow)
try:
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ===============================
# CONFIG
# ===============================
ROOT_DIR = os.getcwd()
DATASET_PATH = os.path.join(ROOT_DIR, "datasets/data_train.arrow")


# ===============================
# READ ARROW (AUTO DETECT)
# ===============================
def read_arrow_auto(file_path):
    """
    Read Arrow file (auto detect file or stream format)
    """
    with pa.memory_map(file_path, 'r') as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            print("✅ Loaded using FileReader")
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            print("✅ Loaded using StreamReader")

        table = reader.read_all()

    return table


# ===============================
# READ USING HUGGINGFACE (BEST FOR DATASETS)
# ===============================
def read_arrow_hf(file_path):
    """
    Read Arrow file using HuggingFace Dataset (if applicable)
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")

    dataset = Dataset.from_file(file_path)
    print("✅ Loaded using HuggingFace Dataset")
    return dataset


# ===============================
# SAVE FUNCTIONS
# ===============================
def save_outputs(df, output_prefix="train_output"):
    """
    Save DataFrame to CSV and JSON
    """
    csv_path = f"{output_prefix}.csv"
    json_path = f"{output_prefix}.json"

    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

    df.to_json(json_path, orient="records", lines=True)
    print(f"✅ Saved JSON: {json_path}")


# ===============================
# MAIN
# ===============================
def main():
    file_path = DATASET_PATH

    print("📂 File path:", file_path)
    print("📌 Exists:", os.path.exists(file_path))

    if not os.path.exists(file_path):
        print("❌ File not found. Check your path.")
        return

    # ---------------------------------
    # Try HuggingFace first (best option)
    # ---------------------------------
    try:
        dataset = read_arrow_hf(file_path)

        print("\n=== Dataset Info ===")
        print(dataset)

        print("\n=== First sample ===")
        print(dataset[0])

        df = dataset.to_pandas()

    except Exception as e:
        print("⚠️ HuggingFace load failed:", str(e))
        print("🔁 Falling back to PyArrow...")

        # ---------------------------------
        # Fallback to PyArrow
        # ---------------------------------
        table = read_arrow_auto(file_path)

        print("\n=== Schema ===")
        print(table.schema)

        print("\n=== First 5 rows (Arrow) ===")
        print(table.slice(0, 5))

        df = table.to_pandas()

    # ---------------------------------
    # Pandas preview
    # ---------------------------------
    print("\n=== DataFrame Info ===")
    print(df.info())

    print("\n=== First 5 rows (Pandas) ===")
    print(df.head())

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Shape ===")
    print(df.shape)

    # ---------------------------------
    # Save outputs
    # ---------------------------------
    save_outputs(df)


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    main()