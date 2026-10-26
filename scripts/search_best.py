import json
from pathlib import Path


def find_max_value_in_jsons(root_dir: str, target_filename: str, target_key: str):
    root_path = Path(root_dir)

    max_value = None
    max_file = None

    for file_path in root_path.rglob(target_filename):
        if not file_path.is_file():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                current_data = data
                for key in target_key.split("."):
                    current_data = current_data[key]

                value = float(current_data)

                if max_value is None or value > max_value:
                    max_value = value
                    max_file = file_path

        except (KeyError, TypeError, ValueError):
            continue
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error in reading ({file_path}): {e}")

    return max_file, max_value


if __name__ == "__main__":
    ROOT_DIRECTORY = "../logs"
    TARGET_JSON = "metrics.json"
    TARGET_KEY = "seleted_test"

    best_file, highest_value = find_max_value_in_jsons(
        ROOT_DIRECTORY, TARGET_JSON, TARGET_KEY
    )

    if best_file:
        print(f"best_file: {best_file}")
        print(f"({TARGET_KEY}): {highest_value}")
    else:
        print("Can not find file or item.")
