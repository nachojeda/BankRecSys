import argparse

parser = argparse.ArgumentParser(description="Run an AI/ML job from YAML/JSON configs.")
parser.add_argument("files", nargs="*", help="Config files for the job (local path only).")
parser.add_argument("-e", "--extras", nargs="*", default=[], help="Additional config strings for the job.")
parser.add_argument("-s", "--schema", action="store_true", help="Print settings schema and exit.")

def main(argv: list[str] | None = None) -> int:
    args = parser.parse_args(argv)
    if args.schema:
        # Print the schema of the settings
        print("Schema details here...")
        return 0
    # Execute the main application logic here...
    # ...
    return 0

if __name__ == "__main__":
    main()

