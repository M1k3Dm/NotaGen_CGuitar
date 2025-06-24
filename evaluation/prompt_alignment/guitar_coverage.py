import os
import glob
import argparse

def main():
    """
    Script to calculate the percentage of .abc files in a given directory
    that contain at least one snm="Guit." field.
    """
    parser = argparse.ArgumentParser(
        description="Calculate percentage of .abc files containing snm=\"Guit.\""
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search (defaults to current directory)"
    )
    args = parser.parse_args()

    # Find all .abc files in the specified directory
    abc_files = glob.glob(os.path.join(args.directory, "*.abc"))
    total = len(abc_files)

    # If no .abc files found, inform the user and exit
    if total == 0:
        print(f"No .abc files found in directory '{args.directory}'")
        return

    # Count how many files contain the target substring
    count = 0
    for filepath in abc_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'snm="Guit."' in content:
                    count += 1
        except Exception as e:
            print(f"Warning: could not read file '{filepath}': {e}")

    # Output the result as a fraction
    print(f"{count}/{total}")

if __name__ == "__main__":
    main()

