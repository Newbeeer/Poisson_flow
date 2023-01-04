import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-t', '--target')
    args = parser.parse_args()

    folders = [f for f in os.listdir(args.dir) if f in SC09]

    for folder in folders:
        folder = os.path.join(args.dir, folder)
        if not os.path.isdir(folder): continue
        print(f"Processing {folder}")
        convert_folder(folder, args.target)

if __name__ == "__main__":
    main()