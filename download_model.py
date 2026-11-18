import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--path-file', default='')
    args = parser.parse_args()
    from modelscope import snapshot_download
    path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    if args.path_file:
        target = Path(args.path_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(Path(path).resolve()) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
