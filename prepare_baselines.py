"""Stage the author model modules used by the baseline drivers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def install(name, spec, checkout=None):
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / 'stage'
        stage.mkdir()
        if checkout is None:
            checkout = Path(tmp) / 'source'
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), '-c', 'core.autocrlf=false',
                            'fetch', '-q', '--depth=1', spec['repository'], spec['revision']], check=True)
            subprocess.run(['git', '-C', str(checkout), '-c', 'core.autocrlf=false',
                            'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)

        for item in spec['files']:
            text = (checkout / item['source']).read_text(encoding='utf-8-sig')
            raw = (text.rstrip('\n') + '\n').encode('utf-8')
            if digest(raw) != item['source_sha256']:
                raise ValueError(f'Unexpected upstream content: {name}/{item["source"]}')
            path = stage / item['destination']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            if item['patch']:
                subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply',
                                str(ROOT / item['patch'])], cwd=stage, check=True)
                path.write_bytes(path.read_text(encoding='utf-8').encode('utf-8'))
            if digest(path.read_bytes()) != item['sha256']:
                raise ValueError(f'Patched source differs: {item["destination"]}')

        for item in spec['files']:
            dest = ROOT / item['destination']
            data = (stage / item['destination']).read_bytes()
            if dest.exists() and dest.read_bytes() != data:
                raise FileExistsError(dest)
        for item in spec['files']:
            dest = ROOT / item['destination']
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((stage / item['destination']).read_bytes())
        print(f'{name}: {spec["revision"]}')


def main():
    lock = json.loads((ROOT / 'upstream/sources.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser()
    parser.add_argument('models', choices=sorted(lock), nargs='+')
    parser.add_argument('--checkout', type=Path, help='Local author checkout for a single model')
    args = parser.parse_args()
    if args.checkout is not None and len(args.models) != 1:
        parser.error('--checkout requires one model')
    for name in args.models:
        install(name, lock[name], args.checkout)


if __name__ == '__main__':
    main()
