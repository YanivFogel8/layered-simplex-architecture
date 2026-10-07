"""Download the pinned scientific inputs and verify their recorded checksums."""
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch():
    data = Path(__file__).resolve().parent / 'data'
    manifest = json.loads((data / 'provenance.json').read_text())
    for name, record in manifest['assets'].items():
        path = data / name
        content = path.read_bytes() if path.exists() else urllib.request.urlopen(record['url'], timeout=120).read()
        if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}; no file was overwritten')
        if not path.exists():
            temporary = path.with_suffix('.download')
            temporary.write_bytes(content)
            temporary.replace(path)
        print(f'Verified {name}')


if __name__ == '__main__':
    fetch()
