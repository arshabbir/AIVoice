#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

def parse_env(path: Path):
    data = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        data[key] = value
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    env_path = root / '.env'
    template_path = root / 'livekit' / 'livekit.yaml.template'
    output_path = root / 'livekit' / 'livekit.yaml'
    data = parse_env(env_path)
    text = template_path.read_text(encoding='utf-8')
    def repl(match):
        key = match.group(1)
        if key not in data:
            raise SystemExit(f"Missing variable in .env: {key}")
        return data[key]
    output_path.write_text(pattern.sub(repl, text), encoding='utf-8')
    print(f"Rendered {output_path}")

if __name__ == '__main__':
    main()
