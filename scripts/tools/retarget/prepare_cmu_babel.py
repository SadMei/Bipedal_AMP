"""Download a small explicit CMU/BABEL preview selection, never entire corpora."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import requests

from cmu_babel import cmu_identity, load_babel, parse_amc, parse_asf, save_json

# IDs and bounds were inspected in the user's BABEL 1.0 dense labels. Broad
# "walk" sequence labels are only candidates: the trajectory gate must pass.
SELECTION = [
    ('straight_07_03', 'straight', '251', .15, 3.30, 'walk'),
    ('turn_46_01', 'turn', '13202', .82, 2.12, 'turn around'),
    ('turn_133_18', 'turn', '378', 1.252, 2.315, 'turn right'),
    ('sideways_right_113_18', 'sideways', '11492', .50, 8.40, 'sidestep right'),
    ('sideways_left_113_18', 'sideways', '11492', 9.20, 17.10, 'sidestep left'),
    ('stand_77_02', 'stand', '6750', .10, 3.10, 'stand'),
]


def download(url, target, max_bytes=16*1024*1024):
    if target.exists():
        payload = target.read_bytes()
    else:
        # Only bounded public CMU files and BABEL's public renders. No login,
        # private credentials, licence acceptance or archive crawling.
        chunks, size = [], 0
        with requests.get(url, timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f'Download exceeded limit: {url}')
                chunks.append(chunk)
        payload = b''.join(chunks)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    return dict(url=url, path=str(target), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def source_description(page, motion):
    # Official rows include an amc link, movie links and a numeric frame rate.
    for row in re.findall(r'<tr\b[^>]*>.*?</tr>', page, re.S | re.I):
        if f'{motion}.amc' not in row:
            continue
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S | re.I)
        cells = [html.unescape(re.sub(r'\s+', ' ', re.sub('<[^>]+>', ' ', c))).strip() for c in cells]
        rates = [int(c) for c in cells if c in ('60', '120', '30')]
        if len(rates) != 1:
            raise ValueError(f'Cannot identify unique official frame rate: {motion}: {cells}')
        return dict(fps=rates[0], row=' | '.join(cells))
    raise ValueError(f'Motion absent from official source page: {motion}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--babel', type=Path, default=Path('/root/mjw/babel_v1-0_release.zip'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-name', default='manifest.json')
    parser.add_argument('--previews', action='store_true')
    args = parser.parse_args()
    if Path(args.manifest_name).name != args.manifest_name or not args.manifest_name.endswith('.json'):
        parser.error('--manifest-name must be a simple JSON filename')
    records = load_babel(args.babel)
    manifest = dict(version=1, babel_archive_sha256=hashlib.sha256(args.babel.read_bytes()).hexdigest(),
                    source='CMU original ASF/AMC + BABEL 1.0 annotations; NOT AMASS fitted SMPL-X',
                    cmu_faq='http://mocap.cs.cmu.edu/faqs.php',
                    babel_license='https://babel.is.tue.mpg.de/license.html',
                    kinematic_preview_only=True, files={}, clips=[])
    manifest['selection_notes'] = [
        'Initial 07_04 slow-walk candidate rejected: source pelvis yaw range 18.76 degrees > 15 degree gate.',
        '77_02 standing window narrowed from [2,10) to [0.1,3.1): latter root XY extent is 2.9cm.',
        'Straight 07_03 has a broad BABEL walk label; straightness is established by the measured trajectory gate.',
    ]
    for stem, category, sid, start, end, label in SELECTION:
        record = records[sid]
        subject, motion = cmu_identity(record['feat_p'])
        frame_labels = (record.get('frame_ann') or {}).get('labels', [])
        sequence_labels = (record.get('seq_ann') or {}).get('labels', [])
        evidence = [a for a in frame_labels if a.get('proc_label') == label
                    and a['start_t'] <= start and a['end_t'] >= end]
        level = 'frame'
        if not evidence:
            evidence = [a for a in sequence_labels if a.get('proc_label') == label]
            level = 'sequence_candidate_requires_geometry'
        if not evidence or not 0 <= start < end <= record['dur'] + .01:
            raise ValueError(f'BABEL no longer agrees with selection {stem}')
        file_specs = [(f'{subject}.html', f'http://mocap.cs.cmu.edu/search.php?subjectnumber={int(subject)}'),
                      (f'{subject}.asf', f'http://mocap.cs.cmu.edu/subjects/{subject}/{subject}.asf'),
                      (f'{motion}.amc', f'http://mocap.cs.cmu.edu/subjects/{subject}/{motion}.amc')]
        if args.previews:
            file_specs.append((f'babel_{int(sid):06d}.mp4', record['url']))
        for filename, url in file_specs:
            if filename not in manifest['files']:
                manifest['files'][filename] = download(url, args.output / filename)
                print(f'[SOURCE] {filename}: {manifest["files"][filename]["bytes"]} bytes', flush=True)
        official = source_description((args.output / f'{subject}.html').read_text(), motion)
        skeleton = parse_asf((args.output / f'{subject}.asf').read_text())
        frames = parse_amc((args.output / f'{motion}.amc').read_text(), skeleton)
        raw_duration = len(frames['root']) / official['fps']
        error = raw_duration - record['dur']
        if abs(error) > .075:
            raise ValueError(f'Timebase mismatch {motion}: raw={raw_duration}, BABEL={record["dur"]}; do not stretch')
        manifest['clips'].append(dict(stem=stem, category=category, babel_sid=sid, split=record['split'],
            feat_p=record['feat_p'], subject=subject, motion=motion, source_fps=official['fps'],
            official_description=official['row'], start_t=start, end_t=end, label=label,
            label_level=level, evidence=evidence, overlapping_labels=[a for a in frame_labels
                if a.get('start_t', 0) < end and a.get('end_t', 0) > start and a not in evidence],
            raw_frames=len(frames['root']), raw_duration_s=raw_duration, babel_duration_s=record['dur'],
            duration_difference_s=error, time_offset_s=0.,
            time_alignment='Same CMU identity and near-equal duration; zero offset, no time stretching. '
                           'Duration agreement alone is not proof of frame-exact AMASS registration.',
            babel_preview_url=record['url']))
    manifest['total_download_bytes'] = sum(f['bytes'] for f in manifest['files'].values())
    save_json(args.output / args.manifest_name, manifest)
    print(f'[DONE] {len(manifest["clips"])} candidates, {manifest["total_download_bytes"]/1e6:.2f} MB')


if __name__ == '__main__':
    main()
