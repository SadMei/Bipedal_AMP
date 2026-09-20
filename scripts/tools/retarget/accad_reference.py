"""Resolve the local G1 reference names without silently dropping duplicate subjects."""
from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath


def reference_weights(path):
    matches = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and
                                               t.attr == 'motion_data_weights' for t in node.targets):
            matches.append(ast.literal_eval(node.value))
    if len(matches) != 1 or not matches[0] or any(not isinstance(k, str) or not k.endswith('_stageii')
                                                or not isinstance(v, (int, float)) or v <= 0
                                                for k, v in matches[0].items()):
        raise ValueError('Expected one literal positive G1 motion-weight dictionary')
    return matches[0]


def category_for_name(name):
    text = name.lower()
    if text.startswith('c'):
        return 'run_transition' if ('walk' in text or 'stand' in text) else 'run'
    if 'side_step' in text:
        return 'sideways'
    if 'turn' in text:
        return 'turn'
    if 'back' in text:
        return 'backward'
    raise ValueError(f'Unclassified reference name: {name}')


def resolve_reference(weights, member_names, extra_selection=()):
    by_stem = {}
    for name in member_names:
        path = PurePosixPath(name)
        if not name.startswith('ACCAD/') or path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe/non-ACCAD path')
        if name.endswith('_stageii.npz'):
            by_stem.setdefault(path.stem, []).append(name)
    clips, mappings = [], []
    for name, weight in weights.items():
        candidates = sorted(by_stem.get(name, []))
        if not candidates:
            raise ValueError(f'No ACCAD source for reference {name}')
        for path in candidates:
            subject = PurePosixPath(path).parts[1]
            clips.append(dict(stem=subject+'__'+name, category=category_for_name(name),
                relative=path.removeprefix('ACCAD/'), label=name.removesuffix('_stageii').replace('_', ' '),
                in_local_g1_amp_name_list=True, reference_name=name, reference_weight=weight,
                reference_source_ambiguous=len(candidates)>1,
                reference_source_candidates=candidates, selection_role='reference_candidate'))
        mappings.append(dict(reference_name=name, reference_weight=weight, source_candidates=candidates,
                             identity_status='unique_filename_match' if len(candidates)==1 else 'ambiguous_subject_not_resolved'))
    selected = {c['relative'] for c in clips}
    for stem, category, relative, label, _ in extra_selection:
        if relative in selected:
            continue
        clips.append(dict(stem=stem, category=category, relative=relative, label=label,
                          in_local_g1_amp_name_list=False, reference_name=None,
                          reference_source_ambiguous=False, selection_role='previous_preview_extra'))
        selected.add(relative)
    if len({c['stem'] for c in clips}) != len(clips):
        raise ValueError('Output stem collision')
    return clips, mappings
