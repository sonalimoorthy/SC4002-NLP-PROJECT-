"""MASSIVE 1.0 en-US preprocessing for joint intent and slot prediction."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tarfile
import urllib.request

DATA_URL = 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz'
EXPECTED_SPLITS = {'train': 11514, 'dev': 2033, 'test': 2974}
PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_ID, UNK_ID, IGNORE_INDEX = 0, 1, -100
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)
ANNOTATION_PATTERN = re.compile(r'\[([A-Za-z_][A-Za-z_0-9.-]*)\s*:\s*([^\[\]]*?)\]')


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_jsonl_lines(lines):
    for line_number, line in enumerate(lines, 1):
        if isinstance(line, bytes):
            line = line.decode('utf-8-sig' if line_number == 1 else 'utf-8')
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON on line {line_number}: {exc}') from exc
        for key in ('id', 'locale', 'partition', 'utt', 'annot_utt', 'intent'):
            if key not in row:
                raise ValueError(f'Line {line_number} missing {key}')
        if row['locale'] != 'en-US':
            raise ValueError(f'Found non-en-US locale on line {line_number}')
        if row['partition'] not in EXPECTED_SPLITS:
            raise ValueError(f'Unexpected partition {row["partition"]!r} on line {line_number}')
        yield row


def read_original_data(input_path=None, archive_path=None):
    """Read en-US records without changing their original split/order."""
    if (input_path is None) == (archive_path is None):
        raise ValueError('Supply exactly one of input_path or archive_path')
    if input_path is not None:
        with Path(input_path).open('r', encoding='utf-8-sig') as f:
            return list(load_jsonl_lines(f))
    with tarfile.open(archive_path, mode='r:gz') as tar:
        match = next((m for m in tar if m.isfile() and m.name.endswith('/data/en-US.jsonl')), None)
        if match is None:
            raise FileNotFoundError('Archive has no 1.0/data/en-US.jsonl')
        stream = tar.extractfile(match)
        if stream is None:
            raise FileNotFoundError('Could not open en-US.jsonl in archive')
        with stream:
            return list(load_jsonl_lines(stream))


def download_official_archive(path):
    """Only fetches the original dataset; never generates training data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        print(f'Using existing dataset archive: {path}')
        return path
    print(f'Downloading official MASSIVE 1.0 archive from: {DATA_URL}')
    try:
        urllib.request.urlretrieve(DATA_URL, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def parse_annotations(utt, annotated):
    """Return gold spans with half-open character offsets in ORIGINAL `utt`.

    Reconstruct unmarked text from `annot_utt`. If whitespace-only differences
    occur between it and `utt`, align via corresponding non-space characters.
    Non-whitespace mismatch fails loudly instead of corrupting labels.
    """
    assembled = []
    slots_in_assembled = []
    cursor, pos = 0, 0
    for match in ANNOTATION_PATTERN.finditer(annotated):
        outside = annotated[cursor:match.start()]
        assembled.append(outside)
        pos += len(outside)
        slot_type, surface = match.group(1), match.group(2)
        # Annotation delimiters use optional spaces around the colon; strip
        # whitespace at the entity boundaries, not whitespace inside the entity.
        # Original matched inner string is preserved for the alignment process.
        raw_start, raw_end = pos, pos + len(surface)
        if not surface.strip():
            raise ValueError(f'Empty slot surface in {annotated!r}')
        slots_in_assembled.append((slot_type, raw_start, raw_end))
        assembled.append(surface)
        pos += len(surface)
        cursor = match.end()
    assembled.append(annotated[cursor:])
    plain = ''.join(assembled)
    # Detect malformed annotations without mistaking normal text for valid slots.
    if '[' in plain or ']' in plain:
        raise ValueError(f'Unparsed square bracket(s) in annot_utt: {annotated!r}')

    if plain == utt:
        spans = [(t, s, e) for (t, s, e) in slots_in_assembled]
    else:
        a = [(i, c) for i, c in enumerate(plain) if not c.isspace()]
        b = [(i, c) for i, c in enumerate(utt) if not c.isspace()]
        if [c for _, c in a] != [c for _, c in b]:
            raise ValueError('annot_utt does not reconstruct utt (even ignoring whitespace): '
                             f'annot_plain={plain!r} / utt={utt!r}')
        align = {a_i: b_i for (a_i, _), (b_i, _) in zip(a, b)}
        spans = []
        for slot_type, start, end in slots_in_assembled:
            nonspace = [idx for idx in range(start, end) if idx in align]
            if not nonspace:
                raise ValueError('Slot has no alignable characters')
            spans.append((slot_type, align[nonspace[0]], align[nonspace[-1]] + 1))

    result = []
    for typ, start, end in spans:
        # Trim any whitespace at the boundaries of annotations.
        while start < end and utt[start].isspace():
            start += 1
        while end > start and utt[end - 1].isspace():
            end -= 1
        if not start < end:
            raise ValueError('Empty aligned span')
        result.append({'type': typ, 'start': start, 'end': end, 'span': utt[start:end]})
    result.sort(key=lambda x: (x['start'], x['end']))
    for left, right in zip(result, result[1:]):
        if left['end'] > right['start']:
            raise ValueError(f'Overlapping annotated spans: {left} / {right}')
    return result


def tokenize_utterance(utt, mode='word'):
    if mode == 'word':
        matches = list(TOKEN_PATTERN.finditer(utt))
        return ([m.group() for m in matches], [[m.start(), m.end()] for m in matches])
    if mode == 'char':
        return ([c for c in utt if not c.isspace()],
                [[i, i + 1] for i, c in enumerate(utt) if not c.isspace()])
    raise ValueError(f'Unknown tokenizer mode: {mode}')


def spans_to_bio(gold_slots, token_offsets):
    """Align annotated spans to token boundaries."""
    tags = ['O'] * len(token_offsets)
    boundary_warnings = []
    for slot in gold_slots:
        start, end, typ = slot['start'], slot['end'], slot['type']
        touching = [i for i, (a, b) in enumerate(token_offsets) if a < end and b > start]
        if not touching:
            raise ValueError(f'No token overlaps slot: {slot}')
        if token_offsets[touching[0]][0] != start or token_offsets[touching[-1]][1] != end:
            boundary_warnings.append({'type': typ, 'start': start, 'end': end,
                                      'span': slot['span'], 'reason': 'slot boundary lies within a token'})
        for rank, i in enumerate(touching):
            if tags[i] != 'O':
                raise ValueError(f'One token intersects multiple annotated slots: token={i}; slot={slot}')
            tags[i] = f'{"B" if rank == 0 else "I"}-{typ}'
    return tags, boundary_warnings


def build_maps(train_rows, min_freq=1):
    """Build vocabulary and label maps from the training partition."""
    intent_names = sorted({r['intent'] for r in train_rows})
    slot_names = sorted({s['type'] for r in train_rows for s in r['gold_slots']})
    counts = Counter(tok.lower() for r in train_rows for tok in r['tokens'])
    words = sorted((tok for tok, count in counts.items() if count >= min_freq),
                   key=lambda tok: (-counts[tok], tok))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for tok in words:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    intent2id = {name: i for i, name in enumerate(intent_names)}
    slot2id = {'O': 0}
    for typ in slot_names:
        slot2id[f'B-{typ}'] = len(slot2id)
        slot2id[f'I-{typ}'] = len(slot2id)
    return vocab, intent2id, slot2id


def encode_row(row, vocab, intent2id, slot2id):
    if row['intent'] not in intent2id:
        raise ValueError(f'Unseen intent in {row["partition"]}: {row["intent"]}')
    missing_tags = [tag for tag in row['slot_tags'] if tag not in slot2id]
    if missing_tags:
        raise ValueError(f'Unseen slot tag in {row["partition"]}: {missing_tags[:3]}')
    return {**row,
            'input_ids': [vocab.get(tok.lower(), UNK_ID) for tok in row['tokens']],
            'intent_id': intent2id[row['intent']],
            'slot_ids': [slot2id[tag] for tag in row['slot_tags']],
            'attention_mask': [1] * len(row['tokens']),
            'length': len(row['tokens'])}


def bio_to_spans(utt, token_offsets, bio_tags):
    """Convert BIO predictions into typed spans with character offsets."""
    if len(token_offsets) != len(bio_tags):
        raise ValueError('Token offsets and BIO predictions have different lengths')
    result = []
    active = None
    for (start, end), tag in zip(token_offsets, bio_tags):
        if tag == 'O':
            if active:
                result.append(active)
                active = None
            continue
        if '-' not in tag or tag[0] not in ('B', 'I'):
            raise ValueError(f'Invalid BIO label: {tag!r}')
        prefix, typ = tag.split('-', 1)
        if prefix == 'B' or active is None or active['type'] != typ:
            if active:
                result.append(active)
            active = {'type': typ, 'start': start, 'end': end}
        else:
            active['end'] = end
    if active:
        result.append(active)
    for slot in result:
        slot['span'] = utt[slot['start']:slot['end']]
    return result


def evaluate_exact(gold_rows, predicted_intent_ids, predicted_slot_id_sequences, slot2id):
    """Calculate intent accuracy and exact-span slot metrics (percent)."""
    if not (len(gold_rows) == len(predicted_intent_ids) == len(predicted_slot_id_sequences)):
        raise ValueError('Number of predictions differs from number of examples')
    id2slot = {v: k for k, v in slot2id.items()}
    correct_intents = tp = fp = fn = 0
    for row, intent_id, slot_ids in zip(gold_rows, predicted_intent_ids, predicted_slot_id_sequences):
        correct_intents += int(int(intent_id) == row['intent_id'])
        tags = [id2slot[int(s)] for s in slot_ids[:row['length']]]
        predicted = bio_to_spans(row['utt'], row['token_offsets'], tags)
        pred_counter = Counter((s['type'], s['start'], s['end']) for s in predicted)
        gold_counter = Counter((s['type'], s['start'], s['end']) for s in row['gold_slots'])
        tp += sum((pred_counter & gold_counter).values())
        fp += sum((pred_counter - gold_counter).values())
        fn += sum((gold_counter - pred_counter).values())
    intent_acc = 100 * correct_intents / len(gold_rows) if gold_rows else 0.0
    precision = 100 * tp / (tp + fp) if (tp + fp) else 0.0
    recall = 100 * tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 100 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {'intent_accuracy': round(intent_acc, 2), 'slot_micro_precision': round(precision, 2),
            'slot_micro_recall': round(recall, 2), 'slot_micro_f1': round(f1, 2),
            'slot_tp': tp, 'slot_fp': fp, 'slot_fn': fn, 'examples': len(gold_rows)}


def pad_batch(rows, pad_id=PAD_ID, ignore_index=IGNORE_INDEX):
    """Pad input and slot-label sequences within a batch."""
    if not rows:
        raise ValueError('Cannot pad an empty batch')
    longest = max(row['length'] for row in rows)
    return {
        'input_ids': [r['input_ids'] + [pad_id] * (longest - r['length']) for r in rows],
        'attention_mask': [[1] * r['length'] + [0] * (longest - r['length']) for r in rows],
        'slot_ids': [r['slot_ids'] + [ignore_index] * (longest - r['length']) for r in rows],
        'intent_ids': [r['intent_id'] for r in rows],
        'lengths': [r['length'] for r in rows],
        'ids': [r['id'] for r in rows],
    }


def build_dataset(original_records, output_dir='data/processed', tokenizer='word',
                  strict_counts=True, allow_boundary_warnings=False, min_freq=1):
    """Write processed partitions, label maps and validation results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_rows = {p: [] for p in EXPECTED_SPLITS}
    problems = []
    boundary_warnings = []
    for idx, raw in enumerate(original_records):
        part = raw['partition']
        try:
            gold_slots = parse_annotations(raw['utt'], raw['annot_utt'])
            tokens, token_offsets = tokenize_utterance(raw['utt'], tokenizer)
            if not tokens:
                raise ValueError('Utterance has no tokens')
            slot_tags, warnings = spans_to_bio(gold_slots, token_offsets)
            for warning in warnings:
                boundary_warnings.append({'id': raw['id'], 'partition': part, **warning})
            split_rows[part].append({
                'id': raw['id'], 'partition': part, 'utt': raw['utt'],
                'annot_utt': raw['annot_utt'], 'intent': raw['intent'],
                'tokens': tokens, 'token_offsets': token_offsets,
                'gold_slots': gold_slots, 'slot_tags': slot_tags
            })
        except Exception as exc:
            problems.append({'row_index': idx, 'id': raw.get('id'), 'partition': part,
                             'reason': str(exc), 'utt': raw.get('utt'),
                             'annot_utt': raw.get('annot_utt')})
    if problems:
        save_jsonl(out / 'preprocessing_errors.jsonl', problems)
        raise ValueError(f'{len(problems)} record(s) could not be safely processed. '
                         f'See {out / "preprocessing_errors.jsonl"}. No examples have been dropped.')
    if boundary_warnings:
        save_jsonl(out / 'boundary_warnings.jsonl', boundary_warnings)
        if not allow_boundary_warnings:
            raise ValueError(f'{len(boundary_warnings)} slots have boundaries not representable '
                             f'with tokenizer={tokenizer!r}. See {out / "boundary_warnings.jsonl"}. '
                             'Choose tokenizer="char" for guaranteed character boundaries, '
                             'or explicitly inspect and approve --allow-boundary-warnings.')
    counts = {p: len(rows) for p, rows in split_rows.items()}
    if strict_counts and counts != EXPECTED_SPLITS:
        raise ValueError(f'Unexpected official counts: actual={counts}; expected={EXPECTED_SPLITS}. '
                         'Not writing a silently incomplete dataset.')
    vocab, intent2id, slot2id = build_maps(split_rows['train'], min_freq)
    for part, rows in split_rows.items():
        split_rows[part] = [encode_row(row, vocab, intent2id, slot2id) for row in rows]
        for row in split_rows[part]:
            size = len(row['tokens'])
            assert size == len(row['input_ids']) == len(row['slot_ids']) == len(row['slot_tags']) == len(row['token_offsets'])
            assert all(row['utt'][a:b] == tok for tok, (a, b) in zip(row['tokens'], row['token_offsets']))
            assert row['intent_id'] == intent2id[row['intent']]
            assert row['slot_ids'] == [slot2id[tag] for tag in row['slot_tags']]
            assert all(slot['span'] == row['utt'][slot['start']:slot['end']] for slot in row['gold_slots'])
        save_jsonl(out / f'{part}.jsonl', split_rows[part])
    save_json(out / 'vocab.json', vocab)
    save_json(out / 'intent2id.json', intent2id)
    save_json(out / 'slot2id.json', slot2id)
    report = {
        'dataset': 'MASSIVE 1.0 en-US (original splits preserved)',
        'split_counts': counts, 'total_examples': sum(counts.values()),
        'num_intents': len(intent2id), 'num_slot_types': (len(slot2id) - 1) // 2,
        'num_bio_tags': len(slot2id), 'vocab_size': len(vocab),
        'tokenizer': tokenizer, 'lowercase_for_word_ids': True,
        'pad_id': PAD_ID, 'unk_id': UNK_ID, 'slot_ignore_index': IGNORE_INDEX,
        'boundary_warnings': len(boundary_warnings),
        'max_tokens_per_utterance': {p: max(r['length'] for r in rows) for p, rows in split_rows.items()},
        'oov_tokens': {p: sum(tok_id == UNK_ID for r in rows for tok_id in r['input_ids'])
                       for p, rows in split_rows.items()},
        'slots_per_split': {p: sum(len(r['gold_slots']) for r in rows) for p, rows in split_rows.items()},
        'train_only_vocabulary': True, 'train_only_label_maps': True,
        'notes': 'No model was trained. No rows were dropped. Test records are processed, not used for fitting.'
    }
    save_json(out / 'preprocessing_report.json', report)
    print(json.dumps(report, indent=2))
    print(f'\nFinished: {out.resolve()}')
    return report


def main():
    parser = argparse.ArgumentParser(description='MASSIVE 1.0 en-US preprocessing')
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--input', type=Path, help='Extracted 1.0/data/en-US.jsonl')
    sources.add_argument('--archive', type=Path, help='Original 1.0 .tar.gz archive')
    sources.add_argument('--download', action='store_true', help='Download original archive then process')
    parser.add_argument('--output-dir', default='data/processed', help='Destination for processed files')
    parser.add_argument('--tokenizer', choices=['word', 'char'], default='word')
    parser.add_argument('--allow-boundary-warnings', action='store_true',
                        help='Allow non-exact word-token BIO boundaries after inspecting warnings')
    parser.add_argument('--min-freq', type=int, default=1)
    args = parser.parse_args()
    if args.min_freq < 1:
        parser.error('--min-freq must be >= 1')
    if args.download:
        archive = download_official_archive(Path('data/raw/amazon-massive-dataset-1.0.tar.gz'))
        records = read_original_data(archive_path=archive)
    elif args.archive:
        records = read_original_data(archive_path=args.archive)
    else:
        records = read_original_data(input_path=args.input)
    build_dataset(records, output_dir=args.output_dir, tokenizer=args.tokenizer,
                  allow_boundary_warnings=args.allow_boundary_warnings, min_freq=args.min_freq)


if __name__ == '__main__':
    main()
