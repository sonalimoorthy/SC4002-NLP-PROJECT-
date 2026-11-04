"""Unit tests for parsing, alignment, encoding and scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from preprocessing import (parse_annotations, tokenize_utterance, spans_to_bio,
                           build_dataset, pad_batch, evaluate_exact, bio_to_spans)


class TestPreprocessing(unittest.TestCase):
    def test_official_readme_example(self):
        utt = 'wake me up at five am this week'
        annot = 'wake me up at [time : five am] [date : this week]'
        spans = parse_annotations(utt, annot)
        self.assertEqual([(s['type'], s['span']) for s in spans],
                         [('time', 'five am'), ('date', 'this week')])
        tokens, offsets = tokenize_utterance(utt)
        tags, warnings = spans_to_bio(spans, offsets)
        self.assertEqual(tags, ['O', 'O', 'O', 'O', 'B-time', 'I-time', 'B-date', 'I-date'])
        self.assertEqual(warnings, [])
        self.assertEqual(bio_to_spans(utt, offsets, tags), spans)

    def test_repeated_text_uses_offsets(self):
        utt = 'play blue then blue'
        annot = 'play [color : blue] then blue'
        spans = parse_annotations(utt, annot)
        self.assertEqual(utt[spans[0]['start']:spans[0]['end']], 'blue')
        self.assertEqual(spans[0]['start'], 5)

    def test_whitespace_only_difference(self):
        utt = 'set  alarm\tfor seven'
        annot = 'set alarm for [time : seven]'
        self.assertEqual(parse_annotations(utt, annot)[0]['span'], 'seven')

    def test_no_slot(self):
        utt = 'hello there!'
        self.assertEqual(parse_annotations(utt, utt), [])
        tokens, offsets = tokenize_utterance(utt)
        self.assertEqual(tokens, ['hello', 'there', '!'])
        self.assertEqual(spans_to_bio([], offsets), (['O'] * 3, []))

    def test_partial_token_requires_warning(self):
        utt = 'play abcdef'
        spans = parse_annotations(utt, 'play ab[part : cd]ef')
        toks, offsets = tokenize_utterance(utt)
        tags, warnings = spans_to_bio(spans, offsets)
        self.assertEqual(tags, ['O', 'B-part'])
        self.assertTrue(warnings)
        chars, char_offsets = tokenize_utterance(utt, mode='char')
        char_tags, char_warnings = spans_to_bio(spans, char_offsets)
        self.assertFalse(char_warnings)
        self.assertEqual(bio_to_spans(utt, char_offsets, char_tags), spans)

    def test_non_whitespace_mismatch_raises(self):
        with self.assertRaises(ValueError):
            parse_annotations('what time', 'what [time : date]')

    def test_build_and_exact_evaluator(self):
        def record(i, part, utt, annot, intent):
            return {'id': str(i), 'locale': 'en-US', 'partition': part,
                    'utt': utt, 'annot_utt': annot, 'intent': intent}
        dataset = [record(1, 'train', 'set at five', 'set at [time : five]', 'alarm_set'),
                   record(2, 'train', 'play music', 'play music', 'music_play'),
                   record(3, 'dev', 'set at six', 'set at [time : six]', 'alarm_set'),
                   record(4, 'test', 'play music', 'play music', 'music_play')]
        with tempfile.TemporaryDirectory() as tmp:
            report = build_dataset(dataset, output_dir=tmp, strict_counts=False)
            self.assertEqual(report['split_counts'], {'train': 2, 'dev': 1, 'test': 1})
            with open(Path(tmp) / 'dev.jsonl', encoding='utf-8') as f:
                dev = [json.loads(line) for line in f]
            with open(Path(tmp) / 'slot2id.json', encoding='utf-8') as f:
                slot_map = json.load(f)
            self.assertEqual(dev[0]['input_ids'][-1], 1)
            perfect = evaluate_exact(dev, [dev[0]['intent_id']], [dev[0]['slot_ids']], slot_map)
            self.assertEqual(perfect['slot_micro_f1'], 100.0)
            self.assertEqual(perfect['intent_accuracy'], 100.0)
            wrong = evaluate_exact(dev, [0 if dev[0]['intent_id'] != 0 else 1], [[0]*dev[0]['length']], slot_map)
            self.assertEqual(wrong['slot_micro_f1'], 0.0)
            padded = pad_batch(dev)
            self.assertEqual(padded['input_ids'], [dev[0]['input_ids']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
