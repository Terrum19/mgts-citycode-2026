import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline import parser, prepare


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = self.root / 'labels.json'
        self.labels.write_text('[]')
        self.cfg = {'variants': {'det_dino': {'kind': 'det_dino', 'checkpoint': str(self.root/'crop.joblib')},
                                 'photo_dino': {'kind': 'photo_dino', 'checkpoint': str(self.root/'photo_dino.joblib')},
                                 'qwen235': {}},
                    'datasets': {'test': {'labels': str(self.labels), 'split': 'test'}}, 'detector': {}}

    def args(self, *args):
        return parser().parse_args(list(args))

    def test_missing_model_requires_training(self):
        with self.assertRaisesRegex(ValueError, 'Нет модели'):
            prepare(self.args('--images', str(self.root)), self.cfg)

    def test_if_missing_and_split_guard(self):
        args = self.args('--dataset', 'test', '--train', 'if-missing', '--annotations', str(self.labels),
                         '--train-images', str(self.root), '--split', 'dev')
        self.assertTrue(prepare(args, self.cfg)[-1])
        args.split = 'test'
        with self.assertRaisesRegex(ValueError, 'Обучение и оценка'):
            prepare(args, self.cfg)
        args.split = 'dev'
        (self.root/'crop.joblib').touch()
        self.assertFalse(prepare(args, self.cfg)[-1])

    def test_vlm_rejects_training(self):
        with self.assertRaisesRegex(ValueError, 'не поддерживает обучение'):
            prepare(self.args('--variant', 'qwen235', '--images', str(self.root), '--train', 'always'), self.cfg)

    def test_tail_only_for_photo(self):
        with self.assertRaisesRegex(ValueError, 'только к photo'):
            prepare(self.args('--images', str(self.root), '--training', 'tail'), self.cfg)

    def test_final_trains_and_keeps_config_unchanged(self):
        args = self.args('--variant', 'final', '--train-images', str(self.root), '--images', str(self.root),
                         '--training', 'tail', '--out', str(self.root/'out'))
        cfg, _, _, checkpoint, train = prepare(args, self.cfg)
        self.assertTrue(train)
        self.assertEqual(checkpoint, (self.root/'out/photo_dino.joblib').resolve())
        self.assertNotEqual(cfg['variants']['photo_dino'], self.cfg['variants']['photo_dino'])


if __name__ == '__main__':
    unittest.main()
