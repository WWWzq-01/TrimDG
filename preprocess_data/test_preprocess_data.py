"""Filesystem-only preprocessing CLI regressions; no training or real datasets."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name('preprocess_data.py')


class PreprocessPathsTest(unittest.TestCase):
    def run_cli(self, cwd, *args):
        result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def write_processed(self, directory, dataset):
        directory.mkdir(parents=True)
        pd.DataFrame({'u': [1, 2], 'i': [3, 3], 'ts': [1., 2.],
                      'label': [0., 1.], 'idx': [1, 2]}).to_csv(directory / ('ml_' + dataset + '.csv'))
        np.save(directory / ('ml_' + dataset + '.npy'), np.array([[0., 0.], [1., 2.], [3., 4.]]))
        np.save(directory / ('ml_' + dataset + '_node.npy'), np.zeros((4, 172)))

    def test_copy_branches_use_custom_output(self):
        for dataset in ('enron', 'SocialEvo', 'uci'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cwd = root / 'work'
                cwd.mkdir()
                source = root / 'DG_data' / dataset
                self.write_processed(source, dataset)
                output = root / 'custom' / dataset
                self.run_cli(cwd, '--dataset_name', dataset, '--output-dir', str(output))
                self.assertFalse((root / 'processed_data').exists())
                for path in source.iterdir():
                    self.assertEqual((output / path.name).read_bytes(), path.read_bytes())

    def test_copy_branches_keep_default_output(self):
        for dataset in ('enron', 'SocialEvo', 'uci'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cwd = root / 'work'
                cwd.mkdir()
                source = root / 'DG_data' / dataset
                self.write_processed(source, dataset)
                self.run_cli(cwd, '--dataset_name', dataset)
                for path in source.iterdir():
                    self.assertEqual((root / 'processed_data' / dataset / path.name).read_bytes(), path.read_bytes())

    def test_check_original_uses_custom_processed_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd = root / 'work'
            cwd.mkdir()
            self.write_processed(root / 'DG_data' / 'wikipedia', 'wikipedia')
            raw = root / 'raw.csv'
            raw.write_text('u,i,ts,label,f0,f1\n0,0,1,0,1,2\n1,0,2,1,3,4\n')
            output = root / 'custom' / 'wikipedia'
            result = self.run_cli(cwd, '--dataset_name', 'wikipedia', '--input-csv', str(raw),
                                  '--output-dir', str(output), '--check-original')
            self.assertIn('passes the checks successfully', result.stdout)
            self.assertFalse((root / 'processed_data').exists())
            self.assertEqual(len(list(output.iterdir())), 3)


if __name__ == '__main__':
    unittest.main()
