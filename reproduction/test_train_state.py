"""Small CPU tests only; synthetic fixtures are never study datasets."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from models.MemoryModel import MessageAggregator
from reproduction.train_state import CONFIG, Observations, ledger_append, make_model, train_batch, check_state, execute, parse_args
from utils.batch_selector import select_batch_indices
from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, create_optimizer, set_random_seed


class TrainStateTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        set_random_seed(0)
        self.data = Data(np.array([1, 1, 2, 1, 2, 1]), np.array([3, 4, 3, 3, 4, 4]),
                         np.arange(1, 7, dtype=np.float64), np.arange(1, 7), np.zeros(6))
        self.sampler = get_neighbor_sampler(self.data, sample_neighbor_strategy='recent', seed=0,
                                            args=SimpleNamespace(**CONFIG))
        # Finite deterministic test fixture; not used for Wikipedia or reported metrics.
        self.model = make_model(np.zeros((5, 172)), np.ones((7, 172)), self.data, self.sampler, 'cpu')

    def test_real_tgn_forward_backward_and_state(self):
        observation = Observations(self.model[0], self.sampler)
        optimizer = create_optimizer(self.model, 'Adam', 1e-4)
        negatives = NegativeEdgeSampler(self.data.src_node_ids, self.data.dst_node_ids)
        before = next(self.model[1].parameters()).detach().clone()
        for batch, idx in enumerate((np.array([0, 1, 2]), np.array([3, 4, 5]))):
            observation.begin(0, batch)
            loss, probabilities = train_batch(self.model, optimizer, self.data, idx, negatives, observation)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(len(probabilities), 3)
            observation.finish(self.data, idx, idx)
            check_state(self.model)
        self.assertFalse(torch.equal(before, next(self.model[1].parameters())))
        self.assertEqual(observation.counts['raw_messages_stored'], 12)
        self.assertEqual(observation.counts['full_memory_clone_calls'], 4)
        self.assertEqual(observation.counts['node_memory_gather_rows'], 288)
        self.assertEqual(observation.counts['neighbor_edge_gather_rows'], 240)
        self.assertEqual(observation.counts['detach_calls'], 2)

    def test_event_absence_does_not_remove_adjacency(self):
        observation = Observations(self.model[0], self.sampler)
        optimizer = create_optimizer(self.model, 'Adam', 1e-4)
        negatives = NegativeEdgeSampler(self.data.src_node_ids, self.data.dst_node_ids)
        observation.begin(1, 0)
        selected = np.array([0, 2, 5])
        train_batch(self.model, optimizer, self.data, selected, negatives, observation)
        observation.finish(self.data, np.arange(6), selected)
        probes = observation.examples[0]['skipped_adjacency_probes']
        self.assertEqual([p['edge_id'] for p in probes], [2, 4, 5])
        self.assertTrue(all(p['in_full_train_adjacency'] and not p['published_new_message'] for p in probes))
        # Node 1 received other retained events; that says nothing about dropped edge 2.
        self.assertGreater(len(self.model[0].memory_bank.node_raw_messages[1]), 0)
        self.assertNotIn(2, observation.published)

    def test_u2_preserves_topk_set_and_restores_last_message(self):
        probabilities = torch.tensor([0.1, 0.4, 0.9, 0.5])
        entropy = -(probabilities * torch.log(probabilities) + (1 - probabilities) * torch.log(1 - probabilities))
        indices = torch.topk(entropy, k=2).indices.numpy()
        corrected = select_batch_indices(probabilities, 0.5)
        self.assertEqual(indices.tolist(), [3, 1])
        self.assertEqual(corrected.tolist(), [1, 3])
        self.assertEqual(set(indices), set(corrected))
        times = np.array([1., 2., 3., 4.])
        src = np.array([1, 1, 1, 1])[indices]
        dst = np.array([3, 4, 3, 4])[indices]
        _, messages = self.model[0].compute_new_node_raw_messages(
            src, dst, torch.zeros((2, 172)), times[indices], np.arange(1, 5)[indices])
        _, _, last_times = MessageAggregator().aggregate_messages(src, messages)
        self.assertEqual(last_times.tolist(), [2.])
        self.assertEqual(max(times[indices]), 4.)
        _, corrected_messages = self.model[0].compute_new_node_raw_messages(
            np.array([1, 1]), np.array([4, 4]), torch.zeros((2, 172)),
            times[corrected], np.arange(1, 5)[corrected])
        _, _, corrected_last = MessageAggregator().aggregate_messages(np.array([1, 1]), corrected_messages)
        self.assertEqual(corrected_last.tolist(), [4.])

    def test_recent_does_not_execute_unused_presampling(self):
        with patch('utils.utils.sample_window_size', side_effect=AssertionError('unused our branch')):
            get_neighbor_sampler(self.data, sample_neighbor_strategy='recent', seed=0,
                                 args=SimpleNamespace(**CONFIG))

    def test_driver_two_epoch_reset_on_tiny_original_loader(self):
        from preprocess_data.preprocess_data import preprocess_data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw.csv'
            raw.write_text('u,i,ts,label,f0,f1\n' + ''.join(
                '{},{},{},0,1.0,2.0\n'.format(i % 2, (i // 2) % 2, i + 1) for i in range(12)))
            preprocess_data('wikipedia', input_csv=raw, output_dir=root / 'processed_data/wikipedia')
            for variant in ('T0', 'T1'):
                args = parse_args(['--data-root', str(root), '--output-dir', str(root / variant),
                                   '--variant', variant, '--device', 'cpu', '--max-batches', '1',
                                   '--ledger', str(root / (variant + '-ledger')), '--gpu-seconds-budget', '30',
                                   '--source-u1', 'test-u1', '--source-u2', 'test-u2'])
                result = execute(args)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['completed_epochs'], 2)
                self.assertEqual(result['optimizer_steps'], 2)
                observed = json.loads((root / variant / 'observations.json').read_text())
                self.assertEqual(observed['counts']['full_memory_clone_calls'], 4)
                train_events = result['data']['train_events']
                self.assertEqual(observed['epochs'][0]['train_event_count'], train_events)
                self.assertEqual(observed['epochs'][1]['train_event_count'], train_events if variant == 'T0' else train_events // 2)
                self.assertEqual(observed['epochs'][1]['reset_state']['selector_table_batches_before'], int(variant == 'T1'))
                if variant == 'T1':
                    self.assertGreater(observed['counts']['sampler_slots_from_skipped_events'], 0)
                    self.assertTrue(observed['actual_training_samples_of_skipped_events'])
                    self.assertEqual(observed['counts']['raw_messages_stored'], 3 * train_events)
                else:
                    self.assertEqual(observed['counts']['raw_messages_stored'], 4 * train_events)

    def test_failed_batch_start_counts_towards_shared_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.jsonl'
            path.write_text(''.join(json.dumps({'event': 'batch_start'}) + '\n' for _ in range(511)))
            self.assertEqual(ledger_append(path, 'batch_start', variant='T1', epoch=1, batch=0), 512)
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                ledger_append(path, 'batch_start')
            self.assertEqual(len(path.read_text().splitlines()), 512)


if __name__ == '__main__':
    unittest.main()
