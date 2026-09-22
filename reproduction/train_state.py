"""Bounded Wikipedia TGN train/state study; no evaluator or quality metrics.

This imports the published MemoryModel, graph attention and MergeLayer. It keeps
negative-before-positive forwards, BCE/Adam, epoch memory reset, batch selector
refresh and memory detach from trim_link.py. It does not run validation/test.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.MemoryModel import MemoryModel, GraphAttentionEmbedding, compute_src_dst_node_time_shifts
from models.modules import MergeLayer
from utils.batch_selector import select_batch_indices
from utils.DataLoader import get_link_prediction_data, get_idx_data_loader
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, set_random_seed, create_optimizer

U0 = 'cceffe826df3b8712c646ca9f46def5ea0f079e3'
CONFIG = dict(model='TGN', dataset='wikipedia', sample_neighbor_strategy='recent',
              num_layers=1, num_neighbors=10, num_heads=2, time_feat_dim=100,
              batch_size=200, learning_rate=1e-4, dropout=0.1, seed=0,
              cache=0, bypass=0.0, batch_rate=0.5, optimizer='Adam', weight_decay=0.0,
              val_ratio=0.15, test_ratio=0.15, save_pkl=0, dataset_name='wikipedia',
              our=False, GIB=False, pre_pruning=False, BaM=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ledger_append(path, event, **fields):
    """The counted batch start is persisted before any training in that batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        records = [json.loads(line) for line in stream if line.strip()]
        count = sum(row.get('event') == 'batch_start' for row in records)
        if event == 'batch_start' and count >= 512:
            raise RuntimeError('Shared 512 training batch budget exhausted (failed starts count)')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(dict(event=event, time=time.time(), pid=os.getpid(), **fields)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        return count + (event == 'batch_start')


def make_model(node_features, edge_features, train_data, sampler, device):
    shifts = compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                             train_data.node_interact_times)
    backbone = MemoryModel(node_features, edge_features, sampler, time_feat_dim=100,
                           model_name='TGN', num_layers=1, num_heads=2, dropout=0.1,
                           src_node_mean_time_shift=shifts[0], src_node_std_time_shift=shifts[1],
                           dst_node_mean_time_shift_dst=shifts[2], dst_node_std_time_shift=shifts[3],
                           device=device)
    assert isinstance(backbone.embedding_module, GraphAttentionEmbedding)
    predictor = MergeLayer(node_features.shape[1], node_features.shape[1], node_features.shape[1], 1)
    return nn.Sequential(backbone, predictor).to(device)


class Observations:
    """Hooks inspect actual gathers, sampler returns, cloned views and messages."""
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler
        self.counts = Counter()
        self.examples = []
        self.phase = 'unset'
        self.published = set()
        self.generated = {}
        self.batch_context = {}
        self.batch_messages = []
        self.skipped_epoch_ids = np.array([], dtype=np.int64)
        self.sampled_skipped_examples = []
        self._attach()

    def gather(self, kind, ids, result):
        # Called immediately after the actual tensor indexing in GraphAttentionEmbedding.
        self.counts[kind + '_calls'] += 1
        self.counts[kind + '_rows'] += int(ids.size)
        self.counts[kind + '_padding_rows'] += int(np.count_nonzero(ids == 0))
        # node_memory_gather does two independent row gathers of this shape.
        self.counts[kind + '_logical_bytes'] += result.numel() * result.element_size() * (2 if kind == 'node_memory_gather' else 1)

    def _attach(self):
        self.model.embedding_module.reproduction_observer = self.gather
        original_sample = self.sampler.get_historical_neighbors
        def sample(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            self.counts['sampler_calls'] += 1
            self.counts['sampler_query_rows'] += result[0].shape[0]
            self.counts['sampler_slots'] += result[1].size
            self.counts['sampler_nonpadding_slots'] += int(np.count_nonzero(result[1]))
            hits = np.argwhere(np.isin(result[1], self.skipped_epoch_ids))
            self.counts['sampler_slots_from_skipped_events'] += len(hits)
            for row, col in hits[:max(0, 6 - len(self.sampled_skipped_examples))]:
                self.sampled_skipped_examples.append(dict(
                    **self.batch_context, phase=self.phase, edge_id=int(result[1][row, col]),
                    query_node=int(kwargs['node_ids'][row]),
                    query_cutoff=float(kwargs['node_interact_times'][row]),
                    sampled_event_time=float(result[2][row, col])))
            return result
        self.sampler.get_historical_neighbors = sample
        original_clone = self.model.memory_updater.get_updated_memories
        def clone(*args, **kwargs):
            result = original_clone(*args, **kwargs)
            bank = self.model.memory_bank
            assert result[0].data_ptr() != bank.node_memories.data_ptr()
            assert result[1].data_ptr() != bank.node_last_updated_times.data_ptr()
            self.counts['full_memory_clone_calls'] += 1
            self.counts['full_memory_clone_rows'] += result[0].shape[0]
            self.counts['full_memory_clone_logical_bytes'] += sum(t.numel() * t.element_size() for t in result)
            return result
        self.model.memory_updater.get_updated_memories = clone
        original_messages = self.model.compute_new_node_raw_messages
        def messages(*args, **kwargs):
            result = original_messages(*args, **kwargs)
            assert self.phase == 'positive', 'Negative forward published messages'
            events = [dict(edge_id=int(e), recipient=int(n), other_node=int(d), time=float(t))
                      for e, n, d, t in zip(kwargs['edge_ids'], kwargs['src_node_ids'],
                                           kwargs['dst_node_ids'], kwargs['node_interact_times'])]
            self.generated[id(result[1])] = events
            self.counts['raw_message_generation_calls'] += 1
            self.counts['raw_messages_generated'] += len(events)
            # This is the actual edge feature gather in compute_new_node_raw_messages.
            self.counts['message_edge_gather_rows'] += len(events)
            return result
        self.model.compute_new_node_raw_messages = messages
        original_store = self.model.memory_bank.store_node_raw_messages
        def store(*args, **kwargs):
            events = self.generated.pop(id(kwargs['new_node_raw_messages']))
            original_store(*args, **kwargs)
            count = sum(len(kwargs['new_node_raw_messages'][n]) for n in kwargs['node_ids'])
            assert count == len(events)
            self.counts['raw_messages_stored'] += count
            self.published.update(event['edge_id'] for event in events)
            if len(self.batch_messages) < 8:
                self.batch_messages.extend(events[:8 - len(self.batch_messages)])
        self.model.memory_bank.store_node_raw_messages = store
        original_detach = self.model.memory_bank.detach_memory_bank
        def detach():
            self.counts['detach_dictionary_nodes_visited'] += len(self.model.memory_bank.node_raw_messages)
            self.counts['detach_messages_visited'] += sum(len(v) for v in self.model.memory_bank.node_raw_messages.values())
            original_detach()
            self.counts['detach_calls'] += 1
        self.model.memory_bank.detach_memory_bank = detach

    def begin(self, epoch, batch):
        self.batch_context = dict(epoch=epoch, batch=batch)
        self.published = set()
        self.batch_messages = []
        self.batch_negatives = []

    def finish(self, train, original, selected):
        selected_ids = set(map(int, train.edge_ids[selected]))
        if self.published != selected_ids:
            raise AssertionError('Published edge identities differ from selected positive events')
        skipped = np.setdiff1d(original, selected, assume_unique=True)
        self.counts['events_without_new_messages'] += len(skipped)
        if self.batch_context['batch'] not in (0, 1, 127):
            return
        probes = []
        for idx in skipped[:3]:
            src, edge = int(train.src_node_ids[idx]), int(train.edge_ids[idx])
            next_pos = np.searchsorted(train.node_interact_times, train.node_interact_times[idx], side='right')
            cutoff = float(train.node_interact_times[next_pos]) if next_pos < len(train.edge_ids) else float(np.nextafter(train.node_interact_times[idx], np.inf))
            _, edge_ids, _, _ = self.sampler.find_neighbors_before(src, cutoff)
            present = bool(np.any(edge_ids == edge))
            assert present and edge not in self.published
            probes.append(dict(edge_id=edge, src=src, event_time=float(train.node_interact_times[idx]),
                               query_cutoff=cutoff, in_full_train_adjacency=present,
                               published_new_message=False, probe='find_neighbors_before (outside training sampler counters)'))
        self.examples.append(dict(**self.batch_context,
                                  original_edge_ids=train.edge_ids[original[:8]].tolist(),
                                  selected_edge_ids=train.edge_ids[selected[:8]].tolist(),
                                  selected_count=len(selected), skipped_count=len(skipped),
                                  new_messages=self.batch_messages, negative_destinations=self.batch_negatives,
                                  skipped_adjacency_probes=probes))


def train_batch(model, optimizer, train, indices, negative_sampler, observations=None, on_optimizer_step=None, on_probabilities=None):
    src, dst = train.src_node_ids[indices], train.dst_node_ids[indices]
    times, edges = train.node_interact_times[indices], train.edge_ids[indices]
    # No re-seeding or separate negative trajectory for T1: same original sampler semantics.
    _, neg_dst = negative_sampler.sample(size=len(src))
    if observations:
        observations.phase = 'negative'
        observations.counts['negative_forward_calls'] += 1
        observations.counts['negative_targets'] += len(src)
        observations.batch_negatives = neg_dst[:8].tolist()
    neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
        src, neg_dst, times, edge_ids=None, edges_are_positive=False, num_neighbors=10)
    if observations:
        observations.phase = 'positive'
        observations.counts['positive_forward_calls'] += 1
        observations.counts['positive_targets'] += len(src)
    src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
        src, dst, times, edge_ids=edges, edges_are_positive=True, num_neighbors=10)
    positive = model[1](src_emb, dst_emb).squeeze(-1).sigmoid()
    negative = model[1](neg_src_emb, neg_dst_emb).squeeze(-1).sigmoid()
    predicts = torch.cat([positive, negative])
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
    if not torch.isfinite(predicts).all():
        raise ValueError('Non-finite model probabilities')
    loss = nn.BCELoss()(predicts, labels)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite loss')
    if on_probabilities is not None:
        on_probabilities(positive)
    optimizer.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError('Non-finite gradient')
    optimizer.step()
    if on_optimizer_step is not None:
        on_optimizer_step(float(loss.item()))
    model[0].memory_bank.detach_memory_bank()
    if torch.device(model[0].device).type == 'cuda':
        torch.cuda.empty_cache()
    return float(loss.item()), positive.detach()


def check_state(model):
    bank = model[0].memory_bank
    tensors = [bank.node_memories, bank.node_last_updated_times]
    messages = [message for values in bank.node_raw_messages.values() for message, _ in values]
    tensors.extend(messages)
    assert all(bool(torch.isfinite(t).all()) for t in tensors), 'Non-finite model state'
    assert all(not t.requires_grad for t in messages), 'Raw message detach missing'
    assert all(np.isfinite(timestamp) for values in bank.node_raw_messages.values() for _, timestamp in values), 'Non-finite raw message timestamp'
    return dict(finite=True, memory_shape=list(bank.node_memories.shape),
                last_updated_time_max=float(bank.node_last_updated_times.max().item()),
                pending_messages=len(messages), raw_message_nodes=len(bank.node_raw_messages),
                memory_norm=float(bank.node_memories.norm().item()))


def execute(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('summary.json', 'epoch_metrics.csv', 'observations.json')):
        raise FileExistsError('Refusing to overwrite an existing attempt')
    started = time.monotonic()
    epochs = []
    summary = dict(system='TrimDG', model='TGN', dataset='wikipedia', benchmark_track='native_system',
                   experiment_group='study', seed=0, variant=args.variant, evaluation_scope='train_state_only',
                   source={'u0': U0, 'u1': args.source_u1, 'u2': args.source_u2,
                           'checkout': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()},
                   status='running', epochs_requested=2, completed_epochs=0, optimizer_steps=0, batches_started=0,
                   config=dict(CONFIG, batch_sampling=int(args.variant == 'T1'), max_batches=args.max_batches),
                   environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                    device=args.device), quality_evaluation='not_measured',
                   timing_scope='instrumented training loop including selector, checks, optimizer and durable ledger; not official performance',
                   not_run=['validation', 'test', 'AP', 'AUC', 'full_epochs', 'E5_selector_refresh'])
    observation = None
    gpu_started = None
    error = None
    ledger_append(args.ledger, 'run_start', variant=args.variant)
    try:
        # Exactly the original loader, including its inductive train split and feature padding.
        loaded = get_link_prediction_data('wikipedia', 0.15, 0.15, data_root=args.data_root)
        nodes, edges, full, train = loaded[:4]
        if any(not np.isfinite(a).all() for a in (nodes, edges, full.node_interact_times, train.node_interact_times)):
            raise ValueError('Non-finite input')
        if np.any(np.diff(full.node_interact_times) < 0) or np.any(np.diff(train.node_interact_times) < 0):
            raise ValueError('Events are not chronologically ordered')
        if len(np.unique(full.edge_ids)) != len(full.edge_ids) or np.any(np.diff(train.edge_ids) <= 0):
            raise ValueError('Non-unique or unordered event ids')
        paths = sorted((Path(args.data_root) / 'processed_data/wikipedia').glob('ml_wikipedia*'))
        summary['data'] = dict(source_dataset_identity='JODIE Wikipedia link prediction (not Wiki-Talk)',
                               files={p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in paths if p.is_file()},
                               full_events=len(full.edge_ids), train_events=len(train.edge_ids),
                               train_edge_ids_sha256=array_hash(train.edge_ids),
                               train_times_sha256=array_hash(train.node_interact_times),
                               split='original loader quantile 0.70/0.85 and random.seed(2020) inductive node holdout',
                               index_scope='full train split; both epochs retain identical adjacency',
                               node_features_shape=list(nodes.shape), edge_features_shape=list(edges.shape),
                               finite=True, chronological=True)
        sampler = get_neighbor_sampler(train, sample_neighbor_strategy='recent', seed=0, args=SimpleNamespace(**CONFIG))
        summary['data']['adjacency_directed_entries'] = int(sum(len(a) for a in sampler.nodes_edge_ids))
        assert summary['data']['adjacency_directed_entries'] == 2 * len(train.edge_ids)
        summary['data']['adjacency_sha256'] = hashlib.sha256(b''.join(np.asarray(a, dtype=np.int64).tobytes() for a in sampler.nodes_edge_ids)).hexdigest()
        # Match author: random seed is reset after constructing the samplers.
        negatives = NegativeEdgeSampler(train.src_node_ids, train.dst_node_ids)
        loader = get_idx_data_loader(list(range(len(train.edge_ids))), 200, shuffle=False)
        prefix_batches = min(args.max_batches, len(loader))
        summary['prefix_batches'] = prefix_batches
        summary['data']['prefix_events'] = min(prefix_batches * 200, len(train.edge_ids))
        set_random_seed(0)
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA is unavailable; CPU substitution is forbidden')
            gpu_started = time.monotonic()
            def timeout(_signum, _frame):
                raise TimeoutError('Per-process remaining GPU time budget exhausted')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.gpu_seconds_budget)
        model = make_model(nodes, edges, train, sampler, args.device)
        summary['dimensions'] = dict(node=model[0].node_feat_dim, edge=model[0].edge_feat_dim,
                                     memory=model[0].memory_dim, embedding=model[0].node_feat_dim,
                                     time=model[0].time_feat_dim, message=model[0].message_dim)
        summary['resident_feature_bytes'] = sum(t.numel() * t.element_size() for t in (model[0].node_raw_features, model[0].edge_raw_features))
        optimizer = create_optimizer(model, 'Adam', 1e-4, 0.0)
        observation = Observations(model[0], sampler)
        selectors = {}
        for epoch in range(2):
            model.train()
            model[0].set_neighbor_sampler(sampler)
            model[0].memory_bank.__init_memory_bank__()
            assert not model[0].memory_bank.node_raw_messages
            assert not torch.count_nonzero(model[0].memory_bank.node_memories)
            assert not torch.count_nonzero(model[0].memory_bank.node_last_updated_times)
            reset_state = dict(memory_all_zero=True, last_updated_all_zero=True, pending_messages=0,
                               selector_table_batches_before=len(selectors))
            if args.variant == 'T1' and epoch == 1:
                skipped = []
                for batch_idx in range(prefix_batches):
                    original = np.arange(batch_idx * 200, min((batch_idx + 1) * 200, len(train.edge_ids)))
                    skipped.extend(train.edge_ids[np.setdiff1d(original, original[selectors[batch_idx]])])
                observation.skipped_epoch_ids = np.asarray(skipped, dtype=np.int64)
            initial_counts = observation.counts.copy()
            epoch_started = time.monotonic()
            row = dict(epoch=epoch, train_loss=None, train_seconds=None, train_batch_count=0,
                       train_event_count=0, optimizer_steps=0, reset_state=reset_state)
            losses = []
            epochs.append(row)
            for batch, original in enumerate(loader):
                if batch >= prefix_batches:
                    break
                original = original.numpy()
                indices = original if args.variant == 'T0' or epoch == 0 else original[selectors[batch]]
                # U2 is needed for this assertion; U1 records the original unsorted selector.
                if np.any(np.diff(train.node_interact_times[indices]) < 0):
                    raise ValueError('Selected input is not in event order')
                ledger_append(args.ledger, 'batch_start', variant=args.variant, epoch=epoch, batch=batch,
                              positive_events=len(indices))
                summary['batches_started'] += 1
                observation.begin(epoch, batch)
                def optimizer_completed(loss):
                    summary['optimizer_steps'] += 1
                    row['optimizer_steps'] += 1
                    ledger_append(args.ledger, 'batch_end', variant=args.variant, epoch=epoch, batch=batch,
                                  loss=loss, optimizer_steps=summary['optimizer_steps'])
                def select_probabilities(positive):
                    if args.variant == 'T1' and epoch % 5 == 0:
                        selectors[batch] = select_batch_indices(positive, 0.5)
                loss, positive = train_batch(model, optimizer, train, indices, negatives, observation,
                                             on_optimizer_step=optimizer_completed,
                                             on_probabilities=select_probabilities)
                observation.finish(train, original, indices)
                losses.append(loss)
                row['train_batch_count'] += 1
                row['train_event_count'] += len(indices)
                row['train_loss'] = float(np.mean(losses))
                row['train_seconds'] = time.monotonic() - epoch_started
                if batch in (0, 1, 127):
                    print(json.dumps(dict(event='batch_observation', variant=args.variant, epoch=epoch,
                                          batch=batch, positive_events=len(indices), loss=loss,
                                          state=check_state(model))), flush=True)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            row['train_seconds'] = time.monotonic() - epoch_started
            row['state'] = check_state(model)
            row['work'] = dict(observation.counts - initial_counts)
            row['selector_table_batches'] = len(selectors)
            summary['completed_epochs'] += 1
            print(json.dumps(dict(event='epoch_end', variant=args.variant, **row)), flush=True)
        summary['status'] = 'complete'
        summary['final_state'] = check_state(model)
        summary['selector_table_sha256'] = hashlib.sha256(b''.join(selectors[k].tobytes() for k in sorted(selectors))).hexdigest()
        if args.device.startswith('cuda'):
            summary['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated(args.device))
            summary['peak_cuda_reserved_bytes'] = int(torch.cuda.max_memory_reserved(args.device))
    except BaseException as exc:
        summary['status'] = 'failed'
        summary['error'] = '{}: {}'.format(type(exc).__name__, str(exc))
        error = exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        summary['elapsed_seconds'] = time.monotonic() - started
        summary['gpu_seconds'] = time.monotonic() - gpu_started if gpu_started is not None else 0.0
        ledger_append(args.ledger, 'run_end', variant=args.variant, status=summary['status'],
                      elapsed_seconds=summary['elapsed_seconds'], gpu_seconds=summary['gpu_seconds'],
                      batches_started=summary['batches_started'], optimizer_steps=summary['optimizer_steps'])
        json_write(output / 'summary.json', summary)
        fields = ['epoch', 'train_loss', 'train_seconds', 'train_batch_count', 'train_event_count', 'optimizer_steps']
        with (output / 'epoch_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(epochs)
        json_write(output / 'observations.json', dict(
            counts=dict(observation.counts) if observation else {}, epochs=epochs,
            examples=observation.examples if observation else [],
            actual_training_samples_of_skipped_events=observation.sampled_skipped_examples if observation else [],
            interpretation='Logical sampled/gathered rows and clone work only; no physical I/O inference. Message absence is checked by original edge id, not recipient state.',
            selector='Positive BCE probability entropy top-k; E0 refresh, E1 consume; table retained across memory reset',
            instrumentation='Actual graph feature indexing callback; sampler return, memory clone, raw-message generation/store and detach wrappers'))
    if error is not None:
        raise error
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--variant', required=True, choices=['T0', 'T1'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-batches', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2, choices=[2])
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--gpu-seconds-budget', type=float, required=True)
    parser.add_argument('--source-u1', required=True)
    parser.add_argument('--source-u2', required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_batches <= 128 or not 0 < args.gpu_seconds_budget <= 5400:
        parser.error('max-batches must be 1..128 and GPU budget must be (0,5400] seconds')
    if args.device != 'cpu' and not args.device.startswith('cuda:'):
        parser.error('device must explicitly be cpu or cuda:N')
    return args


if __name__ == '__main__':
    execute(parse_args())
