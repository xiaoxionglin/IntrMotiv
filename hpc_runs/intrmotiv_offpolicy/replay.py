"""Bounded physical sequence replay, including exact washout-prefix checks."""
from collections import OrderedDict
from dataclasses import dataclass
import copy
import numpy as np
import torch
from .contracts import first_arrival


@dataclass(frozen=True)
class Observation:
    preactivation: torch.Tensor
    bypass: torch.Tensor
    events: torch.Tensor

    def owned(self):
        values = (self.preactivation, self.bypass, self.events)
        if any(x.ndim != 1 for x in values) or self.events.dtype != torch.bool:
            raise ValueError('expected one-dimensional feature and boolean event vectors')
        if any(not torch.isfinite(x).all() for x in values):
            raise ValueError('nonfinite replay observation')
        return Observation(*(x.detach().cpu().clone() for x in values))


@dataclass(frozen=True)
class Transition:
    stream: int
    episode: int
    index: int
    observation: Observation
    successor: Observation
    action: int
    goal: int
    option: int
    budget: int
    terminated: bool = False
    truncated: bool = False
    successor_valid: bool = True
    model_version: int = 0
    environment_remaining: int = 1800


class SequenceReplay:
    schema = 'intrmotiv/first-arrival-replay/v1'

    def __init__(self, capacity=200000, width=71, seed=0, reference_hash=''):
        if capacity <= width or width < 1 or not reference_hash:
            raise ValueError('capacity must exceed prefix width; reference hash is required')
        self.capacity, self.width, self.reference_hash = capacity, width, reference_hash
        self.rows = OrderedDict()
        self.last = {}
        self.slots = [None] * capacity
        self.rng = np.random.default_rng(seed)
        self.insertions = 0

    def append(self, row):
        if row.successor_valid != (row.successor is not None):
            raise ValueError('successor validity must match optional physical observation')
        key = (row.stream, row.episode, row.index)
        if key in self.rows or row.index < 0 or row.budget < 1:
            raise ValueError('duplicate transition or invalid index/budget')
        previous = self.last.get(row.stream)
        if previous is None:
            if row.index != 0:
                raise ValueError('stream must begin at physical reset')
        elif previous.episode == row.episode:
            if previous.terminated or previous.truncated or row.index != previous.index + 1:
                raise ValueError('transition crosses reset or skips physical history')
            if previous.successor_valid:
                for a, b in zip(vars(previous.successor).values(), vars(row.observation).values()):
                    if not torch.equal(a, b):
                        raise ValueError('successor/current observation mismatch')
        elif not (previous.terminated or previous.truncated) or row.index != 0 or row.episode <= previous.episode:
            raise ValueError('new episode requires a physical reset and increasing ID')
        row = Transition(**{**vars(row), 'observation': row.observation.owned(), 'successor': row.successor.owned() if row.successor is not None else None})
        if not 0 <= row.goal < len(row.observation.events) or row.action < 0:
            raise ValueError('invalid action/goal')
        self.rows[key] = row
        self.last[row.stream] = row
        self.slots[self.insertions % self.capacity] = key
        self.insertions += 1
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def segment(self, key, unroll=16, horizon=64):
        row = self.rows[key]
        prefix_start = max(0, row.index - self.width)
        prefix_keys = [(row.stream, row.episode, i) for i in range(prefix_start, row.index)]
        if any(k not in self.rows for k in prefix_keys):
            raise ValueError('evicted or missing recurrent prefix')
        prefix = [self.rows[k] for k in prefix_keys]
        if any(not r.successor_valid for r in prefix):
            raise ValueError('invalid recurrent prefix')
        future = []
        for i in range(row.index, row.index + horizon):
            r = self.rows.get((row.stream, row.episode, i))
            if r is None:
                break
            if not r.successor_valid:
                break
            future.append(r)
            if r.terminated or r.truncated:
                break
        if not future:
            raise ValueError('no valid successor')
        return prefix, future[:unroll], future

    def sample(self, registry, her_fraction=0.8, unroll=16, horizon=64):
        if not 0 <= her_fraction <= 1 or unroll < 1 or horizon < 1:
            raise ValueError('invalid sampling configuration')
        count = len(self.rows)
        # Bounded retry; missing history is never replaced by invented zero state.
        for _ in range(min(count, 128)):
            key = self.slots[int(self.rng.integers(count))]
            try:
                prefix, segment, future = self.segment(key, unroll, horizon)
            except ValueError:
                continue
            start = segment[0]
            goal, budget, relabeled = start.goal, start.budget, False
            virtual_budget = min(horizon, start.environment_remaining)
            future_ready = len(future) >= virtual_budget or future[-1].terminated or future[-1].truncated
            if self.rng.random() < her_fraction and future_ready:
                choices = [(offset, g) for offset, r in enumerate(future[:virtual_budget])
                           for g in registry if not bool(start.observation.events[g]) and bool(r.successor.events[g])]
                if choices:
                    _, goal = choices[int(self.rng.integers(len(choices)))]
                    budget, relabeled = virtual_budget, True
            if not relabeled:
                stop = next((i for i,r in enumerate(segment) if r.option != start.option), len(segment))
                segment = segment[:stop]
            observations = [segment[0].observation] + [r.successor for r in segment]
            events = torch.stack([o.events for o in observations])
            reward, done, mask = first_arrival(events, goal, budget,
                torch.tensor([r.terminated for r in segment]), torch.ones(len(segment), dtype=torch.bool))
            if mask.any():
                return dict(prefix=prefix, segment=segment, observations=observations, goal=goal,
                            budget=budget, relabeled=relabeled, reward=reward, done=done, mask=mask)
        raise ValueError('no eligible replay sequence after bounded sampling')

    def state_dict(self):
        return copy.deepcopy(dict(schema=self.schema, capacity=self.capacity, width=self.width,
            reference_hash=self.reference_hash, rows=self.rows, last=self.last, slots=self.slots,
            rng=self.rng.bit_generator.state, insertions=self.insertions))

    def load_state_dict(self, state):
        if (state['schema'], state['capacity'], state['width'], state['reference_hash']) != (
                self.schema, self.capacity, self.width, self.reference_hash):
            raise ValueError('incompatible replay state')
        self.rows, self.last, self.slots = copy.deepcopy((state['rows'], state['last'], state['slots']))
        self.rng.bit_generator.state = state['rng']
        self.insertions = state['insertions']
