"""Turn the atlas JSON contract into a fully specified fixed-policy process.

The exported directed edges are authoritative. Missing actions are unavailable,
not invented self-loops. The available action set is explicitly observed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def graph_process_arrays(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Uniform available actions; uniform external event labels when unspecified.

Multiple outcomes of the same action/event label must already sum to one.
Alternative external event labels are merged BEFORE predicting the next input.
"""
    nodes, edges = document['nodes'], document['edges']
    ids = [n['id'] for n in nodes]
    if ids != list(range(len(nodes))):
        raise ValueError('atlas node IDs must be consecutive and ordered')
    outgoing: dict[int, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in edges:
        if e['source'] not in ids or e['target'] not in ids or e['probability'] < 0:
            raise ValueError('invalid edge')
        if nodes[e['source']].get('terminal', False):
            raise ValueError('terminal node has outgoing edges')
        outgoing[e['source']][(e['kind'], e['action'])].append(e)
    observation_keys, action_labels = [], set()
    available = {}
    for node in nodes:
        raw = outgoing[node['id']]
        kinds = {kind for kind, _ in raw}
        if len(kinds) > 1:
            raise ValueError('mixed event/reset/controlled branches need an explicit scheduler')
        labels = sorted({action if kind == 'action' else f'__{kind}__' for kind, action in raw})
        action_labels.update(labels)
        available[node['id']] = labels
        observation_keys.append((node['observation'], tuple(labels)))
        for label, branch in raw.items():
            if not np.isclose(sum(e['probability'] for e in branch), 1, atol=1e-10):
                raise ValueError(f'outcome probabilities do not sum to one at {node["id"]}, {label}')
    actions = sorted(action_labels)
    observations = sorted(set(observation_keys), key=str)
    obs_index = {obs: i for i, obs in enumerate(observations)}
    obs = np.array([obs_index[x] for x in observation_keys], dtype=np.int64)
    action_index = {a: i for i, a in enumerate(actions)}
    source, target, action, probability, rewards = [], [], [], [], []
    for i, branches in outgoing.items():
        for (kind, label), branch in branches.items():
            a = action_index[label if kind == 'action' else f'__{kind}__']
            for edge in branch:
                source.append(i)
                target.append(edge['target'])
                action.append(a)
                probability.append(edge['probability']/len(branches))
                rewards.append(edge['reward'])
    source, target, action = [np.asarray(v, dtype=np.int64) for v in (source, target, action)]
    groups = sorted(set(zip(obs[source].tolist(), action.tolist())))
    group_index = {g: k for k, g in enumerate(groups)}
    group = np.array([group_index[(int(obs[i]), int(a))] for i, a in zip(source, action)])
    supports = [sorted(set(obs[target[group == g]].tolist())) for g in range(len(groups))]
    outcome = np.array([supports[g].index(int(obs[j])) for g, j in zip(group, target)], dtype=np.int64)
    if any('initial_probability' in node for node in nodes):
        initial = np.array([node.get('initial_probability', 0.) for node in nodes], dtype=float)
        if not np.isfinite(initial).all() or (initial < 0).any() or not np.isclose(initial.sum(), 1):
            raise ValueError('explicit initial probabilities must form a distribution')
        initial /= initial.sum()
        initial_assumption = 'explicit initial probabilities'
    else:
        initial = np.array([float(n.get('initial', False)) for n in nodes])
        initial_assumption = 'uniform over exported initial states'
        if initial.sum() == 0:
            initial[:] = 1
            initial_assumption = 'no exported starts: uniform over all nodes (analysis assumption)'
        initial /= initial.sum()
    scale = max(1., max(abs(r) for r in rewards))
    arrays = dict(source=source, target=target, action=action, probability=np.array(probability),
                  observation=obs, initial=initial, group=group, outcome=outcome,
                  reward=np.array(rewards)/scale, n_actions=len(actions), n_observations=len(observations),
                  n_groups=len(groups), n_outcomes=max(map(len, supports)))
    metadata = dict(actions=actions, observations=[list(map(str, o)) for o in observations],
                    groups=groups, outcome_observations=supports, reward_scale=scale,
                    initial_assumption=initial_assumption,
                    policy='uniform over exported available action labels',
                    event_assumption='uniform over distinct external event labels, not controllable',
                    observation_assumption='exported observation plus available action/event mask',
                    reset_assumption='exported reset edges preserve neural memory; terminal paths stop')
    return arrays, metadata
