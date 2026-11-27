"""Explicit task graphs and Pelle's unit-conductance resistance curvature.

Optional dependencies: numpy, scipy, networkx; MiniGrid/Gymnasium for extraction.
Imports stay outside the base geomem_env API so its dependency-free uses still work.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import pinvh


@dataclass
class TaskGraph:
    """Action-labelled directed graph, with a separate observation on each node."""

    name: str
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    metadata: dict[str, Any] = field(default_factory=dict)

    def node(self, key: Any, *, observation: str, pos: tuple[float, float],
             label: str = "", initial: bool = False, success: bool = False,
             terminal: bool = False, **attributes: Any) -> None:
        self.graph.add_node(key, observation=observation, pos=list(pos),
                            label=label or str(key), initial=initial,
                            success=success, terminal=terminal, **attributes)

    def edge(self, source: Any, target: Any, action: str, *, reward: float = 0,
             probability: float = 1, kind: str = "action", **attributes: Any) -> None:
        self.graph.add_edge(source, target, action=action, reward=float(reward),
                            probability=float(probability), kind=kind, **attributes)


def undirected_support(task: TaskGraph) -> nx.Graph:
    """Unit conductance per distinct pair; omit self-loops, direction and multiplicity."""
    graph = nx.Graph()
    graph.add_nodes_from(task.graph.nodes)
    graph.add_edges_from((u, v) for u, v in task.graph.edges() if u != v)
    return graph


def resistance_curvature(task: TaskGraph) -> dict[str, Any]:
    """Pelle's formulas, evaluated independently on every connected component.

    R_ij = L+_ii + L+_jj - 2 L+_ij; p_i = 1 - sum_j A_ij R_ij / 2;
    k_ij = 2(p_i+p_j)/R_ij on edges. Cross-component R is infinity.
    """
    support = undirected_support(task)
    nodes = list(task.graph)
    index = {s: i for i, s in enumerate(nodes)}
    adjacency = nx.to_numpy_array(support, nodelist=nodes, dtype=float)
    resistance = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(resistance, 0)
    for component in nx.connected_components(support):
        ids = sorted(index[s] for s in component)
        a = adjacency[np.ix_(ids, ids)]
        laplacian = np.diag(a.sum(axis=1)) - a
        inverse = pinvh(laplacian)
        diagonal = np.diag(inverse)
        r = diagonal[:, None] + diagonal[None, :] - 2 * inverse
        r = np.maximum(r, 0)
        np.fill_diagonal(r, 0)
        resistance[np.ix_(ids, ids)] = r
    edge_resistance = np.where(adjacency > 0, resistance, 0)
    node_curvature = 1 - edge_resistance.sum(axis=1) / 2
    edge_curvature = np.full_like(adjacency, np.nan)
    ii, jj = np.where(adjacency > 0)
    edge_curvature[ii, jj] = 2 * (node_curvature[ii] + node_curvature[jj]) / resistance[ii, jj]
    return dict(nodes=nodes, index=index, adjacency=adjacency, resistance=resistance,
                node_curvature=node_curvature, edge_curvature=edge_curvature)


def graph_diagnostics(task: TaskGraph, measured: dict[str, Any]) -> dict[str, Any]:
    """Common structural diagnostics; undefined/inapplicable values remain None."""
    graph = task.graph
    support = undirected_support(task)
    a, r = measured['adjacency'], measured['resistance']
    upper = np.triu(np.ones_like(a, dtype=bool), 1)
    finite_pairs = upper & np.isfinite(r)
    edge_mask = upper & (a > 0)
    n, m = len(graph), support.number_of_edges()
    components = nx.number_connected_components(support)
    groups: dict[str, list[Any]] = defaultdict(list)
    for s, data in graph.nodes(data=True):
        groups[data['observation']].append(s)
    transitions: dict[Any, dict[str, Any]] = defaultdict(dict)
    deterministic = True
    for u, v, data in graph.edges(data=True):
        action = data['action']
        if action in transitions[u] and transitions[u][action] != v:
            deterministic = False
        transitions[u][action] = v
        deterministic &= data['probability'] == 1

    # Finite directed shortest-to-success policies, not reward-optimal policies.
    goals = [s for s, d in graph.nodes(data=True) if d['success']]
    distances: dict[Any, int] = {}
    queue = deque(goals)
    distances.update((s, 0) for s in goals)
    while queue:
        v = queue.popleft()
        for u in graph.predecessors(v):
            if u not in distances:
                distances[u] = distances[v] + 1
                queue.append(u)
    policies = {s: frozenset(d['action'] for _, v, d in graph.out_edges(s, data=True)
                            if d['kind'] == 'action' and v in distances and distances[v] == distances[s] - 1)
                for s in distances if distances[s] > 0
                and any(d['kind'] == 'action' for _, _, d in graph.out_edges(s, data=True))}
    comparable_policy = ambiguous_policy = 0
    comparable_transition = ambiguous_transition = ambiguous_next_observation = 0
    for states in groups.values():
        eligible = [s for s in states if s in policies]
        if eligible:
            comparable_policy += len(eligible)
            ambiguous_policy += len(eligible) * (len({policies[s] for s in eligible}) > 1)
        by_action: dict[str, list[Any]] = defaultdict(list)
        for s in states:
            for action, dest in transitions[s].items():
                by_action[action].append(dest)
        for destinations in by_action.values():
            comparable_transition += len(destinations)
            outcomes = {graph.nodes[v]['observation'] for v in destinations}
            ambiguous_next_observation += len(destinations) * (len(outcomes) > 1)
            ambiguous_transition += len(destinations) * (len(set(destinations)) > 1)

    # Only actual action edges participate; stimulus scheduling is not an action.
    action_transitions: dict[Any, dict[str, Any]] = defaultdict(dict)
    for u, v, d in graph.edges(data=True):
        if d['kind'] == 'action':
            action_transitions[u][d['action']] = v
    preserved = valid_pairs = raw_preserved = raw_pairs = 0
    if deterministic:
        for s, actions in action_transitions.items():
            labels = sorted(actions)
            for i, a1 in enumerate(labels):
                for a2 in labels[i + 1:]:
                    s1, s2 = actions[a1], actions[a2]
                    after1, after2 = action_transitions.get(s1, {}), action_transitions.get(s2, {})
                    if a2 in after1 and a1 in after2:
                        raw_pairs += 1
                        raw_preserved += after1[a2] == after2[a1]
                        if s1 != s and s2 != s and after1[a2] != s1 and after2[a1] != s2:
                            valid_pairs += 1
                            preserved += after1[a2] == after2[a1]
    transition_ambiguity = (ambiguous_transition / comparable_transition
                            if deterministic and comparable_transition else None)
    policy_ambiguity = (ambiguous_policy / comparable_policy
                        if deterministic and comparable_policy else None)
    directions = {(u, v) for u, v in graph.edges() if u != v}
    return dict(name=task.name, representation=task.metadata['representation'],
                nodes=n, transitions=graph.number_of_edges(), undirected_edges=m,
                components=components, cycle_rank=m - n + components,
                one_way_fraction=sum((v, u) not in directions for u, v in directions) / len(directions)
                if directions else 0,
                mean_pair_resistance=float(r[finite_pairs].mean()) if finite_pairs.any() else None,
                disconnected_pair_fraction=1 - int(finite_pairs.sum()) / int(upper.sum()) if n > 1 else 0,
                mean_edge_resistance=float(r[edge_mask].mean()) if m else None,
                mean_node_curvature=float(measured['node_curvature'].mean()),
                mean_edge_curvature=float(measured['edge_curvature'][edge_mask].mean()) if m else None,
                min_edge_curvature=float(measured['edge_curvature'][edge_mask].min()) if m else None,
                max_edge_curvature=float(measured['edge_curvature'][edge_mask].max()) if m else None,
                foster_error=abs(float(r[edge_mask].sum()) - (n - components)),
                node_sum_error=abs(float(measured['node_curvature'].sum()) - components),
                observation_aliased_state_fraction=sum(len(s) for s in groups.values() if len(s) > 1) / n,
                normalized_observation_ambiguity=(sum(len(s)**2 for s in groups.values()) / n - 1) / (n - 1)
                if n > 1 else 0,
                transition_ambiguity=transition_ambiguity,
                next_observation_ambiguity=ambiguous_next_observation / comparable_transition
                if deterministic and comparable_transition else None,
                directed_policy_ambiguity=policy_ambiguity,
                residual_history_score=max(transition_ambiguity, policy_ambiguity)
                if transition_ambiguity is not None and policy_ambiguity is not None else None,
                valid_pair_commutativity=preserved / valid_pairs if valid_pairs else None,
                raw_pair_commutativity=raw_preserved / raw_pairs if raw_pairs else None,
                commutativity_pairs=valid_pairs,
                policy_evaluable_states=comparable_policy)


def minigrid_task(env_id: str, *, seeds: tuple[int, ...] = (0,),
                  max_nodes: int = 12000) -> TaskGraph:
    """Enumerate reachable native MiniGrid transitions with the clock factored out.

    Supported classes have no hidden dynamic state beyond grid/pose/carrying and
    seed-specific Memory success/failure locations. Timeouts are removed and
    success reward is binary; this is an explicitly declared dynamics quotient.
    """
    import gymnasium as gym
    import minigrid  # noqa: F401 -- registers the environments

    allowed = {'EmptyEnv', 'FourRoomsEnv', 'DoorKeyEnv', 'MemoryEnv'}
    task = TaskGraph(env_id, metadata=dict(
        representation='native MiniGrid; clock-free quotient', seeds=list(seeds),
        observation='categorical image only; direction/mission excluded',
        abstraction='Actual seven actions; timeout omitted; success reward binarized. '
                    'Every reachable grid/pose/inventory state expanded; terminal states stop.',
        source=env_id))
    snapshots: dict[Any, Any] = {}
    ids: dict[Any, int] = {}
    queue: deque = deque()
    layers: dict[Any, int] = {}

    def intern(env: Any, obs: dict, terminal: bool, reward: float, initial=False) -> int:
        grid = env.grid.encode().tobytes()
        carry = None if env.carrying is None else tuple(env.carrying.encode())
        context = (getattr(env, 'success_pos', None), getattr(env, 'failure_pos', None))
        context = tuple(tuple(v) if v is not None else None for v in context)
        key = (tuple(int(x) for x in env.agent_pos), int(env.agent_dir), grid, carry, context, terminal)
        if key not in ids:
            if len(ids) >= max_nodes:
                raise RuntimeError(f'{env_id}: enumeration exceeded {max_nodes}; no partial graph exported')
            idx = ids[key] = len(ids)
            layer_key = (grid, carry, context)
            layer = layers.setdefault(layer_key, len(layers))
            x, y = key[0]
            angle = env.agent_dir * np.pi / 2
            # Repacked into a square of layers after enumeration.
            task.node(idx, observation=sha256(obs['image'].tobytes()).hexdigest(),
                      pos=(x + .24 * np.cos(angle), -y - .24 * np.sin(angle)),
                      label=f'({x},{y},h={env.agent_dir}), layer {layer}', initial=initial,
                      terminal=terminal, success=terminal and reward > 0,
                      pose=[x, y, int(env.agent_dir)], layer=layer,
                      grid=env.grid.encode().tolist(), carrying=carry)
            if not terminal:
                snapshots[idx] = deepcopy(env)
                queue.append(idx)
        idx = ids[key]
        if initial:
            task.graph.nodes[idx]['initial'] = True
        return idx

    for seed in seeds:
        wrapper = gym.make(env_id)
        obs, _ = wrapper.reset(seed=seed)
        env = wrapper.unwrapped
        if type(env).__name__ not in allowed:
            raise ValueError(f'Unvalidated MiniGrid state schema: {type(env).__name__}')
        intern(env, obs, False, 0, initial=True)
        wrapper.close()
    while queue:
        idx = queue.popleft()
        env = snapshots.pop(idx)
        for action in env.actions:
            nxt = deepcopy(env)
            nxt.step_count = 0
            obs, reward, terminated, _, _ = nxt.step(action)
            dest = intern(nxt, obs, terminated, reward)
            task.edge(idx, dest, action.name, reward=float(reward > 0))
    columns = int(np.ceil(np.sqrt(len(layers))))
    for _, d in task.graph.nodes(data=True):
        d['pos'][0] += (d['layer'] % columns) * (env.width + 1)
        d['pos'][1] -= (d['layer'] // columns) * (env.height + 1)
    task.metadata.update(layers=len(layers), complete_enumeration=True)
    return task


def delayed_match(*, delay: int = 4) -> TaskGraph:
    """Two samples/probes; exact finite symbolic trial model, not a NeuroGym run."""
    task = TaskGraph('Delayed match-to-sample',
                     metadata=dict(representation='finite symbolic task abstraction',
                                   delay=delay,
                                   observation='sample visible only at onset; probe at choice',
                                   abstraction='Two object identities; fixed delay; no rendered movement.'))
    for cue in range(2):
        for t in range(delay + 1):
            s = ('delay', cue, t)
            task.node(s, observation=f'sample {cue}' if t == 0 else 'blank',
                      pos=(t, cue * 4), initial=t == 0)
            if t:
                task.edge(('delay', cue, t - 1), s, 'tick', kind='event')
        for probe in range(2):
            s = ('probe', cue, probe)
            task.node(s, observation=f'probe {probe}', pos=(delay + 1, cue * 4 + probe - .5))
            task.edge(('delay', cue, delay), s, f'present_{probe}', kind='event')
            for response in range(2):
                correct = response == int(cue == probe)
                end = ('outcome', cue, probe, response)
                task.node(end, observation='reward' if correct else 'error',
                          pos=(delay + 2, cue * 4 + probe + response * .45 - .7),
                          success=correct, terminal=True)
                task.edge(s, end, f'response_{response}', reward=int(correct))
    return task


def dmlab_nonmatching() -> TaskGraph:
    """Object-choice/teleport/reset automaton from DMLab's nonmatching task rules."""
    task = TaskGraph('DMLab nonmatching: task skeleton', metadata=dict(
        representation='finite symbolic task abstraction',
        abstraction='Two identities and fixed object placement; within-room motion collapsed. '
                    'Teleport reward +1; matching collection -10, nonmatching +10; same sample after reset.',
        source='DMLab rooms_select_nonmatching_object.lua and select_nonmatching_object_factory.lua'))
    for sample in range(2):
        base = sample * 3
        for phase, x, obs in [('sample', 0, f'sample {sample}'), ('pad', 1, 'teleport pad'),
                              ('choice', 2, 'objects A and B')]:
            task.node((sample, phase), observation=obs, pos=(x, base), initial=phase == 'sample')
        task.edge((sample, 'sample'), (sample, 'pad'), 'approach_pad')
        task.edge((sample, 'pad'), (sample, 'choice'), 'teleport', reward=1)
        for obj in range(2):
            success = obj != sample
            end = (sample, 'collected', obj)
            task.node(end, observation=f'collected {obj}; reward {10 if success else -10}',
                      pos=(3, base + (obj - .5)), success=success)
            task.edge((sample, 'choice'), end, f'collect_{obj}', reward=10 if success else -10)
            task.edge(end, (sample, 'sample'), 'reset', kind='reset')
    return task


def context_decision(delay: int = 2) -> TaskGraph:
    """Binary conflicting modalities, initially cued rule, explicitly finite variant."""
    task = TaskGraph('Context-dependent choice', metadata=dict(
        representation='finite symbolic task abstraction', delay=delay,
        abstraction='Binary modalities; rule shown initially then hidden; no evidence noise.'))
    for rule, a, b in product(range(2), repeat=3):
        row = 4 * rule + 2 * a + b
        for t in range(delay + 1):
            s = (rule, a, b, t)
            obs = f'rule {rule}; A={a},B={b}' if t == 0 else f'A={a},B={b}'
            task.node(s, observation=obs, pos=(t, row), initial=t == 0)
            if t:
                task.edge((rule, a, b, t - 1), s, 'tick', kind='event')
        for choice in range(2):
            s = (rule, a, b, delay + 1, choice)
            correct = choice == (a if rule == 0 else b)
            task.node(s, observation='reward' if correct else 'error',
                      pos=(delay + 1, row + choice * .35), success=correct, terminal=True)
            task.edge((rule, a, b, delay), s, f'choose_{choice}', reward=int(correct))
    return task


def alternation(length: int = 4) -> TaskGraph:
    """Repeated-trial alternation with explicit reset events and retained last side."""
    task = TaskGraph('Delayed alternation', metadata=dict(
        representation='finite symbolic task abstraction', length=length,
        abstraction='Corridor plus left/right choice; last rewarded side persists across reset.'))
    for previous in range(2):
        for x in range(length + 1):
            s = (previous, x)
            task.node(s, observation=f'corridor {x}', pos=(x, previous * 2), initial=x == 0)
            if x:
                task.edge((previous, x - 1), s, 'forward')
                task.edge(s, (previous, x - 1), 'backward')
        for side in range(2):
            correct = side != previous
            s = ('outcome', previous, side)
            task.node(s, observation=f'side {side}; reward {int(correct)}',
                      pos=(length + 1, previous * 2 + side * .5), success=correct)
            task.edge((previous, length), s, f'choose_{side}', reward=int(correct))
            task.edge(s, (side if correct else previous, 0), 'reset', kind='reset')
    return task


def hanoi(disks: int = 3) -> TaskGraph:
    """All 3**disks legal configurations; smallest disk is tuple index zero."""
    task = TaskGraph(f'Tower of Hanoi ({disks} disks)', metadata=dict(
        representation='exact symbolic transition graph', disks=disks,
        abstraction='All legal one-disk moves; goal remains traversable.'))
    for state in product(range(3), repeat=disks):
        # Recursive triangular layout.
        p = np.zeros(2)
        for disk, peg in enumerate(state):
            p += 2 ** disk * np.array([(0, 0), (1, 0), (.5, .866)][peg])
        task.node(state, observation=str(state), pos=tuple(p), initial=state == (0,) * disks,
                  success=state == (2,) * disks)
    for state in list(task.graph):
        top = {peg: next((i for i, p in enumerate(state) if p == peg), disks) for peg in range(3)}
        for source in range(3):
            disk = top[source]
            if disk == disks:
                continue
            for dest in range(3):
                if dest != source and disk < top[dest]:
                    nxt = list(state)
                    nxt[disk] = dest
                    task.edge(state, tuple(nxt), f'{source}_to_{dest}')
    return task


def taxi_task(destination: int = 0) -> TaskGraph:
    """Reachable native Taxi-v3 P-table graph for one fixed destination."""
    import gymnasium as gym

    env = gym.make('Taxi-v3').unwrapped
    task = TaskGraph('Taxi (fixed destination)', metadata=dict(
        representation='native Gymnasium transition table', destination=destination,
        abstraction='Fixed destination; terminal arrival split from ordinary state; timeout omitted.'))
    starts = [s for s, p in enumerate(env.initial_state_distrib)
              if p > 0 and list(env.decode(s))[-1] == destination]
    queue = deque((s, False) for s in starts)
    seen = set(queue)

    def add(state, initial=False):
        s, terminal = state
        row, col, passenger, dest = list(env.decode(s))
        layer = max(0, passenger - 1)
        pos = (col + (layer % 2) * 7, -row - (layer // 2) * 7) if not terminal else (15, -6)
        task.node(state, observation=str(s), pos=pos,
                  initial=initial, terminal=terminal, success=terminal,
                  taxi_state=[row, col, passenger, dest])

    for state in queue:
        add(state, True)
    while queue:
        state = queue.popleft()
        s, terminal = state
        if terminal:
            continue
        for action, outcomes in env.P[s].items():
            for p, dest, reward, done in outcomes:
                nxt = (dest, done)
                if nxt not in seen:
                    seen.add(nxt)
                    add(nxt)
                    queue.append(nxt)
                task.edge(state, nxt, ['south', 'north', 'east', 'west', 'pickup', 'dropoff'][action],
                          probability=p, reward=reward)
    env.close()
    return task


def dmlab_layout_graph(layout: str, *, skymaze: bool = False) -> TaskGraph:
    """Actual DMLab ASCII layout -> tile/heading graph, at Pelle's map abstraction.

    Navigation assumes cell-to-cell movement. SkyMaze only permits same-height or
    downward adjacent moves. These are map-derived edges, not engine rollout proof.
    """
    rows = layout.rstrip('\n').splitlines()
    task = TaskGraph('DMLab SkyMaze' if skymaze else 'DMLab goal-location maze', metadata=dict(
        representation='live DMLab layout; geometric abstraction',
        abstraction='Cells x four headings; turn/forward/backward macro moves. '
                    'SkyMaze forbids ascent; physics, goal/reset state and clock omitted.',
        layout=layout, movement_validated=False))
    cells = {(x, y): c for y, row in enumerate(rows) for x, c in enumerate(row)
             if ('a' <= c <= 'z' if skymaze else c != '*')}
    vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for (x, y), char in cells.items():
        for h, (dx, dy) in enumerate(vectors):
            s = (x, y, h)
            task.node(s, observation=str(s), pos=(x + .24 * dx, -y - .24 * dy),
                      tile=[x, y], heading=h, height=40 - (ord(char) - ord('a')) if skymaze else 0)
    for x, y, h in list(task.graph):
        s = (x, y, h)
        task.edge(s, (x, y, (h - 1) % 4), 'L')
        task.edge(s, (x, y, (h + 1) % 4), 'R')
        dx, dy = vectors[h]
        for action, sign in [('F', 1), ('B', -1)]:
            dest = (x + sign * dx, y + sign * dy)
            if dest in cells and (not skymaze or cells[dest] >= cells[(x, y)]):
                task.edge(s, (*dest, h), action)
    task.metadata['observation'] = 'privileged cell/heading identity; RGB aliasing not evaluated'
    return task
