# Exploration Grid Helper Design

## Goal

Create a helper data structure for a new exploration environment derived from `planning`.

The helper should:

- discretize the env-local training map into a 2D grid over `(x, y)`
- track which cells the drone has visited
- support multi-environment training
- support reset/update/reward logic efficiently on GPU
- handle spawn initialization and out-of-bounds behavior cleanly

This helper is meant to support exploration reward shaping in unknown environments.

## Coordinate Convention

Use env-local coordinates, not world coordinates.

That matches the current AirGym task design where the drone state is already expressed in local coordinates via:

- `root_pos_local = root_pos_w - scene.env_origins`

This avoids ambiguity across replicated environments.

## Core State

Minimum map:

```python
visited: torch.Tensor  # (num_envs, grid_h, grid_w), dtype=torch.bool
```

Meaning:

- `False`: unvisited
- `True`: visited at least once

Additional state to keep:

```python
spawn_cell: torch.Tensor      # (num_envs, 2), long
current_cell: torch.Tensor    # (num_envs, 2), long
previous_cell: torch.Tensor   # (num_envs, 2), long
out_of_bounds: torch.Tensor   # (num_envs,), bool
coverage_count: torch.Tensor  # (num_envs,), long
```

Optional but recommended:

```python
visit_count: torch.Tensor     # (num_envs, grid_h, grid_w), int32
```

Use `visit_count` only if repeated visitation needs to be tracked explicitly. For first-visit reward only, `visited` is enough.

## Grid Definition

Define fixed local map limits:

```python
x_min, x_max
y_min, y_max
cell_size
```

Then derive:

```python
grid_h = ceil((x_max - x_min) / cell_size)
grid_w = ceil((y_max - y_min) / cell_size)
```

For a planning-like map, a natural first configuration is:

```python
x_min = -LENGTH - 0.5
x_max =  LENGTH + 0.5
y_min = -WIDTH
y_max =  WIDTH
```

The grid should be allocated once:

```python
visited = torch.zeros(num_envs, grid_h, grid_w, device=device, dtype=torch.bool)
```

## Helper API

The helper should expose the following methods.

### 1. `xy_to_grid(xy)`

Input:

- `xy`: `(num_envs, 2)` local positions

Output:

- `cell_ij`: `(num_envs, 2)` long
- `in_bounds`: `(num_envs,)` bool

Computation:

```python
ix = floor((x - x_min) / cell_size)
iy = floor((y - y_min) / cell_size)
```

Important:

- clamp indices only for safe indexing
- keep a separate `in_bounds` mask for semantics

Do not treat clamped out-of-range positions as genuine visited cells.

### 2. `reset(env_ids, spawn_xy)`

Responsibilities:

- zero the selected env grids
- compute spawn cell from `spawn_xy`
- initialize `spawn_cell`, `previous_cell`, `current_cell`
- mark the spawn cell as visited
- reset `coverage_count`
- clear `out_of_bounds`

Recommended behavior:

- spawn cell counts toward coverage
- spawn cell does not produce reward

### 3. `update(env_ids, xy)`

Responsibilities:

- compute current grid cell
- detect out-of-bounds
- update the grid with new visited cells
- update `previous_cell` and `current_cell`
- return newly visited cell count per env

This method should return:

```python
new_visit_count: torch.Tensor  # (num_envs,), long
```

### 4. `coverage_ratio()`

Return:

```python
coverage_count.float() / total_valid_cells
```

This is useful for logging and curriculum logic.

## Spawn Handling

Spawn position should come directly from the env reset logic.

The current `planning` reset already constructs the local spawn position explicitly before writing robot state. That same local spawn `(x, y)` should initialize the exploration grid.

Important rule:

- if the spawn point falls outside the configured grid, fail early

That should be treated as a configuration error, not silently clamped.

## Out-of-Bounds Handling

Two layers are needed:

### Safety layer

Clamp indices so tensor indexing never crashes.

### Semantic layer

Keep `in_bounds` and `out_of_bounds` masks.

Recommended semantics:

- if a point is out of bounds, do not mark visited
- optionally penalize
- optionally terminate if the env design requires strict map containment

This is better than pretending the agent visited the nearest edge cell.

## Important Improvement: Mark the Path, Not Just the Endpoint

If the drone moves more than one cell between steps, endpoint-only marking misses intermediate coverage.

Therefore, do not update only the final cell.

Instead:

- sample points along the segment from `previous_xy` to `current_xy`
- convert those samples to grid cells
- mark all resulting cells

GPU-friendly approach:

```python
delta = current_xy - previous_xy
num_samples = ceil(max(abs(delta_x), abs(delta_y)) / cell_size) + 1
samples = lerp(previous_xy, current_xy, t)
```

This is preferable to a planner-style CPU Bresenham or DDA implementation in the RL loop.

## Reward Logic

Simplest version:

```python
reward = reward_per_new_cell * new_visit_count.float()
```

Where `new_visit_count` counts newly visited path cells, not just the final cell.

Recommended additions:

- small out-of-bounds penalty
- collision penalty
- stagnation penalty if no new cells are visited for too long

Recommended interpretation:

- coverage reward should be first-visit only
- revisits should not generate positive reward

## Efficient Multi-Env Update

Flattened indexing is the cleanest approach:

```python
flat = visited.view(num_envs, -1)
flat_idx = ix * grid_w + iy
```

Then:

- read old values
- count cells that were previously `False`
- set them to `True`

This is compatible with batched path updates as well.

## Reset Behavior For Multi-Env Training

Per-env reset should:

```python
visited[env_ids] = False
out_of_bounds[env_ids] = False
coverage_count[env_ids] = 0
spawn_cell[env_ids] = ...
current_cell[env_ids] = ...
previous_cell[env_ids] = ...
visited[env_ids, spawn_ix, spawn_iy] = True
coverage_count[env_ids] = 1
```

If spawn should not count as covered for the metric, that can be changed, but counting it is usually the cleaner definition.

## Better Long-Term Design

For exploration in unknown environments, `visited` alone is often not enough.

Long-term, separate:

```python
visited: bool    # where the drone physically traveled
explored: bool   # where the drone has observed through sensors
occupied: bool   # optional local occupancy estimate
```

Why:

- `visited` drives physical coverage
- `explored` better matches frontier exploration
- `occupied` supports mapping and safer motion

Recommended reward mix later:

- small reward for new `visited`
- larger reward for new `explored`
- penalties for collision and leaving bounds

## Recommended First Version

Implement first:

- env-local 2D `visited` map only
- spawn marking on reset
- path-cell marking each step
- first-visit reward
- optional out-of-bounds penalty

Do not start with a full explored/occupied map unless it is immediately needed.

## Suggested Config

```python
@dataclass
class CoverageGridCfg:
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    cell_size: float
    reward_per_new_cell: float = 1.0
    out_of_bounds_penalty: float = 1.0
    mark_path: bool = True
    count_spawn_as_visited: bool = True
    reward_spawn: bool = False
```

## Observation Strategy

Do not assume the full coverage grid should go to the actor.

Recommended asymmetric setup:

- actor: local state + depth / local crop
- critic: full `visited` map and later `explored` map

This fits the current asymmetric-observation direction in the AirGym tasks.

## Implementation Notes

The helper should be implemented as a pure torch utility or a small stateful helper class with no simulator dependencies.

That allows:

- isolated unit tests with fake data
- reuse across multiple tasks
- clean integration into a new exploration env derived from `planning`
