# Push-optimal Sokoban solver

Entry for make.fun `sokoban-solver-01-run-20261121` ($100, 10 places x $10).

Every level is a 1-wide hallway plus 1-4 rooms (3-4 boxes each); boxes never
leave their room, so the level's push-optimal count is the sum of its rooms'
optima. This program decomposes each level into rooms, solves every room
push-optimally with A* over pushes (min-cost-matching heuristic over exact
BFS box-to-goal distances, plus static deadlock pruning), and stitches the
rooms together with hallway walks.

Practice set (kit): 1,000,000 / 1,000,000 — every instance solved in exactly
P* pushes (13/16/29/47/60). House baseline plain-astar scores 200,000 on
the same set.

Run: `node main.ts` (Node 22), reads `/input/<id>/level.txt`, writes
`/output/<id>/solution.txt`. Deterministic; no network, no clock reads.
