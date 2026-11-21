// Push-optimal Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Structure of every level (by construction): a 1-wide hallway (the row the
// player starts on) plus 1-4 rooms in 6x6 slots, one door each; boxes never
// leave their room, and a level's push-optimal count P* is the sum of its
// rooms' optima. So: decompose into rooms, solve each room push-optimally
// with A* over pushes from its door, and stitch the rooms together with
// hallway walks. Deterministic: fixed iteration order, work budgets (never
// the wall clock), no randomness.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (deterministic, counted in expansions, never the clock).
const ASTAR_BUDGET = 400_000;
const GREEDY_BUDGET = 400_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet: new Set(goals), boxes, player };
}

// --- room detection -------------------------------------------------------
// Hallway = the player's row. Rooms = connected floor components off it.
interface Room {
  cells: number[]; // global cell indices, sorted
  cellSet: Set<number>;
  doors: number[]; // room cells adjacent to a hallway floor cell
  entry: number; // hallway floor cell adjacent to doors[0]
}

function findRooms(L: Level): Room[] {
  const { w, h, wall, player } = L;
  const hy = Math.floor(player / w);
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  const inBounds = (x: number, y: number): boolean => x >= 0 && x < w && y >= 0 && y < h;
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || Math.floor(i / w) === hy || seen[i]) continue;
    const comp: number[] = [];
    const q: number[] = [i];
    seen[i] = 1;
    while (q.length > 0) {
      const c = q.pop()!;
      comp.push(c);
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (!inBounds(nx, ny)) continue;
        const n = ny * w + nx;
        if (wall[n] || ny === hy || seen[n]) continue;
        seen[n] = 1;
        q.push(n);
      }
    }
    comp.sort((a, b) => a - b);
    const doors = comp.filter((c) => {
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (!inBounds(nx, ny)) continue;
        const n = ny * w + nx;
        if (ny === hy && !wall[n]) return true;
      }
      return false;
    });
    if (doors.length === 0) continue;
    const door = doors[0]!;
    const dcx = door % w;
    const dcy = Math.floor(door / w);
    let entry = -1;
    for (let d = 0; d < 4; d++) {
      const nx = dcx + DX[d]!;
      const ny = dcy + DY[d]!;
      if (!inBounds(nx, ny)) continue;
      const n = ny * w + nx;
      if (ny === hy && !wall[n]) {
        entry = n;
        break;
      }
    }
    rooms.push({ cells: comp, cellSet: new Set(comp), doors, entry });
  }
  return rooms;
}

// --- BFS helpers on the full grid ------------------------------------------
function bfsWalk(
  L: Level,
  from: number,
  to: number,
  blocked: Set<number>, // box cells (treated as walls) — does NOT contain `from`
): string | null {
  if (from === to) return '';
  const { w, h, wall } = L;
  const prev = new Int32Array(w * h).fill(-2);
  const prevMove = new Int8Array(w * h).fill(-1);
  prev[from] = -1;
  const q: number[] = [from];
  let head = 0;
  while (head < q.length) {
    const c = q[head++]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] || blocked.has(n) || prev[n] !== -2) continue;
      prev[n] = c;
      prevMove[n] = d;
      if (n === to) {
        let s = '';
        for (let t = n; t !== from; t = prev[t]!) s = LETTER[prevMove[t]!]! + s;
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// --- per-room push-optimal A* -----------------------------------------------
// Local floor = room cells + the entry cell (hallway). The entry cell is
// floor for the player but a wall for boxes: a box pushed into the hallway
// can never come back.
interface RoomResult {
  moves: string;
  pushes: number;
  endPlayer: number; // global cell
}

function solveRoom(
  L: Level,
  room: Room,
  roomBoxes: number[],
  roomGoals: number[],
): RoomResult | null {
  const { w, wall, goalSet } = L;
  // Local indexing.
  const cells: number[] = [...room.cells, room.entry].sort((a, b) => a - b);
  const n = cells.length;
  const local = new Map<number, number>();
  cells.forEach((c, i) => local.set(c, i));
  const isGoal = new Uint8Array(n);
  roomGoals.forEach((g) => {
    const li = local.get(g);
    if (li !== undefined) isGoal[li] = 1;
  });
  // Neighbours (local index or -1). The entry cell is floor for the player;
  // boxes may not occupy it: mark it as a box-wall via boxWall[].
  const boxWall = new Uint8Array(n);
  boxWall[local.get(room.entry)!] = 1;
  const nbr: Int32Array[] = [];
  for (let i = 0; i < n; i++) {
    const g = cells[i]!;
    const gx = g % w;
    const gy = Math.floor(g / w);
    const row = new Int32Array(4).fill(-1);
    for (let d = 0; d < 4; d++) {
      const nx = gx + DX[d]!;
      const ny = gy + DY[d]!;
      if (nx < 0 || nx >= w || ny < 0 || ny >= L.h) continue;
      const nc = ny * w + nx;
      const li = local.get(nc);
      if (li !== undefined) row[d] = li;
    }
    nbr.push(row);
  }
  const startBoxes = roomBoxes.map((b) => local.get(b)!).sort((a, b) => a - b);
  const startPlayer = local.get(room.entry)!;
  const goalList = roomGoals.map((g) => local.get(g)!);

  if (startBoxes.every((b) => isGoal[b])) {
    return { moves: '', pushes: 0, endPlayer: room.entry };
  }

  // BFS distances from each goal over the local floor graph (box travel).
  const distToGoal: Int32Array[] = goalList.map((g) => {
    const dist = new Int32Array(n).fill(-1);
    dist[g] = 0;
    const q: number[] = [g];
    for (let head = 0; head < q.length; head++) {
      const c = q[head]!;
      for (let d = 0; d < 4; d++) {
        const m = nbr[c]![d]!;
        if (m >= 0 && dist[m] < 0) {
          dist[m] = dist[c]! + 1;
          q.push(m);
        }
      }
    }
    return dist;
  });

  // live[]: a cell is live if a box there can still reach some goal in the
  // box-move graph (edge c->n when n is floor and the cell behind, 2c-n, is
  // floor). Reverse BFS from all goals.
  const live = new Uint8Array(n);
  {
    const q: number[] = [];
    goalList.forEach((g) => {
      if (!live[g]) {
        live[g] = 1;
        q.push(g);
      }
    });
    for (let head = 0; head < q.length; head++) {
      const x = q[head]!;
      const gx = cells[x]! % w;
      const gy = Math.floor(cells[x]! / w);
      for (let d = 0; d < 4; d++) {
        // predecessor c = x - d, needs behind cell x - 2d to be floor
        const cx = gx - DX[d]!;
        const cy = gy - DY[d]!;
        const bx = gx - 2 * DX[d]!;
        const by = gy - 2 * DY[d]!;
        if (cx < 0 || cx >= w || cy < 0 || cy >= L.h) continue;
        if (bx < 0 || bx >= w || by < 0 || by >= L.h) continue;
        const c = cy * w + cx;
        const b = by * w + bx;
        const ci = local.get(c);
        if (ci === undefined || live[ci]) continue;
        if (local.get(b) === undefined) continue; // behind cell must be floor
        live[ci] = 1;
        q.push(ci);
      }
    }
  }

  // 2x2 freeze blocks: four local cells (or walls) forming a square, none a
  // goal. A box inside one can never move.
  const freezeBlocks: number[][] = [];
  {
    const at = (x: number, y: number): number => {
      if (x < 0 || x >= w || y < 0 || y >= L.h) return -2; // outside = wall
      const li = local.get(y * w + x);
      return li === undefined ? -2 : li;
    };
    const seenBlocks = new Set<string>();
    for (let i = 0; i < n; i++) {
      const gx = cells[i]! % w;
      const gy = Math.floor(cells[i]! / w);
      for (const [ox, oy] of [
        [0, 0],
        [-1, 0],
        [0, -1],
        [-1, -1],
      ] as const) {
        const x0 = gx + ox;
        const y0 = gy + oy;
        const key = `${x0},${y0}`;
        if (seenBlocks.has(key)) continue;
        seenBlocks.add(key);
        const cs = [at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)];
        if (cs.some((c) => c >= 0 && isGoal[c])) continue; // a goal saves it
        freezeBlocks.push(cs);
      }
    }
  }

  const boxSetOf = (bs: number[]): Set<number> => new Set(bs);
  const isDead = (bs: number[]): boolean => {
    for (const b of bs) if (!live[b]) return true;
    const set = boxSetOf(bs);
    for (const blk of freezeBlocks) {
      let boxCount = 0;
      let allBlocked = true;
      for (const c of blk) {
        if (c === -2) continue; // wall/outside
        if (set.has(c)) boxCount++;
        else allBlocked = false;
      }
      if (boxCount > 0 && allBlocked) return true;
    }
    return false;
  };

  // Admissible heuristic: min-cost matching of boxes to goals over exact
  // BFS distances (each push moves one box one cell). Brute-force
  // permutations: at most 4 boxes.
  const heuristic = (bs: number[]): number => {
    const m = bs.length;
    if (m === 0) return 0;
    if (m !== goalList.length) {
      let s = 0;
      for (const b of bs) {
        let best = Infinity;
        for (let gi = 0; gi < goalList.length; gi++) {
          const d = distToGoal[gi]![b]!;
          if (d >= 0 && d < best) best = d;
        }
        if (best === Infinity) return Infinity;
        s += best;
      }
      return s;
    }
    // Permute goals over boxes.
    const idx = goalList.map((_, i) => i);
    let best = Infinity;
    const permute = (k: number): void => {
      if (k === m) {
        let s = 0;
        for (let i = 0; i < m; i++) {
          const d = distToGoal[idx[i]!]![bs[i]!]!;
          if (d < 0) return;
          s += d;
          if (s >= best) return;
        }
        if (s < best) best = s;
        return;
      }
      for (let i = k; i < m; i++) {
        [idx[k], idx[i]] = [idx[i]!, idx[k]!];
        permute(k + 1);
        [idx[k], idx[i]] = [idx[i]!, idx[k]!];
      }
    };
    permute(0);
    return best;
  };

  // Flood fill the player's reachable region from `p`, avoiding boxes.
  // Returns {reach: Set, parent: Int32Array, parentMove: Int8Array}.
  const flood = (p: number, bs: Set<number>): { reach: Set<number>; parent: Int32Array; parentMove: Int8Array } => {
    const parent = new Int32Array(n).fill(-2);
    const parentMove = new Int8Array(n).fill(-1);
    parent[p] = -1;
    const q: number[] = [p];
    const reach = new Set<number>([p]);
    for (let head = 0; head < q.length; head++) {
      const c = q[head]!;
      for (let d = 0; d < 4; d++) {
        const m = nbr[c]![d]!;
        if (m < 0 || bs.has(m) || parent[m] !== -2) continue;
        parent[m] = c;
        parentMove[m] = d;
        reach.add(m);
        q.push(m);
      }
    }
    return { reach, parent, parentMove };
  };

  const walkPath = (fl: { parent: Int32Array; parentMove: Int8Array }, from: number, to: number): string => {
    let s = '';
    for (let t = to; t !== from; t = fl.parent[t]!) s = LETTER[fl.parentMove[t]!]! + s;
    return s;
  };

  // A* (greedy=false) or greedy best-first (greedy=true). Returns the
  // solution or null when the budget runs out / no solution.
  const search = (greedy: boolean, budget: number): RoomResult | null => {
    const sp: number[] = [startPlayer];
    const sb: number[][] = [startBoxes];
    const sg: number[] = [0];
    const parent: number[] = [-1];
    const seg: string[] = [''];
    const startKey = (() => {
      const fl = flood(startPlayer, boxSetOf(startBoxes));
      let canon = Infinity;
      fl.reach.forEach((c) => {
        if (c < canon) canon = c;
      });
      return `${startBoxes.join(',')}:${canon}`;
    })();
    const best = new Map<string, number>([[startKey, 0]]);
    const h0 = heuristic(startBoxes);
    if (h0 === Infinity) return null;
    // Heap entries: [f, g, idx]; less = lower f, then HIGHER g, then lower idx.
    const heap: [number, number, number][] = [[greedy ? h0 : h0, 0, 0]];
    const less = (a: [number, number, number], b: [number, number, number]): boolean =>
      a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
    const heapPush = (e: [number, number, number]): void => {
      let i = heap.length;
      heap.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!less(heap[i]!, heap[p]!)) break;
        [heap[i], heap[p]] = [heap[p]!, heap[i]!];
        i = p;
      }
    };
    const heapPop = (): [number, number, number] => {
      const top = heap[0]!;
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= heap.length) break;
          if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
          if (!less(heap[c]!, heap[i]!)) break;
          [heap[i], heap[c]] = [heap[c]!, heap[i]!];
          i = c;
        }
      }
      return top;
    };
    const keyOf = (bs: number[], p: number): string => {
      const fl = flood(p, boxSetOf(bs));
      let canon = Infinity;
      fl.reach.forEach((c) => {
        if (c < canon) canon = c;
      });
      return `${bs.join(',')}:${canon}`;
    };
    let expanded = 0;
    while (heap.length > 0 && expanded < budget) {
      const [, g, s] = heapPop();
      const bs = sb[s]!;
      const p = sp[s]!;
      // Stale heap entry?
      if (best.get(keyOf(bs, p)) !== s) continue;
      if (bs.every((b) => isGoal[b])) {
        let moves = '';
        for (let t = s; t > 0; t = parent[t]!) moves = seg[t]! + moves;
        return { moves, pushes: g, endPlayer: cells[p]! };
      }
      expanded++;
      const boxSet = boxSetOf(bs);
      const fl = flood(p, boxSet);
      for (let k = 0; k < bs.length; k++) {
        const b = bs[k]!;
        for (let d = 0; d < 4; d++) {
          const stance = nbr[b]![d ^ 1]!; // cell behind the box (opposite of d: 0<->1, 2<->3)
          const dest = nbr[b]![d]!;
          if (stance < 0 || dest < 0) continue;
          if (!fl.reach.has(stance)) continue;
          if (boxWall[dest] || boxSet.has(dest)) continue;
          const nb = bs.slice();
          nb[k] = dest;
          nb.sort((a, b) => a - b);
          if (isDead(nb)) continue;
          const np = b; // player ends on the box's old cell
          const nk = keyOf(nb, np);
          const seen = best.get(nk);
          if (seen !== undefined && sg[seen]! <= g + 1) continue;
          const h = heuristic(nb);
          if (h === Infinity) continue;
          const t = sp.length;
          sp.push(np);
          sb.push(nb);
          sg.push(g + 1);
          parent.push(s);
          seg.push(walkPath(fl, p, stance) + LETTER[d]!);
          best.set(nk, t);
          heapPush([greedy ? h : g + 1 + h, g + 1, t]);
        }
      }
    }
    return null;
  };

  return search(false, ASTAR_BUDGET) ?? search(true, GREEDY_BUDGET);
}

// --- full-level solve --------------------------------------------------------
function solveLevel(text: string): string {
  const L = parseLevel(text);
  const rooms = findRooms(L);
  let moves = '';
  let player = L.player;
  const boxes = L.boxes.slice();
  for (const room of rooms) {
    const roomBoxSet = room.cellSet;
    const rBoxes = boxes.filter((b) => roomBoxSet.has(b));
    const rGoals = L.goals.filter((g) => roomBoxSet.has(g));
    if (rBoxes.length === 0) continue;
    const blocked = new Set(boxes);
    blocked.delete(player);
    const walk = bfsWalk(L, player, room.entry, blocked);
    if (walk === null) return '';
    moves += walk;
    player = room.entry;
    const res = solveRoom(L, room, rBoxes, rGoals);
    if (res === null) return '';
    moves += res.moves;
    player = res.endPlayer;
    // Room boxes are now on goals: replace them in the global box list.
    const solved = new Set(rGoals);
    for (let i = 0; i < boxes.length; i++) {
      if (roomBoxSet.has(boxes[i]!)) {
        // remove; re-add goals below
        boxes[i] = -1;
      }
    }
    const kept = boxes.filter((b) => b >= 0);
    boxes.length = 0;
    kept.forEach((b) => boxes.push(b));
    rGoals.forEach((g) => boxes.push(g));
    boxes.sort((a, b) => a - b);
    void solved;
  }
  // Self-verify by replaying on the full level; on failure, answer empty.
  if (!verifyLevel(L, moves)) return '';
  return moves;
}

function verifyLevel(L: Level, moves: string): boolean {
  const { w, wall, goalSet } = L;
  const boxes = new Set(L.boxes);
  let p = L.player;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch.toLowerCase());
    if (d < 0) return false;
    const q = p + DY[d]! * w + DX[d]!;
    if (wall[q]) return false;
    if (boxes.has(q)) {
      const r = q + DY[d]! * w + DX[d]!;
      if (wall[r] || boxes.has(r)) return false;
      boxes.delete(q);
      boxes.add(r);
    }
    p = q;
  }
  for (const b of boxes) if (!goalSet.has(b)) return false;
  return true;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
