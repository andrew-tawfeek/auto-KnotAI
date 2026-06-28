# An implementation of mosaic theory in pure Python (no SageMath required).

import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


# SageMath-compatible constants
oo = math.inf
pi = math.pi


def flatten(lst):
    """Recursively flattens a (possibly nested) list or tuple of scalars."""
    result = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


class Matrix:
    """Lightweight 2D integer matrix with SageMath-style repr.

    Backed by a numpy array internally (for fast slicing/assignment), but
    scalar access yields plain Python ints, row access yields plain lists,
    and repr/str prints aligned rows without the leading ``array(...)``.
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        if isinstance(data, Matrix):
            arr = np.array(data._data, dtype=int)
        else:
            arr = np.array(data, dtype=int)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        self._data = arr

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                return Matrix(result)
            return [int(x) for x in result]
        return int(result)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        for row in self._data:
            yield [int(x) for x in row]

    def __eq__(self, other):
        if isinstance(other, Matrix):
            return np.array_equal(self._data, other._data)
        if isinstance(other, (list, tuple, np.ndarray)):
            return np.array_equal(self._data, np.asarray(other))
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def copy(self):
        return Matrix(self._data.copy())

    def tolist(self):
        return self._data.tolist()

    def rows(self):
        """SageMath-compatible accessor: list of rows (each a Python list)."""
        return [[int(x) for x in row] for row in self._data]

    def nonzero_positions(self):
        """List of (i, j) tuples of nonzero entries, in row-major order."""
        rs, cs = np.nonzero(self._data)
        return [(int(r), int(c)) for r, c in zip(rs, cs)]

    def __repr__(self):
        if self._data.size == 0:
            return "[]"
        cells = [[str(int(x)) for x in row] for row in self._data]
        width = max(len(s) for row in cells for s in row)
        return "\n".join(
            "[" + " ".join(s.rjust(width) for s in row) + "]" for row in cells
        )

    def __str__(self):
        return self.__repr__()


def matrix(data):
    """SageMath-style constructor for a Matrix."""
    return Matrix(data)


def _zeros(n, m):
    return Matrix(np.zeros((n, m), dtype=int))


def _diagonal_matrix(values):
    """Pure-Python replacement for SageMath's diagonal_matrix."""
    n = len(values)
    arr = np.zeros((n, n), dtype=int)
    for i, v in enumerate(values):
        arr[i, i] = int(v)
    return Matrix(arr)


def _block_matrix(blocks):
    """Pure-Python replacement for SageMath's block_matrix (list of lists of matrices)."""
    rows = []
    for row in blocks:
        row_arrays = []
        for b in row:
            if isinstance(b, Matrix):
                row_arrays.append(b._data)
            else:
                row_arrays.append(np.asarray(b, dtype=int))
        rows.append(np.hstack(row_arrays))
    return Matrix(np.vstack(rows))


# Tile configuration data - defines connection directions for each tile type
TILE_CONNECTIONS = {
    0: [],
    1: ['left', 'down'],
    2: ['right', 'down'],
    3: ['up', 'right'],
    4: ['left', 'up'],
    5: ['left', 'right'],
    6: ['up', 'down'],
    7: [['down', 'left'], ['up', 'right']],
    8: [['down', 'right'], ['left', 'up']],
    9: [['down', 'up'], ['left', 'right']],
    10: [['left', 'right'], ['down', 'up']],
}

# Tiles that have 4 connection points (2 strands)
FOUR_POINT_TILES = {7, 8, 9, 10}

# Crossing tiles
CROSSING_TILES = {9, 10}

# Zoom mappings for each tile type (3x3 matrix representation)
TILE_ZOOM_MAPS = {
    0: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    1: [[0, 0, 0], [5, 1, 0], [0, 6, 0]],
    2: [[0, 0, 0], [0, 2, 5], [0, 6, 0]],
    3: [[0, 6, 0], [0, 3, 5], [0, 0, 0]],
    4: [[0, 6, 0], [5, 4, 0], [0, 0, 0]],
    5: [[0, 0, 0], [5, 5, 5], [0, 0, 0]],
    6: [[0, 6, 0], [0, 6, 0], [0, 6, 0]],
    7: [[0, 3, 1], [1, 0, 3], [3, 1, 0]],
    8: [[2, 4, 0], [4, 0, 2], [0, 2, 4]],
    9: [[0, 6, 0], [5, 9, 5], [0, 6, 0]],
    10: [[0, 6, 0], [5, 10, 5], [0, 6, 0]],
}

# Special zoom for tile 9 with onlyUpDown option
TILE_9_UP_DOWN_ZOOM = [[2, 8, 1], [7, 10, 7], [3, 8, 4]]

# Tiles that connect in each direction
TILES_GOING_UP = {3, 4, 6, 7, 8, 9, 10}
TILES_GOING_DOWN = {1, 2, 6, 7, 8, 9, 10}
TILES_GOING_LEFT = {1, 4, 5, 7, 8, 9, 10}
TILES_GOING_RIGHT = {2, 3, 5, 7, 8, 9, 10}


# Precomputed, immutable per-tile-type lookups. These let the hot paths avoid
# constructing a Tile object (and re-running flatten) for every cell access.
# Keys are tile ints 0..10; values are derived directly from TILE_CONNECTIONS so
# they stay in lockstep with the canonical definitions above.
#   TILE_FLAT_LIST[n]  -- flattened connection directions, original order
#                         (order matters: random.choice over it must be stable)
#   TILE_FLAT_SET[n]   -- same directions as a frozenset for O(1) membership
#   TILE_EXIT[n]       -- {entry_direction: exit_direction} per strand
#   TILE_NUM_STRANDS[n]-- number of strands on the tile (0, 1, or 2)
TILE_FLAT_LIST = {n: flatten(TILE_CONNECTIONS.get(n, [])) for n in range(11)}
TILE_FLAT_SET = {n: frozenset(TILE_FLAT_LIST[n]) for n in range(11)}


def _build_tile_exit():
    exits = {}
    for n in range(11):
        conns = TILE_CONNECTIONS.get(n, [])
        mapping = {}
        if n in FOUR_POINT_TILES:
            for strand in conns:  # each strand is a [entry, exit] pair
                mapping[strand[0]] = strand[1]
                mapping[strand[1]] = strand[0]
        elif conns:  # single-strand tile: conns is a flat [dir_a, dir_b]
            mapping[conns[0]] = conns[1]
            mapping[conns[1]] = conns[0]
        exits[n] = mapping
    return exits


TILE_EXIT = _build_tile_exit()
TILE_NUM_STRANDS = {n: (0 if n == 0 else (1 if n in range(1, 7) else 2)) for n in range(11)}
# numpy lookup table for vectorized strand-count maps (index == tile int).
_NUM_STRANDS_LUT = np.array([TILE_NUM_STRANDS[n] for n in range(11)], dtype=int)


def opposite(direction):
    """Returns the opposite direction."""
    direction_opposites = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left',
    }
    assert direction in direction_opposites
    return direction_opposites[direction]


class Tile:
    def __init__(self, N):
        N = int(N)
        self.tile = N
        self.orientation = []
        connections = TILE_CONNECTIONS.get(N, [])

        if N == 0:
            self.numConnectionPoints = 0
            self.numStrands = 0
            self.isCrossing = False
            self.connectionDirections = []
        elif N in range(1, 7):
            self.numConnectionPoints = 2
            self.numStrands = 1
            self.isCrossing = False
            self.connectionDirections = connections
        elif N in FOUR_POINT_TILES:
            self.numConnectionPoints = 4
            self.numStrands = 2
            self.isCrossing = N in CROSSING_TILES
            self.connectionDirections = connections

    def exitPath(self, direction):
        """Given a direction of entry, returns the exit direction."""
        assert direction in TILE_FLAT_SET[self.tile]
        return TILE_EXIT[self.tile][direction]

    def show(self, ax=None, resolution=5, color='blue'):
        """Draws the tile on the given matplotlib Axes (creates a new figure if None)."""
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(resolution, resolution))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

        lw = resolution
        N = self.tile

        def arc(cx, cy, r, theta1_deg, theta2_deg):
            ax.add_patch(patches.Arc(
                (cx, cy), 2 * r, 2 * r,
                theta1=theta1_deg, theta2=theta2_deg,
                linewidth=lw, fill=False, color=color,
            ))

        def line(x1, y1, x2, y2):
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle='butt')

        if N == 0:
            pass
        elif N == 1:  # left-down quarter arc, centered at (0, 0)
            arc(0, 0, 0.5, 0, 90)
        elif N == 2:  # right-down quarter arc, centered at (1, 0)
            arc(1, 0, 0.5, 90, 180)
        elif N == 3:  # up-right quarter arc, centered at (1, 1)
            arc(1, 1, 0.5, 180, 270)
        elif N == 4:  # left-up quarter arc, centered at (0, 1)
            arc(0, 1, 0.5, 270, 360)
        elif N == 5:  # horizontal
            line(0, 0.5, 1, 0.5)
        elif N == 6:  # vertical
            line(0.5, 0, 0.5, 1)
        elif N == 7:  # down-left and up-right
            arc(0, 0, 0.5, 0, 90)
            arc(1, 1, 0.5, 180, 270)
        elif N == 8:  # down-right and left-up
            arc(1, 0, 0.5, 90, 180)
            arc(0, 1, 0.5, 270, 360)
        elif N == 9:  # horizontal over, vertical under (break the vertical)
            line(0, 0.5, 1, 0.5)
            line(0.5, 0, 0.5, 0.35)
            line(0.5, 0.65, 0.5, 1)
        elif N == 10:  # vertical over, horizontal under (break the horizontal)
            line(0.5, 0, 0.5, 1)
            line(0, 0.5, 0.35, 0.5)
            line(0.65, 0.5, 1, 0.5)

        return ax

    def isGoing(self, direction):
        """Check if tile has a connection in the given direction."""
        return direction in TILE_FLAT_SET[self.tile]

    def zoom(self, onlyUpDown=False):
        """Returns 3x3 matrix representation of the tile for zooming."""
        N = self.tile
        if N == 9 and onlyUpDown:
            # Twists to center a 10-tile instead
            return TILE_9_UP_DOWN_ZOOM
        return TILE_ZOOM_MAPS.get(N)

    def orient(self, direction):
        """Assigns an orientation to a tile."""
        assert direction in flatten(self.connectionDirections)
        self.orientation = self.orientation + [direction]


class Mosaic:
    def __init__(self, mosaic_matrix):
        """Takes input matrix or list of lists (array)."""
        self.matrixRepresentation = Matrix(mosaic_matrix)
        self.size = self.matrixRepresentation.shape[0]

    def __repr__(self):
        return f"Mosaic of dimension {self.size}."

    def show(self, resolution=5, color='blue'):
        """Outputs a graphic for the mosaic."""
        M = self.matrixRepresentation
        n = self.size
        _, axes = plt.subplots(
            n, n,
            figsize=(resolution, resolution),
            gridspec_kw={'wspace': 0.15, 'hspace': 0.15},
        )
        axes = np.array(axes).reshape(n, n)
        for i in range(n):
            for j in range(n):
                Tile(int(M[i, j])).show(ax=axes[i, j], color=color)

    def matrix(self):
        """Returns the matrix representation of the mosaic."""
        return self.matrixRepresentation

    def directions(self, i, j):
        """Returns the connection points of the (i,j)th tile.

        Position (0,0) is the tile in the upper-left (matrix notation, indexed at 0).
        """
        M = self.matrixRepresentation
        return list(TILE_FLAT_LIST[int(M[i][j])])

    def isSuitablyConnected(self):
        """Checks if all tile edges connect properly."""
        arr = self.matrixRepresentation._data
        n = self.size
        G = TILE_FLAT_SET
        last = n - 1
        for i in range(n):
            for j in range(n):
                dirs = G[int(arr[i, j])]
                if not dirs:  # empty (0) tile: no outgoing connections to check
                    continue

                if 'up' in dirs and (i == 0 or 'down' not in G[int(arr[i - 1, j])]):
                    return False
                if 'left' in dirs and (j == 0 or 'right' not in G[int(arr[i, j - 1])]):
                    return False
                if 'right' in dirs and (j == last or 'left' not in G[int(arr[i, j + 1])]):
                    return False
                if 'down' in dirs and (i == last or 'up' not in G[int(arr[i + 1, j])]):
                    return False
        return True

    def zoom(self, onlyUpDown=False):
        """Zooms by 3x, replaces each tile by a 3x3 isotopy equivalent tile.

        If onlyUpDown=True, all 9 tiles are replaced by twisted 10 tiles (isotopy equivalent).
        """
        M = self.matrixRepresentation
        M_tensored = [[Tile(int(x)).zoom(onlyUpDown) for x in row] for row in M]

        # Unwrap inner 3x3 subtiles
        A = []
        for n in range(len(M_tensored) * 3):
            # Euclidean division: n = floor(n/3)*3 + n%3
            A.append([x[n % 3] for x in M_tensored[n // 3]])

        # Unwrap inner 1x3 subtiles
        B = []
        for row in A:
            flat_row = []
            for subtuple in row:
                flat_row += subtuple
            B.append(flat_row)

        return Mosaic(B)

    def findCrossings(self):
        """Returns a list of coordinates (i,j) in the matrix of crossings (9/10 tiles)."""
        arr = self.matrixRepresentation._data
        # np.nonzero yields row-major order, matching the original double loop.
        rs, cs = np.nonzero((arr == 9) | (arr == 10))
        return [(int(r), int(c)) for r, c in zip(rs, cs)]

    def numCrossings(self):
        """Returns the number of crossings in the mosaic."""
        return len(self.findCrossings())

    def exitPath(self, i, j, direction):
        """Given a tile (i,j) and direction of entry, returns the next tile and exit direction."""
        M = self.matrixRepresentation
        tile_val = int(M[i][j])
        assert direction in TILE_FLAT_SET[tile_val]

        exit_dir = TILE_EXIT[tile_val][direction]

        next_positions = {
            'up': ((i - 1, j), 'up'),
            'down': ((i + 1, j), 'down'),
            'left': ((i, j - 1), 'left'),
            'right': ((i, j + 1), 'right'),
        }
        return list(next_positions[exit_dir])

    def shift(self, i, j, dictionary=False):
        """Returns coordinates of adjacent connected tiles.

        Setting 'dictionary=True' returns a dict mapping directions to tile coordinates.
        """
        assert self.isSuitablyConnected()
        M = self.matrixRepresentation
        N = Tile(M[i][j])
        directions = N.connectionDirections

        def shifter(direction):
            directions_dict = {}
            if 'up' in directions:
                directions_dict['up'] = (i - 1, j)
            if 'down' in directions:
                directions_dict['down'] = (i + 1, j)
            if 'left' in directions:
                directions_dict['left'] = (i, j - 1)
            if 'right' in directions:
                directions_dict['right'] = (i, j + 1)
            return directions_dict

        if N.tile not in FOUR_POINT_TILES:
            directions_dict = shifter(directions)
        else:
            # Directions is a list of lists here, for each strand
            directions_dict = [shifter(strand_directions) for strand_directions in directions]

        if dictionary:
            return directions_dict
        if isinstance(directions_dict, list):
            # 4-point tile: flatten per-strand dicts into a single list of coords
            return [coord for strand_dict in directions_dict for coord in strand_dict.values()]
        return list(directions_dict.values())

    def walk(self, crossing, direction, pathList=False, tangent=False):
        """Given a crossing and direction, returns crossing reached and orientation demanded.

        W.walk(W.walk(crossing, direction)[0], W.walk(crossing, direction)[1])
        is actually just the identity, returns (crossing, direction) as expected.
        """
        all_crossings = self.findCrossings()
        assert crossing in all_crossings

        M = self.matrixRepresentation
        # CAREFUL: pos_x, pos_y are row, col -- not Cartesian coords!
        (pos_x, pos_y) = crossing

        direction_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }

        # Step off the starting crossing in the given direction.
        current_direction = direction
        dx, dy = direction_deltas[current_direction]
        pos_x += dx
        pos_y += dy

        path = [crossing, (pos_x, pos_y)]

        # Walk forward, using each tile's exitPath to follow the correct strand
        # (important for tiles 7 and 8, which have two independent strands),
        # until we land on a crossing (9 or 10).
        while int(M[pos_x][pos_y]) not in CROSSING_TILES:
            entrance = opposite(current_direction)
            current_direction = TILE_EXIT[int(M[pos_x][pos_y])][entrance]
            dx, dy = direction_deltas[current_direction]
            pos_x += dx
            pos_y += dy
            path.append((pos_x, pos_y))

        # Incidence = the edge of the final crossing we entered through.
        incidence = opposite(current_direction)

        if pathList:
            return path
        elif tangent:
            return (pos_x, pos_y), opposite(incidence)
        return (pos_x, pos_y), incidence

    def arcList(self):
        """Run walk on each crossing and with condition pathList=True, remove duplicates."""
        # TODO: Create graph based on crossings - each vertex should have degree 4 (4-regular)
        # This is a singular knot representation; orientations indicate knot
        pass

    def strandOf(self, tile, direction=None, direction_tracking=False, verbose=False):
        """Traces a complete strand through the mosaic starting from the given tile.

        Returns empty list if tile is empty (0 tile).
        """
        tile_type = int(self.matrixRepresentation[tile[0]][tile[1]])
        if tile_type == 0:
            return []

        if direction is None:
            # Order preserved (TILE_FLAT_LIST mirrors flatten()) so the
            # random.choice draw matches the pre-optimization RNG sequence.
            directions = list(TILE_FLAT_LIST[tile_type])
            direction = opposite(random.choice(directions))

        start_tile = tile
        start_direction = direction
        path = []

        tile, direction = self.exitPath(start_tile[0], start_tile[1], opposite(start_direction))
        path.append((tile, direction))

        # Keep track of initial direction to handle 2-strand tile starting points
        while not (tile == start_tile and direction == start_direction):
            tile, direction = self.exitPath(tile[0], tile[1], opposite(direction))
            path.append((tile, direction))

        if verbose:
            direction_tracking = True
            for step in path:
                print(f"Went {step[1]} into tile {step[0]}.")

        if direction_tracking:
            return path
        return [tile for tile, direction in path]

    def strandMatrix(self):
        """Returns a matrix showing the number of strands at each position."""
        # Vectorized lookup: index the strand-count table by tile value.
        return Matrix(_NUM_STRANDS_LUT[self.matrixRepresentation._data])

    def strandOrientationAt(self, tile, previous_tile):
        """Returns the induced orientation on a tile based on entering from previous_tile."""
        if previous_tile[0] < tile[0]:
            return 'down'
        elif previous_tile[0] > tile[0]:
            return 'up'
        elif previous_tile[1] < tile[1]:
            return 'right'
        return 'left'

    def strands(self):
        """Returns all strands (applies when there are multiple connected components)."""
        strand_list = []
        M = self.matrixRepresentation
        remaining = {}

        for tile in M.nonzero_positions():
            tile_value = int(M[tile[0], tile[1]])
            T = Tile(tile_value)
            if T.numStrands == 1:
                remaining[tile] = [frozenset(T.connectionDirections)]
            else:
                remaining[tile] = [frozenset(s) for s in T.connectionDirections]

        while remaining:
            start = min(remaining.keys())
            strand_candidates = sorted(tuple(sorted(s)) for s in remaining[start])
            chosen = strand_candidates[0]
            start_direction = opposite(chosen[0])
            strand = self.strandOf(start, direction=start_direction, direction_tracking=True)
            strand_list.append([tile for tile, direction in strand])

            strand_length = len(strand)
            for k in range(strand_length):
                tile = tuple(strand[k][0])
                entry_dir = strand[k][1]
                exit_dir = strand[(k + 1) % strand_length][1]
                strand_used = frozenset({opposite(entry_dir), exit_dir})

                if tile not in remaining:
                    continue
                try:
                    remaining[tile].remove(strand_used)
                except ValueError:
                    continue
                if not remaining[tile]:
                    del remaining[tile]

        return strand_list

    def numComponents(self):
        """Returns the number of connected components in the mosaic."""
        assert self.isSuitablyConnected()
        return len(self.strands())

    def unknot_check(self):
        """Returns True if this one-component mosaic is detected as the unknot."""
        if not self.isSuitablyConnected():
            raise ValueError("unknot_check requires a suitably connected mosaic")

        num_components = self.numComponents()
        if num_components != 1:
            raise ValueError(f"unknot_check only works for knots; got {num_components} components")

        if self.numCrossings() == 0:
            return True

        try:
            import spherogram
        except ImportError as exc:
            raise ImportError("unknot_check requires spherogram or snappy to be installed") from exc

        K = spherogram.Link(pdCode(self))

        # We assume total_rank = 1 implies the knot K must be trivial.
        K.simplify("global")
        return int(K.knot_floer_homology()["total_rank"]) == 1

    def localFrames(self):
        """Returns the tile above/right (as pairs) for each crossing in the mosaic."""
        crossings = self.findCrossings()
        frames = []
        for crossing in crossings:
            shift_dict = self.shift(crossing[0], crossing[1], True)
            frames.append((shift_dict['up'], shift_dict['right']))
        return frames

    def planarDiagramCode(self):
        """Returns output compatible with SageMath Links package.

        TODO: Implement for https://doc.sagemath.org/html/en/reference/knots/sage/knots/link.html
        """
        pass

    def flip(self):
        """Flips the mosaic upside-down while maintaining tile connections."""
        M = self.matrixRepresentation
        flipped_matrix = M[::-1, :].copy()

        # Map tiles to their upside-down counterparts
        flip_map = {1: 4, 4: 1, 2: 3, 3: 2, 7: 8, 8: 7}

        for i in range(self.size):
            for j in range(self.size):
                tile_val = int(flipped_matrix[i, j])
                if tile_val in flip_map:
                    flipped_matrix[i, j] = flip_map[tile_val]

        return Mosaic(flipped_matrix)

    def potential_tiles(self, i, j):
        """Returns a list of potential tile insertions based on surrounding connections.

        Checks up-down/left-right open connections around the (i,j)th tile.
        """
        necessary_connections = []
        top_boundary = False
        bottom_boundary = False
        left_boundary = False
        right_boundary = False
        boundary_tile = False

        # Check above
        if i == 0:
            top_boundary = True
        elif self.directions(i - 1, j) == []:
            top_boundary = True
        elif 'down' in self.directions(i - 1, j):
            necessary_connections.append('up')
        else:
            top_boundary = True

        # Check below
        if i == self.size - 1:
            bottom_boundary = True
        elif 'up' in self.directions(i + 1, j):
            necessary_connections.append('down')

        # Check left
        if j == 0:
            left_boundary = True
        elif 'right' in self.directions(i, j - 1):
            necessary_connections.append('left')
        elif self.directions(i, j - 1) == []:
            left_boundary = True
        else:
            left_boundary = True

        # Check right
        if j == self.size - 1:
            right_boundary = True
        elif 'left' in self.directions(i, j + 1):
            necessary_connections.append('right')

        # Find tiles that satisfy necessary connections
        required = set(necessary_connections)
        tile_set = [
            tile_num for tile_num in range(11)
            if required.issubset(TILE_FLAT_SET[tile_num])
        ]

        # Remove tiles that would go into boundaries
        if top_boundary:
            tile_set = [t for t in tile_set if t not in TILES_GOING_UP]
        if bottom_boundary:
            tile_set = [t for t in tile_set if t not in TILES_GOING_DOWN]
        if left_boundary:
            tile_set = [t for t in tile_set if t not in TILES_GOING_LEFT]
        if right_boundary:
            tile_set = [t for t in tile_set if t not in TILES_GOING_RIGHT]

        if top_boundary or bottom_boundary or left_boundary or right_boundary:
            boundary_tile = True

        # If no necessary connections and not on boundary, only allow empty tile
        if necessary_connections == [] and not boundary_tile:
            tile_set = [0]

        return tile_set
    
    def combine_components(self,tile=-1, _depth=0):
        if _depth > 5000:
            raise ValueError("Could not generate mosaic satisfying constraints after 5000 attempts")
        
        M = self.matrixRepresentation
        strands = self.strands()

        if tile == -1:    # If tile is not set, pick a random nonzero tile to start on
            longest_strand = max(strands, key=len)
            tile = random.choice(longest_strand)
        
        strand = self.strandOf(tile)
        _strandMatrix = self.strandMatrix()

        for tile in strand:
            _strandMatrix[tile[0],tile[1]] -= 1

        for tile in strand:
            if int(_strandMatrix[tile[0],tile[1]]) == int(1):
                tile_type = int(M[tile[0],tile[1]])
                if tile_type == int(7):
                    M[tile[0],tile[1]] = 8
                elif tile_type == int(8):
                    M[tile[0],tile[1]] = 7
                elif tile_type == int(9):
                    M[tile[0],tile[1]] = random.choice([7,8])
                elif tile_type == int(10):
                    M[tile[0],tile[1]] = random.choice([7,8])

                return Mosaic(M).combine_components(tile, _depth+1)
        
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if (i,j) not in strand:
                    M[i,j] = 0

        return Mosaic(M)


def random_mosaic(
    dimension,
    suitably_connected=True,
    num_crossings=-1,
    num_components=-1,
    unknot=None,
    _depth=0,
):
    """Generates a random mosaic with optional constraints.

    Args:
        dimension: Size of the mosaic (n x n)
        suitably_connected: If True, ensures all tiles connect properly
        num_crossings: Required number of crossings (-1 for any)
        num_components: Required number of components (-1 for any)
        unknot: If True, require an unknot; if False, require a
            non-unknot knot; None skips this check
    """
    if unknot is not None and not suitably_connected:
        raise ValueError("unknot requires suitably_connected=True")

    attempt = _depth
    while attempt <= 5000:
        # Generate base mosaic
        if suitably_connected:
            # Fill a single Mosaic in place. potential_tiles only reads the
            # already-committed cells, so reusing one Mosaic (instead of
            # rebuilding it -- and deep-copying its array -- for every cell)
            # is behavior-identical while avoiding O(n^2) array copies/attempt.
            M = Mosaic(_zeros(dimension, dimension))
            cells = M.matrixRepresentation
            for i in range(dimension):
                for j in range(dimension):
                    cells[i, j] = random.choice(M.potential_tiles(i, j))
        else:
            template = np.random.randint(0, 11, size=(dimension, dimension))
            M = Mosaic(template)

        # Check constraints (if given)
        crossing_validity = (num_crossings == -1) or (M.numCrossings() == num_crossings)
        component_validity = (num_components == -1) or (M.numComponents() == num_components)
        unknot_validity = True
        if unknot is not None:
            try:
                unknot_validity = M.unknot_check() == unknot
            except ValueError:
                unknot_validity = False

        if crossing_validity and component_validity and unknot_validity:
            return M
        attempt += 1

    raise ValueError("Could not generate mosaic satisfying constraints after 5000 attempts")


def tangleConstructor(value, flip=False):
    """Creates a rational tangle mosaic for the given value.

    Args:
        value: The tangle value (oo for infinity, 0, or any integer)
        flip: If True, presents the tangle upside down (necessary for longer tangles)
    """
    if value == oo:
        return Mosaic([[7]])
    if value == 0:
        return Mosaic([[8]])

    def jordan_block_modified(eigenvalue, size, flip=False):
        try:
            size = int(size)
        except (TypeError, ValueError):
            raise TypeError(f"size of block needs to be an integer, not {size}")
        if size < 0:
            raise ValueError(f"size of block must be nonnegative, not {size}")

        block = _diagonal_matrix([eigenvalue] * size)

        if flip:
            for i in range(size - 1):
                block[i, i + 1] = 1
            for i in range(size):
                if i > 0:
                    block[i, i - 1] = 3
            return block
        else:
            for i in range(size - 1):
                block[i, i + 1] = 4
            for i in range(size):
                if i > 0:
                    block[i, i - 1] = 2
            return block[::-1, :].copy()

    if value > 0:
        return Mosaic(jordan_block_modified(10, value, flip=flip))
    if value < 0:
        return Mosaic(jordan_block_modified(9, -value, flip=flip))


def tangleJoin(tangle_list):
    """Joins two tangles together.

    Note: Currently only supports joining exactly two tangles.
    """
    assert len(tangle_list) == 2

    def tangleConnector(n, m, direction):
        assert direction in ['bottom-right', 'top-left']

        if direction == 'bottom-right':
            row = [6] + [0 for _ in range(m - 1)]
            matrix_data = [row for _ in range(n - 1)] + [[4] + [0 for _ in range(m - 1)]]
            return Matrix(matrix_data)
        elif direction == 'top-left':
            row = [0 for _ in range(m)]
            matrix_data = [row for _ in range(n - 1)] + [[2] + [5 for _ in range(m - 1)]]
            return Matrix(matrix_data)

    tangle0 = tangleConstructor(tangle_list[0])
    tangle1 = tangleConstructor(tangle_list[1])
    tangle0_flipped = tangleConstructor(tangle_list[0], flip=True)

    block = _block_matrix([
        [tangleConnector(tangle1.size, tangle0.size, 'top-left'), tangle1.matrix()],
        [tangle0_flipped.matrix(), tangleConnector(tangle0.size, tangle1.size, 'bottom-right')]
    ])

    return Mosaic(block)


def orientedGaussCode(M):
    """Generates oriented Gauss code for SageMath Link() compatibility.

    Returns the code in the format expected by SageMath's Link class.
    """
    def pick_starting_tile(M):
        """Ensures starting tile is not a crossing/hyperbolic tile."""
        strand_matrix = M.strandMatrix()
        for i in range(M.size):
            for j in range(M.size):
                if strand_matrix[i, j] == 1:
                    return (i, j)

    def crossing_handedness(tile_type, orientation_pair):
        """Determines the handedness (+1 or -1) of a crossing."""
        assert tile_type in CROSSING_TILES
        sorted_pair = sorted(orientation_pair)

        if tile_type == 9:
            if sorted_pair in [["right", "up"], ["down", "left"]]:
                return 1
            if sorted_pair in [["down", "right"], ["left", "up"]]:
                return -1
        elif tile_type == 10:
            if sorted_pair in [["left", "up"], ["down", "right"]]:
                return 1
            if sorted_pair in [["down", "left"], ["right", "up"]]:
                return -1

    def over_under(tile_type, orientation, numeric=False):
        """Determines if the strand goes over or under at a crossing."""
        assert tile_type in CROSSING_TILES

        if tile_type == 9:
            positioning = "under" if orientation in ["up", "down"] else "over"
        elif tile_type == 10:
            positioning = "over" if orientation in ["up", "down"] else "under"

        if numeric:
            return 1 if positioning == "over" else -1
        return positioning

    path = M.strandOf(pick_starting_tile(M))
    path = list(enumerate(path))

    crossings = M.findCrossings()
    appearances = []

    for c in crossings:
        for index, tile in path:
            if tile == c:
                # (tile_type, coord, index, previous_coord)
                appearances.append((int(M.matrixRepresentation[c]), c, index, path[index - 1][1]))

    appearances.sort()

    orientations = []
    for appearance in appearances:
        tile, coord, index, prev_coord = appearance
        entrance = M.strandOrientationAt(coord, prev_coord)
        # (index, tile, coord, strand orientation, over/under)
        orientations.append((index, tile, coord, entrance, over_under(tile, entrance, numeric=True)))

    # Ensure crossings are in correct order
    crossings = list(dict.fromkeys([crossing for index, tile, crossing, orientation, positioning in orientations]))

    crossing_orientations = []
    for c in crossings:
        tile_type = int(M.matrixRepresentation[c])
        orientation_pair = [entrance for index, tile, crossing, entrance, positioning in orientations if crossing == c]
        crossing_orientations.append(crossing_handedness(tile_type, orientation_pair))

    orientations.sort()

    # Generate the filter (already ordered by traversal)
    code_filter = [
        (crossings.index(crossing) + 1) * positioning
        for index, tile, crossing, entrance, positioning in orientations
    ]

    # Format for Link() compatibility: [[traversal_code], handedness_list]
    return [[code_filter], crossing_orientations]


def pdCode(M):
    """Generates a planar diagram (PD) code for the mosaic.

    Output is a list of 4-tuples compatible with spherogram.Link(...), e.g.
    Link([[1,4,2,5],[3,6,4,1],[5,2,6,3]]) for the trefoil.

    Convention: each crossing is ``[a, b, c, d]`` where the four arc labels
    appear counterclockwise around the crossing (viewed from above), starting
    from the incoming under-strand arc. Arc labels run from 1 to 2*n_crossings.

    The output is deterministic: given the same mosaic, pdCode returns the
    same PD code on every call.

    Raises:
        ValueError: if the mosaic has no crossings (an empty PD code does not
            define a link in spherogram).
    """
    if M.numCrossings() == 0:
        raise ValueError(
            "Mosaic has no crossings; PD code cannot represent a crossing-free link. "
            "Use spherogram.Link('0_1') or an equivalent name instead."
        )

    matrix_rep = M.matrixRepresentation
    crossings = sorted(tuple(c) for c in M.findCrossings())
    all_visits = {c: [] for c in crossings}
    arc_counter = 0

    # CCW port order when viewed from above (standard math orientation):
    # east -> north -> west -> south, i.e. right -> up -> left -> down.
    ccw_order = ['right', 'up', 'left', 'down']

    # Per-tile list of still-unwalked strands, each stored as a frozenset of
    # its two port names. Keys are in row-major order (tuples compare that way).
    remaining = {}
    for i in range(M.size):
        for j in range(M.size):
            tile_val = int(matrix_rep[i, j])
            if tile_val == 0:
                continue
            T = Tile(tile_val)
            if T.numStrands == 1:
                remaining[(i, j)] = [frozenset(T.connectionDirections)]
            else:
                remaining[(i, j)] = [frozenset(s) for s in T.connectionDirections]

    # Walk every strand (= link component) deterministically.
    while remaining:
        start = min(remaining.keys())
        # Pick the lex-smallest remaining strand for reproducibility.
        strand_candidates = sorted(tuple(sorted(s)) for s in remaining[start])
        chosen = strand_candidates[0]
        # chosen == (port_a, port_b); arbitrarily enter via port_a.
        start_direction = opposite(chosen[0])

        walk = M.strandOf(start, direction=start_direction, direction_tracking=True)

        # At walk[k] we entered via opposite(walk[k][1]) and exit via
        # walk[(k+1) % len][1] (the direction moved to reach the next tile).
        crossing_visits_this_strand = []
        L = len(walk)
        for k in range(L):
            tile_t = tuple(walk[k][0])
            entry_dir = walk[k][1]
            exit_dir = walk[(k + 1) % L][1]
            strand_used = frozenset({opposite(entry_dir), exit_dir})

            if tile_t in remaining:
                # Remove the exact matching strand; there is at most one match.
                try:
                    remaining[tile_t].remove(strand_used)
                except ValueError:
                    pass
                if not remaining[tile_t]:
                    del remaining[tile_t]

            if tile_t in all_visits:
                crossing_visits_this_strand.append((tile_t, entry_dir))

        num_visits = len(crossing_visits_this_strand)
        if num_visits == 0:
            continue

        # Arcs are labeled globally 1..2n. The outgoing arc from visit k is
        # strand_arcs[k]; the incoming arc is strand_arcs[k-1] (cyclically).
        strand_arcs = [arc_counter + i + 1 for i in range(num_visits)]
        arc_counter += num_visits

        for idx, (tile, direc) in enumerate(crossing_visits_this_strand):
            arc_in = strand_arcs[(idx - 1) % num_visits]
            arc_out = strand_arcs[idx]
            entry_port = opposite(direc)

            tile_type = int(matrix_rep[tile[0], tile[1]])
            if tile_type == 9:
                # tile 9: horizontal strand over, vertical strand under.
                is_over = direc in ('left', 'right')
            else:  # tile_type == 10
                # tile 10: vertical strand over, horizontal strand under.
                is_over = direc in ('up', 'down')

            all_visits[tile].append({
                'entry_port': entry_port,
                'is_over': is_over,
                'arc_in': arc_in,
                'arc_out': arc_out,
            })

    pd_code = []
    for c in crossings:
        visits = all_visits[c]
        assert len(visits) == 2, (
            f"Expected exactly 2 visits at crossing {c}, got {len(visits)}. "
            "Is the mosaic suitably connected?"
        )

        # Each visit contributes arcs at two opposite ports of the crossing.
        port_to_arc = {}
        for v in visits:
            port_to_arc[v['entry_port']] = v['arc_in']
            port_to_arc[opposite(v['entry_port'])] = v['arc_out']
        assert len(port_to_arc) == 4, (
            f"Crossing {c} did not receive arcs at all four ports: {port_to_arc}"
        )

        under = next(v for v in visits if not v['is_over'])
        a_port = under['entry_port']
        a_idx = ccw_order.index(a_port)

        pd_tuple = [port_to_arc[ccw_order[(a_idx + i) % 4]] for i in range(4)]
        pd_code.append(pd_tuple)

    return pd_code


# Example code:
# M = [[0,2,1,0,0],[2,9,10,1,0],[3,10,9,10,1],[0,3,7,8,4],[0,0,3,4,0]]; W = Mosaic(M);
# W.matrix()
# W.show()
# W.isSuitablyConnected()

# W = Mosaic(M).zoom()
# W.walk((4,7), 'right', pathList = True) # Putting 'True' provides the pathing

# hopf = Mosaic([[0,2,1,0],[2,9,10,1],[3,10,10,4],[0,3,4,0]]); hopfBig = hopf.zoom(); hopfBig.show(10)
# hopfBig.strandOf((4,4),'up')
# hopfBig.strandOf((4,4),'left')
# These are two different strands (knots) in the hopf! Going left/going right at the crossing determines what was taken.

# hopfBig.shift(3,4, dictionary = True) # Returns directions of tiles *connected to*

# W = Mosaic([(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),...])
# W.strandOf((4,4), direction = 'right').count((4,4)) == 2
# This indicates the crossing was visited twice in the walk.
