from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pygame


Vec = Tuple[int, int]

ROOT = Path(__file__).resolve().parent


def latest_texture_atlas() -> Path:
    images = sorted(ROOT.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not images:
        return ROOT / "Source.png"
    return images[0]


ATLAS_PATH = latest_texture_atlas()

TILE = 56
GRID_W = 14
GRID_H = 10
GRID_X = 30
GRID_Y = 96
PANEL_X = GRID_X + GRID_W * TILE + 30
WIDTH = PANEL_X + 370
HEIGHT = 720
FPS = 60
BUTTON_W = 50
BUTTON_H = 36

FLOOR_COST = 1.0
DISTORTION_COST = 5.0
RIFT_COST = 2.0
WAIT_COST = 1.0

DIRS: Dict[str, Vec] = {
    "U": (0, -1),
    "D": (0, 1),
    "L": (-1, 0),
    "R": (1, 0),
    "W": (0, 0),
}


@dataclass(frozen=True)
class SearchState:
    x: int
    y: int
    fragments: int
    rifts: int
    t: int

    @property
    def pos(self) -> Vec:
        return self.x, self.y


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    states: List[SearchState]
    actions: List[str]
    cost: float
    visited: int
    elapsed_ms: float
    frontier_peak: int
    message: str


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    command: str
    hotkey: str

    def contains(self, pos: Vec) -> bool:
        return self.rect.collidepoint(pos)


class TextureAtlas:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing texture atlas: {path}")
        self.sheet = pygame.image.load(str(path)).convert_alpha()
        self.tile_floor: List[pygame.Surface] = []
        self.tile_wall: List[pygame.Surface] = []
        self.tile_distortion: List[pygame.Surface] = []
        self.player_idle: List[pygame.Surface] = []
        self.player_walk: Dict[str, List[pygame.Surface]] = {}
        self.enemy_ghost_frames: List[pygame.Surface] = []
        self.enemy_spider_frames: List[pygame.Surface] = []
        self.button_assets: Dict[str, pygame.Surface] = {}
        self.assets: Dict[str, pygame.Surface] = {}
        self._load()

    def crop(self, rect: Tuple[int, int, int, int]) -> pygame.Surface:
        safe = pygame.Rect(rect).clip(self.sheet.get_rect())
        return self.sheet.subsurface(safe).copy()

    def with_colorkey(self, surf: pygame.Surface) -> pygame.Surface:
        keyed = surf.convert()
        keyed.set_colorkey((0, 0, 0))
        return keyed.convert_alpha()

    def crop_scaled(
        self,
        rect: Tuple[int, int, int, int],
        size: Tuple[int, int],
        smooth: bool = False,
        transparent: bool = False,
    ) -> pygame.Surface:
        surf = self.crop(rect)
        if transparent:
            surf = self.with_colorkey(surf)
        scaler = pygame.transform.smoothscale if smooth else pygame.transform.scale
        return scaler(surf, size)

    def crop_fit(
        self,
        rect: Tuple[int, int, int, int],
        box: Tuple[int, int],
        smooth: bool = True,
        transparent: bool = False,
    ) -> pygame.Surface:
        surf = self.crop(rect)
        if transparent:
            surf = self.with_colorkey(surf)
        w, h = surf.get_size()
        scale = min(box[0] / w, box[1] / h)
        size = max(1, int(w * scale)), max(1, int(h * scale))
        scaler = pygame.transform.smoothscale if smooth else pygame.transform.scale
        return scaler(surf, size)

    def _load(self) -> None:
        floor_rects = [
            (4, 1023, 181, 172),
            (205, 1023, 183, 172),
            (4, 1215, 181, 172),
            (205, 1215, 183, 172),
            (4, 1405, 181, 173),
            (205, 1405, 183, 172),
        ]
        wall_rects = [
            (408, 1405, 186, 172),
            (614, 1405, 207, 173),
            (842, 1404, 172, 174),
            (4, 1592, 181, 174),
            (205, 1592, 183, 174),
            (408, 1592, 186, 174),
            (613, 1592, 208, 174),
            (842, 1596, 172, 170),
        ]
        distortion_rects = [
            (1033, 1023, 174, 172),
            (1226, 1023, 174, 172),
            (1033, 1215, 174, 172),
            (1226, 1215, 174, 172),
        ]
        idle_rects = [(42, 25, 106, 158), (234, 25, 107, 158), (427, 25, 107, 158), (625, 25, 106, 158)]
        walk_down_rects = [(42, 25, 106, 158), (234, 25, 107, 158), (427, 25, 107, 158), (625, 25, 106, 158), (865, 25, 107, 158)]
        walk_up_rects = [(1079, 25, 106, 161), (1280, 25, 108, 161), (1485, 25, 108, 161), (1687, 25, 107, 161), (1891, 25, 107, 161)]
        walk_right_rects = [(1078, 231, 105, 157), (1286, 231, 92, 160), (1487, 231, 96, 161), (1694, 230, 92, 161), (1898, 230, 98, 162)]
        walk_left_rects = [(50, 433, 92, 162), (243, 433, 95, 161), (440, 434, 97, 161), (658, 434, 94, 160), (868, 434, 92, 161)]
        ghost_rects = [(21, 630, 137, 191), (224, 629, 137, 192), (427, 634, 135, 187), (636, 625, 136, 196), (847, 627, 135, 194)]
        spider_rects = [(5, 873, 182, 141), (210, 873, 181, 136), (410, 872, 183, 136), (619, 872, 188, 138), (830, 872, 186, 139)]

        self.tile_floor = [self.crop_scaled(r, (TILE, TILE)) for r in floor_rects]
        self.tile_wall = [self.crop_scaled(r, (TILE, TILE)) for r in wall_rects]
        self.tile_distortion = [self.crop_scaled(r, (TILE, TILE)) for r in distortion_rects]
        self.player_idle = [self.crop_fit(r, (43, 54), smooth=False, transparent=True) for r in idle_rects]
        self.player_walk = {
            "D": [self.crop_fit(r, (43, 54), smooth=False, transparent=True) for r in walk_down_rects],
            "U": [self.crop_fit(r, (43, 54), smooth=False, transparent=True) for r in walk_up_rects],
            "R": [self.crop_fit(r, (43, 54), smooth=False, transparent=True) for r in walk_right_rects],
            "L": [self.crop_fit(r, (43, 54), smooth=False, transparent=True) for r in walk_left_rects],
        }
        self.enemy_ghost_frames = [self.crop_fit(r, (50, 58), smooth=False, transparent=True) for r in ghost_rects]
        self.enemy_spider_frames = [self.crop_fit(r, (58, 48), smooth=False, transparent=True) for r in spider_rects]

        self.assets["pad"] = self.crop_fit((427, 1047, 162, 132), (54, 46), smooth=True, transparent=True)
        self.assets["fragment"] = self.crop_fit((1106, 679, 44, 75), (28, 38), smooth=True, transparent=True)
        self.assets["fragment_alt"] = self.crop_fit((1308, 860, 73, 119), (30, 40), smooth=True, transparent=True)
        self.assets["core"] = self.crop_fit((1040, 1594, 160, 160), (86, 86), smooth=True, transparent=True)
        self.assets["core_alt"] = self.crop_fit((1231, 1595, 160, 159), (86, 86), smooth=True, transparent=True)
        self.assets["rift"] = self.crop_fit((1494, 848, 131, 154), (52, 58), smooth=True, transparent=True)
        self.assets["rift_alt"] = self.crop_fit((1703, 1223, 141, 156), (52, 58), smooth=True, transparent=True)
        self.assets["rift_sealed"] = self.crop_fit((1887, 1224, 139, 155), (52, 58), smooth=True, transparent=True)
        self.assets["wisp"] = self.crop_fit((1722, 648, 77, 132), (42, 50), smooth=True, transparent=True)
        self.assets["path_arrow"] = self.crop_scaled((1040, 1410, 158, 156), (TILE, TILE), smooth=True, transparent=True)
        self.assets["path_dash"] = self.crop_scaled((1234, 1410, 157, 156), (TILE, TILE), smooth=True, transparent=True)
        self.assets["visited_overlay"] = self.crop_scaled((1712, 1034, 133, 156), (TILE, TILE), smooth=True, transparent=True)
        self.assets["logo"] = self.crop_fit((1040, 1594, 160, 160), (46, 46), smooth=True, transparent=True)

        button_rects = {
            "BFS": (187, 1773, 144, 118),
            "DFS": (358, 1773, 143, 118),
            "UCS": (528, 1773, 143, 118),
            "Greedy": (698, 1773, 143, 118),
            "A*": (867, 1773, 144, 118),
            "RESET": (1163, 1773, 108, 114),
        }
        self.button_assets = {
            key: self.crop_scaled(rect, (BUTTON_W, BUTTON_H), smooth=True, transparent=True)
            for key, rect in button_rects.items()
        }

    def get(self, name: str) -> pygame.Surface:
        return self.assets[name]


class World:
    layout = [
        "..F.#...~...C.",
        ".##.#.##.~.##.",
        ".P..#....~..P.",
        ".#..###..##...",
        ".#R..~..F..R#.",
        "...##.####....",
        ".F..~....#..~.",
        ".##..P##..##..",
        "...#..R..~..F.",
        "S..#....~.....",
    ]

    enemy_paths: List[List[Vec]] = [
        [(3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (8, 7), (9, 7)],
        [(9, 0), (10, 0), (11, 0), (13, 0), (13, 1), (13, 2), (12, 2), (10, 2)],
        [(0, 4), (0, 5), (1, 5), (2, 5), (2, 6), (3, 6), (3, 5), (1, 5)],
    ]

    def __init__(self) -> None:
        self.terrain: List[List[str]] = []
        self.start: Vec = (0, 0)
        self.core: Vec = (0, 0)
        self.fragments: List[Vec] = []
        self.rifts: List[Vec] = []
        self.pads: List[Vec] = []

        for y, row in enumerate(self.layout):
            terrain_row: List[str] = []
            for x, ch in enumerate(row):
                terrain = "."
                if ch == "#":
                    terrain = "#"
                elif ch == "~":
                    terrain = "~"
                elif ch == "P":
                    terrain = "P"
                    self.pads.append((x, y))
                elif ch == "F":
                    self.fragments.append((x, y))
                elif ch == "R":
                    self.rifts.append((x, y))
                elif ch == "C":
                    self.core = (x, y)
                elif ch == "S":
                    self.start = (x, y)
                terrain_row.append(terrain)
            self.terrain.append(terrain_row)

        self.fragment_ids = {pos: i for i, pos in enumerate(self.fragments)}
        self.rift_ids = {pos: i for i, pos in enumerate(self.rifts)}
        self.rift_requirements = [1, 2, 3]
        self.all_fragments_mask = (1 << len(self.fragments)) - 1
        self.all_rifts_mask = (1 << len(self.rifts)) - 1
        self.period = 1
        for path in self.enemy_paths:
            self.period = math.lcm(self.period, len(path))

    def initial_state(self) -> SearchState:
        return SearchState(self.start[0], self.start[1], 0, 0, 0)

    def in_bounds(self, pos: Vec) -> bool:
        x, y = pos
        return 0 <= x < GRID_W and 0 <= y < GRID_H

    def terrain_at(self, pos: Vec) -> str:
        x, y = pos
        return self.terrain[y][x]

    def passable(self, pos: Vec) -> bool:
        return self.in_bounds(pos) and self.terrain_at(pos) != "#"

    def enemy_positions(self, t: int) -> List[Vec]:
        return [path[t % len(path)] for path in self.enemy_paths]

    def blocked_by_enemy(self, pos: Vec, t: int) -> bool:
        return pos in self.enemy_positions(t)

    def enemy_crosses(self, old_pos: Vec, new_pos: Vec, t: int, nt: int) -> bool:
        for path in self.enemy_paths:
            old_enemy = path[t % len(path)]
            new_enemy = path[nt % len(path)]
            if old_enemy == new_pos and new_enemy == old_pos:
                return True
        return False

    def pad_exit(self, pos: Vec) -> Vec:
        if pos not in self.pads:
            return pos
        index = self.pads.index(pos)
        return self.pads[(index + 1) % len(self.pads)]

    def tile_cost(self, pos: Vec) -> float:
        terrain = self.terrain_at(pos)
        if terrain == "~":
            return DISTORTION_COST
        if pos in self.rift_ids:
            return RIFT_COST
        return FLOOR_COST

    def apply_objectives(self, pos: Vec, fragments: int, rifts: int) -> Tuple[int, int]:
        if pos in self.fragment_ids:
            fragments |= 1 << self.fragment_ids[pos]
        if pos in self.rift_ids:
            rid = self.rift_ids[pos]
            if fragments.bit_count() >= self.rift_requirements[rid]:
                rifts |= 1 << rid
        return fragments, rifts

    def is_goal(self, state: SearchState) -> bool:
        return state.pos == self.core and state.fragments == self.all_fragments_mask and state.rifts == self.all_rifts_mask

    def step(self, state: SearchState, action: str) -> Optional[Tuple[SearchState, float]]:
        dx, dy = DIRS[action]
        old_pos = state.pos
        entry_pos = (state.x + dx, state.y + dy)
        nt = (state.t + 1) % self.period

        if action != "W":
            if not self.passable(entry_pos):
                return None
            new_pos = self.pad_exit(entry_pos)
        else:
            new_pos = old_pos

        if not self.passable(new_pos):
            return None
        if self.blocked_by_enemy(new_pos, nt):
            return None
        if self.enemy_crosses(old_pos, new_pos, state.t, nt):
            return None

        fragments, rifts = self.apply_objectives(new_pos, state.fragments, state.rifts)
        cost = WAIT_COST if action == "W" else self.tile_cost(entry_pos)
        if action != "W" and entry_pos != new_pos:
            cost += 0.5
        return SearchState(new_pos[0], new_pos[1], fragments, rifts, nt), cost

    def neighbors(self, state: SearchState) -> Iterable[Tuple[str, SearchState, float]]:
        for action in ("R", "U", "D", "L", "W"):
            result = self.step(state, action)
            if result is not None:
                yield action, result[0], result[1]

    def heuristic(self, state: SearchState) -> float:
        raise NotImplementedError("TODO: implement an admissible or useful heuristic for Greedy and A* search.")


def manhattan(a: Vec, b: Vec) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct(
    goal: SearchState,
    parent: Dict[SearchState, Tuple[SearchState, str, float]],
) -> Tuple[List[SearchState], List[str], float]:
    states = [goal]
    actions: List[str] = []
    total = 0.0
    cursor = goal
    while cursor in parent:
        prev, action, cost = parent[cursor]
        actions.append(action)
        total += cost
        cursor = prev
        states.append(cursor)
    states.reverse()
    actions.reverse()
    return states, actions, total


def solve(world: World, start: SearchState, algorithm: str, node_limit: int = 90000) -> SearchResult:
    raise NotImplementedError("TODO: implement BFS, DFS, UCS, Greedy, and A* search here.")


def priority(world: World, state: SearchState, g: float, algorithm: str) -> float:
    raise NotImplementedError("TODO: return the frontier priority for UCS, Greedy, and A*.")


def result(
    algorithm: str,
    found: bool,
    states: List[SearchState],
    actions: List[str],
    cost: float,
    visited: int,
    begin: float,
    frontier_peak: int,
    message: str,
) -> SearchResult:
    return SearchResult(algorithm, found, states, actions, cost, visited, (time.perf_counter() - begin) * 1000.0, frontier_peak, message)


class ChronosGame:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.world = World()
        self.atlas = TextureAtlas(ATLAS_PATH)
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.big_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.clock_accum = 0.0
        self.auto_timer = 0.0
        self.auto_interval = 0.14
        self.auto_states: deque[SearchState] = deque()
        self.auto_actions: deque[str] = deque()
        self.path_preview: List[SearchState] = []
        self.path_preview_actions: List[str] = []
        self.last_result: Optional[SearchResult] = None
        self.status = "Manual control"
        self.selected_algorithm = ""
        self.player_direction = "D"
        self.player_motion_timer = 0.0
        self.state = self.world.initial_state()
        self.buttons = self._make_buttons()

    def _make_buttons(self) -> List[Button]:
        labels = [
            ("BFS", "BFS", "1"),
            ("DFS", "DFS", "2"),
            ("UCS", "UCS", "3"),
            ("Greedy", "Greedy", "4"),
            ("A*", "A*", "5"),
            ("Reset", "RESET", "R"),
        ]
        buttons: List[Button] = []
        x = PANEL_X + 24
        y = 292
        gap_x = 16
        gap_y = 12
        for index, (label, command, hotkey) in enumerate(labels):
            row = index // 2
            col = index % 2
            rect = pygame.Rect(x + col * (BUTTON_W + gap_x), y + row * (BUTTON_H + gap_y), BUTTON_W, BUTTON_H)
            buttons.append(Button(rect, label, command, hotkey))
        return buttons

    def reset(self) -> None:
        self.state = self.world.initial_state()
        self.auto_states.clear()
        self.auto_actions.clear()
        self.path_preview = []
        self.path_preview_actions = []
        self.last_result = None
        self.status = "Manual control"
        self.selected_algorithm = ""
        self.player_direction = "D"
        self.player_motion_timer = 0.0

    def run_algorithm(self, algorithm: str) -> None:
        self.status = f"{algorithm} searching..."
        pygame.event.pump()
        try:
            result_obj = solve(self.world, self.state, algorithm)
        except NotImplementedError as exc:
            self.last_result = SearchResult(algorithm, False, [], [], 0.0, 0, 0.0, 0, str(exc))
            self.selected_algorithm = algorithm
            self.path_preview = []
            self.path_preview_actions = []
            self.auto_states.clear()
            self.auto_actions.clear()
            self.status = "TODO: implement search"
            return
        self.last_result = result_obj
        self.selected_algorithm = algorithm
        self.path_preview = result_obj.states[:]
        self.path_preview_actions = result_obj.actions[:]
        self.auto_states.clear()
        self.auto_actions.clear()
        if result_obj.found:
            self.auto_states = deque(result_obj.states[1:])
            self.auto_actions = deque(result_obj.actions)
            steps = max(1, len(result_obj.states) - 1)
            self.auto_interval = 0.14 if steps <= 120 else max(0.015, 10.0 / steps)
            self.status = f"{algorithm} plan ready"
        else:
            self.status = result_obj.message

    def try_action(self, action: str) -> None:
        self.auto_states.clear()
        self.auto_actions.clear()
        self.path_preview = []
        self.path_preview_actions = []
        step = self.world.step(self.state, action)
        if step is None:
            self.status = "Blocked"
            return
        self.state = step[0]
        self._mark_player_motion(action)
        self.status = "Manual control"
        if self.world.is_goal(self.state):
            self.status = "Timeline stabilized"

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.contains(event.pos):
                    if button.command == "RESET":
                        self.reset()
                    else:
                        self.run_algorithm(button.command)
                    return True
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_1:
                self.run_algorithm("BFS")
            elif event.key == pygame.K_2:
                self.run_algorithm("DFS")
            elif event.key == pygame.K_3:
                self.run_algorithm("UCS")
            elif event.key == pygame.K_4:
                self.run_algorithm("Greedy")
            elif event.key == pygame.K_5:
                self.run_algorithm("A*")
            elif event.key in (pygame.K_UP, pygame.K_w):
                self.try_action("U")
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.try_action("D")
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.try_action("L")
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.try_action("R")
            elif event.key == pygame.K_SPACE:
                self.try_action("W")
        return True

    def update(self, dt: float) -> None:
        self.clock_accum += dt
        self.player_motion_timer = max(0.0, self.player_motion_timer - dt)
        self.auto_timer += dt
        if self.auto_states and self.auto_timer >= self.auto_interval:
            self.auto_timer = 0.0
            action = self.auto_actions.popleft() if self.auto_actions else "W"
            self.state = self.auto_states.popleft()
            self._mark_player_motion(action)
            if not self.auto_states:
                self.status = "Timeline stabilized" if self.world.is_goal(self.state) else "Plan finished"

    def _mark_player_motion(self, action: str) -> None:
        if action in ("U", "D", "L", "R"):
            self.player_direction = action
            self.player_motion_timer = max(self.player_motion_timer, 0.16)

    def draw(self) -> None:
        self.screen.fill((14, 21, 31))
        self._draw_background_grid()
        self._draw_board()
        self._draw_panel()
        pygame.display.flip()

    def _draw_background_grid(self) -> None:
        for x in range(0, WIDTH, 32):
            pygame.draw.line(self.screen, (18, 31, 43), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 32):
            pygame.draw.line(self.screen, (18, 31, 43), (0, y), (WIDTH, y), 1)
        pygame.draw.rect(self.screen, (10, 15, 23), (0, 0, WIDTH, 76))
        self.screen.blit(self.atlas.get("logo"), (GRID_X + 360, 14))
        title = self.big_font.render("Temporal Repair Board", True, (230, 244, 255))
        self.screen.blit(title, (GRID_X, 28))

    def _draw_board(self) -> None:
        board_rect = pygame.Rect(GRID_X - 8, GRID_Y - 8, GRID_W * TILE + 16, GRID_H * TILE + 16)
        pygame.draw.rect(self.screen, (5, 10, 17), board_rect, border_radius=8)
        pygame.draw.rect(self.screen, (48, 90, 112), board_rect, 2, border_radius=8)

        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = (x, y)
                rect = self.cell_rect(pos)
                terrain = self.world.terrain_at(pos)
                floor = self.atlas.tile_floor[(x + y * 2) % len(self.atlas.tile_floor)]
                self.screen.blit(floor, rect)
                if terrain == "#":
                    image = self.atlas.tile_wall[(x * 3 + y) % len(self.atlas.tile_wall)]
                    self.screen.blit(image, rect)
                    shade = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                    shade.fill((0, 0, 0, 64))
                    self.screen.blit(shade, rect)
                    pygame.draw.rect(self.screen, (98, 133, 154), rect.inflate(-4, -4), 2)
                elif terrain == "~":
                    image = self.atlas.tile_distortion[(x + y + int(self.clock_accum * 8)) % len(self.atlas.tile_distortion)]
                    self.screen.blit(image, rect)
                    pygame.draw.rect(self.screen, (198, 78, 255), rect.inflate(-4, -4), 2)
                elif terrain == "P":
                    image = self.atlas.get("pad")
                    pulse = int(math.sin(self.clock_accum * 5 + x) * 3)
                    pygame.draw.circle(self.screen, (63, 255, 164), rect.center, 25 + pulse, 2)
                    self.screen.blit(image, image.get_rect(center=rect.center))
                pygame.draw.rect(self.screen, (5, 13, 20), rect, 1)

        self._draw_path_preview()
        self._draw_core()
        self._draw_objectives()
        self._draw_enemies()
        self._draw_player()

    def _draw_path_preview(self) -> None:
        if len(self.path_preview) <= 1:
            return
        for idx, state in enumerate(self.path_preview[1:]):
            rect = self.cell_rect(state.pos)
            action = self.path_preview_actions[idx] if idx < len(self.path_preview_actions) else "W"
            marker = self._path_marker(action, idx)
            self.screen.blit(marker, rect)
            pygame.draw.rect(self.screen, (91, 245, 255), rect.inflate(-10, -10), 1)
            self._draw_direction_arrow(rect, action)

    def _path_marker(self, action: str, index: int) -> pygame.Surface:
        base = self.atlas.get("path_dash" if index % 3 == 0 else "path_arrow")
        if action == "U":
            marker = pygame.transform.rotate(base, 90)
        elif action == "D":
            marker = pygame.transform.rotate(base, -90)
        elif action == "L":
            marker = pygame.transform.rotate(base, 180)
        elif action == "W":
            marker = self.atlas.get("visited_overlay").copy()
        else:
            marker = base.copy()
        marker.set_alpha(115)
        return marker

    def _draw_direction_arrow(self, rect: pygame.Rect, action: str) -> None:
        if action == "W":
            pygame.draw.circle(self.screen, (122, 245, 255), rect.center, 8, 2)
            return
        dx, dy = DIRS[action]
        if dx == 0 and dy == 0:
            return
        start = (rect.centerx - dx * 12, rect.centery - dy * 12)
        end = (rect.centerx + dx * 14, rect.centery + dy * 14)
        pygame.draw.line(self.screen, (143, 246, 255), start, end, 4)
        if action == "U":
            points = [(end[0], end[1] - 8), (end[0] - 7, end[1] + 4), (end[0] + 7, end[1] + 4)]
        elif action == "D":
            points = [(end[0], end[1] + 8), (end[0] - 7, end[1] - 4), (end[0] + 7, end[1] - 4)]
        elif action == "L":
            points = [(end[0] - 8, end[1]), (end[0] + 4, end[1] - 7), (end[0] + 4, end[1] + 7)]
        elif action == "R":
            points = [(end[0] + 8, end[1]), (end[0] - 4, end[1] - 7), (end[0] - 4, end[1] + 7)]
        else:
            return
        pygame.draw.polygon(self.screen, (224, 254, 255), points)

    def _draw_objectives(self) -> None:
        for i, pos in enumerate(self.world.fragments):
            rect = self.cell_rect(pos)
            image = self.atlas.get("fragment" if i % 2 == 0 else "fragment_alt")
            if self.state.fragments & (1 << i):
                ghost = image.copy()
                ghost.set_alpha(55)
                self.screen.blit(ghost, ghost.get_rect(center=rect.center))
            else:
                bob = int(math.sin(self.clock_accum * 4 + i) * 3)
                self.screen.blit(image, image.get_rect(center=(rect.centerx, rect.centery + bob)))

        for i, pos in enumerate(self.world.rifts):
            rect = self.cell_rect(pos)
            sealed = bool(self.state.rifts & (1 << i))
            image_name = "rift_sealed" if sealed else ("rift" if i % 2 == 0 else "rift_alt")
            image = self.atlas.get(image_name).copy()
            color = (77, 255, 166) if sealed else (205, 87, 255)
            self.screen.blit(image, image.get_rect(center=rect.center))
            pygame.draw.circle(self.screen, color, rect.center, 22, 2)
            need = self.world.rift_requirements[i]
            txt = self.small_font.render(str(need), True, (238, 247, 255))
            self.screen.blit(txt, (rect.right - 17, rect.top + 5))

    def _draw_core(self) -> None:
        rect = self.cell_rect(self.world.core)
        glow_radius = 33 + int(math.sin(self.clock_accum * 4) * 4)
        pygame.draw.circle(self.screen, (255, 222, 96), rect.center, glow_radius + 4, 2)
        pygame.draw.circle(self.screen, (30, 209, 231), rect.center, glow_radius, 2)
        core = self.atlas.get("core" if int(self.clock_accum * 4) % 2 == 0 else "core_alt")
        self.screen.blit(core, core.get_rect(center=rect.center))

    def _draw_enemies(self) -> None:
        positions = self.world.enemy_positions(self.state.t)
        for i, pos in enumerate(positions):
            rect = self.cell_rect(pos)
            pulse = 12 + int(math.sin(self.clock_accum * 8 + i) * 3)
            color = (245, 71, 255) if i != 1 else (255, 86, 174)
            pygame.draw.circle(self.screen, color, rect.center, pulse + 5, 1)
            if i == 1:
                frames = self.atlas.enemy_spider_frames
            elif i == 2:
                frames = [self.atlas.get("wisp")]
            else:
                frames = self.atlas.enemy_ghost_frames
            image = frames[int(self.clock_accum * 8 + i) % len(frames)].copy()
            if i == 2:
                image = pygame.transform.flip(image, True, False)
            self.screen.blit(image, image.get_rect(center=rect.center))

    def _draw_player(self) -> None:
        rect = self.cell_rect(self.state.pos)
        if self.player_motion_timer > 0.0:
            frames = self.atlas.player_walk.get(self.player_direction, self.atlas.player_walk["D"])
            image = frames[int(self.clock_accum * 14) % len(frames)]
        else:
            image = self.atlas.player_idle[0]
        shadow = pygame.Surface((42, 12), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
        self.screen.blit(shadow, shadow.get_rect(center=(rect.centerx, rect.bottom - 9)))
        self.screen.blit(image, image.get_rect(midbottom=(rect.centerx, rect.bottom - 4)))

    def _draw_panel(self) -> None:
        panel = pygame.Rect(PANEL_X, 0, WIDTH - PANEL_X, HEIGHT)
        pygame.draw.rect(self.screen, (11, 17, 27), panel)
        pygame.draw.line(self.screen, (52, 88, 109), (PANEL_X, 0), (PANEL_X, HEIGHT), 2)

        emblem = self.atlas.get("logo")
        self.screen.blit(emblem, (PANEL_X + 24, 30))
        header = self.big_font.render("Chronos Weaver", True, (231, 246, 255))
        self.screen.blit(header, (PANEL_X + 84, 38))

        self._draw_progress(PANEL_X + 24, 130)
        self._draw_buttons()
        self._draw_result_box()

    def _draw_progress(self, x: int, y: int) -> None:
        frag_count = self.state.fragments.bit_count()
        rift_count = self.state.rifts.bit_count()
        lines = [
            ("Fragments", frag_count, len(self.world.fragments), (88, 224, 255)),
            ("Rifts", rift_count, len(self.world.rifts), (194, 105, 255)),
        ]
        for i, (label, value, total, color) in enumerate(lines):
            row_y = y + i * 46
            text = self.font.render(f"{label}: {value}/{total}", True, (222, 236, 244))
            self.screen.blit(text, (x, row_y))
            track = pygame.Rect(x, row_y + 24, 294, 10)
            pygame.draw.rect(self.screen, (36, 49, 63), track, border_radius=5)
            fill_w = int(track.w * (value / total))
            if fill_w:
                pygame.draw.rect(self.screen, color, (track.x, track.y, fill_w, track.h), border_radius=5)

        step_text = self.small_font.render(f"Time phase: {self.state.t}/{self.world.period - 1}", True, (155, 179, 193))
        self.screen.blit(step_text, (x, y + 94))
        status = self.font.render(self.status, True, (122, 243, 255))
        self.screen.blit(status, (x, y + 124))

    def _draw_buttons(self) -> None:
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            active = button.command == self.selected_algorithm
            hover = button.contains(mouse)
            asset = self.atlas.button_assets.get(button.command)
            if asset is not None:
                self.screen.blit(asset, button.rect)
            else:
                pygame.draw.rect(self.screen, (24, 51, 68), button.rect, border_radius=7)
                text = self.font.render(button.label, True, (231, 248, 255))
                self.screen.blit(text, text.get_rect(center=button.rect.center))
            border = (105, 255, 203) if active else (78, 143, 166)
            if hover:
                border = (143, 238, 255)
            pygame.draw.rect(self.screen, border, button.rect, 2, border_radius=7)
            hotkey = self.small_font.render(button.hotkey, True, (132, 205, 224))
            self.screen.blit(hotkey, (button.rect.right - 18, button.rect.top + 10))

    def _draw_result_box(self) -> None:
        box = pygame.Rect(PANEL_X + 24, 458, 302, 204)
        pygame.draw.rect(self.screen, (15, 26, 38), box, border_radius=8)
        pygame.draw.rect(self.screen, (50, 83, 101), box, 1, border_radius=8)

        if self.last_result is None:
            lines = [
                "Temporal repair active.",
                "Collect every fragment.",
                "Seal charged rifts.",
                "Reach the Chronos Core.",
            ]
        else:
            r = self.last_result
            steps = max(0, len(r.states) - 1)
            lines = [
                f"Algorithm: {r.algorithm}",
                f"Result: {r.message}",
                f"Steps: {steps}",
                f"Path cost: {r.cost:.1f}",
                f"Visited: {r.visited}",
                f"Frontier peak: {r.frontier_peak}",
                f"Runtime: {r.elapsed_ms:.1f} ms",
            ]

        for i, line in enumerate(lines):
            color = (226, 239, 247) if i == 0 else (160, 184, 198)
            text = self.font.render(line, True, color)
            self.screen.blit(text, (box.x + 18, box.y + 18 + i * 24))

    def cell_rect(self, pos: Vec) -> pygame.Rect:
        x, y = pos
        return pygame.Rect(GRID_X + x * TILE, GRID_Y + y * TILE, TILE, TILE)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Chronos Weaver: Search Algorithms")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game = ChronosGame(screen)
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            running = game.handle_event(event)
            if not running:
                break
        game.update(dt)
        game.draw()
    pygame.quit()


if __name__ == "__main__":
    main()
