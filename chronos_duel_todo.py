from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


Vec = Tuple[int, int]
ROOT = Path(__file__).resolve().parent
ATLAS_PATH = sorted(ROOT.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[0]

TILE = 58
GRID_W = 12
GRID_H = 9
GRID_X = 32
GRID_Y = 96
PANEL_X = GRID_X + GRID_W * TILE + 34
WIDTH = PANEL_X + 390
HEIGHT = 700
FPS = 60

PLAYER = 0
ENEMY = 1
MAX_HP = 12
MAX_ENERGY = 5
MINE_COST = 2
MINE_DAMAGE = 3

DIRS: Dict[str, Vec] = {
    "U": (0, -1),
    "D": (0, 1),
    "L": (-1, 0),
    "R": (1, 0),
}

ACTION_LABELS = {
    "MOVE_U": "Move up",
    "MOVE_D": "Move down",
    "MOVE_L": "Move left",
    "MOVE_R": "Move right",
    "STRIKE": "Chrono strike",
    "BEAM": "Time beam",
    "MINE": "Mine",
    "WAIT": "Charge",
}


@dataclass(frozen=True)
class DuelState:
    px: int
    py: int
    ex: int
    ey: int
    php: int
    ehp: int
    penergy: int
    eenergy: int
    shards: int
    turn: int
    t: int
    pmines: int = 0
    emines: int = 0
    pprevx: int = -1
    pprevy: int = -1
    eprevx: int = -1
    eprevy: int = -1

    @property
    def player_pos(self) -> Vec:
        return self.px, self.py

    @property
    def enemy_pos(self) -> Vec:
        return self.ex, self.ey


@dataclass
class AiReport:
    algorithm: str
    action: str
    score: float
    nodes: int
    depth: int
    elapsed_ms: float


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    command: str

    def contains(self, pos: Vec) -> bool:
        return self.rect.collidepoint(pos)


@dataclass
class Projectile:
    start: Vec
    end: Vec
    owner: int
    age: float = 0.0
    duration: float = 0.34


@dataclass
class UnitMotion:
    start: Vec
    end: Vec
    age: float = 0.0
    duration: float = 0.22


def manhattan(a: Vec, b: Vec) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pos_bit(pos: Vec) -> int:
    x, y = pos
    return 1 << (y * GRID_W + x)


def bit_positions(mask: int) -> List[Vec]:
    out: List[Vec] = []
    for y in range(GRID_H):
        for x in range(GRID_W):
            if mask & pos_bit((x, y)):
                out.append((x, y))
    return out


class TextureAtlas:
    def __init__(self, path: Path) -> None:
        self.sheet = pygame.image.load(str(path)).convert_alpha()
        self.tile_floor: List[pygame.Surface] = []
        self.tile_wall: List[pygame.Surface] = []
        self.tile_distortion: List[pygame.Surface] = []
        self.player_idle: List[pygame.Surface] = []
        self.enemy_spider: List[pygame.Surface] = []
        self.assets: Dict[str, pygame.Surface] = {}
        self._load()

    def crop(self, rect: Tuple[int, int, int, int]) -> pygame.Surface:
        return self.sheet.subsurface(pygame.Rect(rect).clip(self.sheet.get_rect())).copy()

    def with_colorkey(self, surf: pygame.Surface) -> pygame.Surface:
        keyed = surf.convert()
        keyed.set_colorkey((0, 0, 0))
        return keyed.convert_alpha()

    def crop_scaled(
        self,
        rect: Tuple[int, int, int, int],
        size: Tuple[int, int],
        transparent: bool = False,
        smooth: bool = True,
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
        transparent: bool = False,
        smooth: bool = True,
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
        ]
        distortion_rects = [
            (1033, 1023, 174, 172),
            (1226, 1023, 174, 172),
            (1033, 1215, 174, 172),
            (1226, 1215, 174, 172),
        ]
        player_rects = [
            (42, 25, 106, 158),
            (234, 25, 107, 158),
            (427, 25, 107, 158),
            (625, 25, 106, 158),
        ]
        enemy_rects = [
            (1893, 439, 114, 154),
        ]
        self.tile_floor = [self.crop_scaled(r, (TILE, TILE)) for r in floor_rects]
        self.tile_wall = [self.crop_scaled(r, (TILE, TILE)) for r in wall_rects]
        self.tile_distortion = [self.crop_scaled(r, (TILE, TILE)) for r in distortion_rects]
        self.player_idle = [self.crop_fit(r, (43, 54), transparent=True, smooth=False) for r in player_rects]
        self.enemy_spider = [self.crop_fit(r, (43, 54), transparent=True, smooth=False) for r in enemy_rects]
        self.assets["pad"] = self.crop_fit((427, 1047, 162, 132), (54, 46), transparent=True)
        self.assets["shard"] = self.crop_fit((1106, 679, 44, 75), (30, 42), transparent=True)
        self.assets["core"] = self.crop_fit((1040, 1594, 160, 160), (84, 84), transparent=True)
        self.assets["rift"] = self.crop_fit((1494, 848, 131, 154), (52, 58), transparent=True)
        self.assets["blast"] = self.crop_fit((1150, 498, 72, 66), (38, 34), transparent=True)
        self.assets["mine_player"] = self.crop_fit((1722, 648, 77, 132), (30, 40), transparent=True)
        self.assets["mine_enemy"] = self.crop_fit((1911, 851, 69, 124), (30, 40), transparent=True)

    def get(self, name: str) -> pygame.Surface:
        return self.assets[name]


class DuelWorld:
    layout = [
        "..F.#...F...",
        ".##.#.##..#.",
        ".P..~...P...",
        "...##..#..F.",
        "..F..C..F...",
        ".#..#..##...",
        ".P..~...P...",
        "...##.#.#...",
        ".S....#...E.",
    ]

    def __init__(self) -> None:
        self.terrain: List[List[str]] = []
        self.shards: List[Vec] = []
        self.pads: List[Vec] = []
        self.start_player: Vec = (1, 8)
        self.start_enemy: Vec = (10, 8)
        self.core: Vec = (5, 4)
        for y, row in enumerate(self.layout):
            out: List[str] = []
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
                    self.shards.append((x, y))
                elif ch == "S":
                    self.start_player = (x, y)
                elif ch == "E":
                    self.start_enemy = (x, y)
                elif ch == "C":
                    self.core = (x, y)
                out.append(terrain)
            self.terrain.append(out)
        self.shard_ids = {pos: i for i, pos in enumerate(self.shards)}

    def initial_state(self) -> DuelState:
        return DuelState(
            self.start_player[0],
            self.start_player[1],
            self.start_enemy[0],
            self.start_enemy[1],
            MAX_HP,
            MAX_HP,
            2,
            2,
            0,
            PLAYER,
            0,
        )

    def in_bounds(self, pos: Vec) -> bool:
        x, y = pos
        return 0 <= x < GRID_W and 0 <= y < GRID_H

    def terrain_at(self, pos: Vec) -> str:
        x, y = pos
        return self.terrain[y][x]

    def passable(self, pos: Vec) -> bool:
        return self.in_bounds(pos) and self.terrain_at(pos) != "#"

    def pad_exit(self, pos: Vec) -> Vec:
        if pos not in self.pads:
            return pos
        index = self.pads.index(pos)
        return self.pads[(index + 2) % len(self.pads)]

    def terminal(self, state: DuelState) -> bool:
        return state.php <= 0 or state.ehp <= 0

    def actor_pos(self, state: DuelState) -> Vec:
        return state.player_pos if state.turn == PLAYER else state.enemy_pos

    def target_pos(self, state: DuelState) -> Vec:
        return state.enemy_pos if state.turn == PLAYER else state.player_pos

    def previous_pos(self, state: DuelState) -> Optional[Vec]:
        if state.turn == PLAYER:
            return None if state.pprevx < 0 else (state.pprevx, state.pprevy)
        return None if state.eprevx < 0 else (state.eprevx, state.eprevy)

    def legal_actions(self, state: DuelState) -> List[str]:
        if self.terminal(state):
            return []
        actions: List[str] = []
        actor = self.actor_pos(state)
        target = self.target_pos(state)
        previous = self.previous_pos(state)
        energy = state.penergy if state.turn == PLAYER else state.eenergy
        if energy < MAX_ENERGY:
            actions.append("WAIT")
        for name, (dx, dy) in DIRS.items():
            entered = (actor[0] + dx, actor[1] + dy)
            if not self.passable(entered):
                continue
            moved = self.pad_exit(entered)
            if previous is not None and (entered == previous or moved == previous):
                continue
            if moved != target:
                actions.append(f"MOVE_{name}")
        if manhattan(actor, target) <= 1:
            actions.append("STRIKE")
        if energy > 0 and self.line_of_sight(actor, target, max_range=4):
            actions.append("BEAM")
        if energy >= MINE_COST and not self.mine_at(state, actor):
            actions.append("MINE")
        if not actions:
            actions.append("WAIT")
        return actions

    def apply_action(self, state: DuelState, action: str) -> DuelState:
        px, py = state.player_pos
        ex, ey = state.enemy_pos
        php, ehp = state.php, state.ehp
        penergy, eenergy = state.penergy, state.eenergy
        shards = state.shards
        pmines, emines = state.pmines, state.emines
        pprevx, pprevy = state.pprevx, state.pprevy
        eprevx, eprevy = state.eprevx, state.eprevy
        actor_is_player = state.turn == PLAYER
        actor = state.player_pos if actor_is_player else state.enemy_pos

        if action.startswith("MOVE_"):
            dx, dy = DIRS[action[-1]]
            entered = (actor[0] + dx, actor[1] + dy)
            moved = self.pad_exit(entered)
            if actor_is_player:
                pprevx, pprevy = actor
                px, py = moved
            else:
                eprevx, eprevy = actor
                ex, ey = moved
            if self.terrain_at(entered) == "~":
                if actor_is_player:
                    php -= 1
                else:
                    ehp -= 1
            php, ehp, pmines, emines = self.trigger_mine(entered, actor_is_player, php, ehp, pmines, emines)
            if moved != entered:
                php, ehp, pmines, emines = self.trigger_mine(moved, actor_is_player, php, ehp, pmines, emines)
            if moved in self.shard_ids and not (shards & (1 << self.shard_ids[moved])):
                shards |= 1 << self.shard_ids[moved]
                if actor_is_player:
                    penergy = min(MAX_ENERGY, penergy + 2)
                    php = min(MAX_HP, php + 1)
                else:
                    eenergy = min(MAX_ENERGY, eenergy + 2)
                    ehp = min(MAX_HP, ehp + 1)
        elif action == "STRIKE":
            if actor_is_player:
                ehp -= 2
            else:
                php -= 2
        elif action == "BEAM":
            if actor_is_player:
                penergy = max(0, penergy - 1)
                ehp -= 3
            else:
                eenergy = max(0, eenergy - 1)
                php -= 3
        elif action == "MINE":
            bit = pos_bit(actor)
            if actor_is_player:
                penergy = max(0, penergy - MINE_COST)
                pmines |= bit
            else:
                eenergy = max(0, eenergy - MINE_COST)
                emines |= bit
        elif action == "WAIT":
            if actor_is_player:
                penergy = min(MAX_ENERGY, penergy + 1)
            else:
                eenergy = min(MAX_ENERGY, eenergy + 1)

        return DuelState(
            px,
            py,
            ex,
            ey,
            max(0, php),
            max(0, ehp),
            penergy,
            eenergy,
            shards,
            1 - state.turn,
            (state.t + 1) % 16,
            pmines,
            emines,
            pprevx,
            pprevy,
            eprevx,
            eprevy,
        )

    def mine_at(self, state: DuelState, pos: Vec) -> bool:
        bit = pos_bit(pos)
        return bool((state.pmines | state.emines) & bit)

    def trigger_mine(
        self,
        pos: Vec,
        actor_is_player: bool,
        php: int,
        ehp: int,
        pmines: int,
        emines: int,
    ) -> Tuple[int, int, int, int]:
        bit = pos_bit(pos)
        if not ((pmines | emines) & bit):
            return php, ehp, pmines, emines
        pmines &= ~bit
        emines &= ~bit
        if actor_is_player:
            php -= MINE_DAMAGE
        else:
            ehp -= MINE_DAMAGE
        return php, ehp, pmines, emines

    def line_of_sight(self, a: Vec, b: Vec, max_range: int) -> bool:
        if manhattan(a, b) > max_range:
            return False
        if a[0] != b[0] and a[1] != b[1]:
            return False
        dx = 0 if a[0] == b[0] else (1 if b[0] > a[0] else -1)
        dy = 0 if a[1] == b[1] else (1 if b[1] > a[1] else -1)
        cur = (a[0] + dx, a[1] + dy)
        while cur != b:
            if not self.passable(cur):
                return False
            cur = (cur[0] + dx, cur[1] + dy)
        return True

    def evaluate_enemy(self, state: DuelState) -> float:
        # Utility from Chaos's (the enemy's) perspective: positive is good
        # for Chaos, negative is good for Chronos (the player).
        if state.php <= 0:
            return 10000 + state.ehp * 20
        if state.ehp <= 0:
            return -10000 - state.php * 20

        distance = manhattan(state.player_pos, state.enemy_pos)

        # HP is what actually wins the duel, so it dominates the score.
        # We weight it a bit more heavily than the baseline (55 vs 45) and
        # make the weight itself react to how close either side is to
        # dying: HP swings matter more when someone is low, since that's
        # when a couple of points decide the fight.
        hp_diff = state.ehp - state.php
        low_hp_urgency = 1.0 + max(0, (6 - min(state.php, state.ehp))) * 0.08
        score = hp_diff * 55 * low_hp_urgency

        # Energy fuels Beam and Mine, both real damage sources, so being
        # ahead on energy is a genuine advantage, not just a number.
        score += (state.eenergy - state.penergy) * 12

        # Being adjacent enables Strike (cheap, reliable 2 damage), so
        # distance itself is only mildly penalized; the bigger reward for
        # being close is captured by the Strike-range bonus below.
        score -= distance * 4
        if distance <= 1:
            score += 26

        # Line-of-sight + energy means Beam is available *this turn*, which
        # is a strong threat/opportunity. We only credit it while energy is
        # actually available, since a state with sight but no energy can't
        # act on it yet.
        enemy_can_beam = self.line_of_sight(state.enemy_pos, state.player_pos, 4) and state.eenergy > 0
        player_can_beam = self.line_of_sight(state.player_pos, state.enemy_pos, 4) and state.penergy > 0
        if enemy_can_beam:
            score += 40
        if player_can_beam:
            score -= 34

        # Mines are positional threats: a mine sitting near the opponent is
        # good for whoever planted it (they'll likely step on it), and a
        # mine sitting near yourself is a liability you'd want to avoid or
        # already be clear of. We scale the bonus/penalty by how close the
        # relevant unit is, so a mine that's about to be triggered matters
        # far more than a wall paper threat 5 tiles away.
        for mine in bit_positions(state.emines):
            score += max(0, 5 - manhattan(state.player_pos, mine)) * 11
            score -= max(0, 3 - manhattan(state.enemy_pos, mine)) * 6
        for mine in bit_positions(state.pmines):
            score -= max(0, 5 - manhattan(state.enemy_pos, mine)) * 11
            score += max(0, 3 - manhattan(state.player_pos, mine)) * 6

        # Uncollected shards heal + refill energy, so whoever is closer to
        # an unclaimed shard has a resource-race advantage.
        for i, shard in enumerate(self.shards):
            if state.shards & (1 << i):
                continue
            score += manhattan(state.player_pos, shard) - manhattan(state.enemy_pos, shard)

        return score


def ordered_actions(world: DuelWorld, state: DuelState) -> List[str]:
    # Move ordering for alpha-beta: the earlier a strong move is tried at
    # each node, the sooner alpha/beta can tighten and cause a cutoff. A
    # static "type priority" (baseline's approach) is a decent guess, but a
    # cheap one-ply lookahead is a much better guess in practice, since it
    # actually accounts for the current HP/energy/position instead of
    # assuming Beam is always better than Strike.
    #
    # We score each resulting child state with evaluate_enemy() from the
    # perspective of whoever is about to move (the mover wants a high score
    # for themselves), then sort best-first. This ordering doesn't change
    # what the *search* eventually decides -- alpha-beta with any legal
    # ordering returns the same value as plain minimax -- it only changes
    # how many branches get pruned before that value is found.
    actions = world.legal_actions(state)
    if len(actions) <= 1:
        return actions

    mover_is_player = state.turn == PLAYER

    def move_score(action: str) -> float:
        child = world.apply_action(state, action)
        value = world.evaluate_enemy(child)
        # evaluate_enemy() is from Chaos's (enemy's) perspective; flip the
        # sign if the mover is actually the player, so "higher is better
        # for the side that just moved" in both cases.
        return value if not mover_is_player else -value

    return sorted(actions, key=move_score, reverse=True)


def choose_ai_action(world: DuelWorld, state: DuelState, algorithm: str, perspective: int = ENEMY) -> AiReport:
    depth = {"Minimax": 3, "Alpha-Beta": 5, "Expectimax": 4}.get(algorithm, 4)
    begin = time.perf_counter()
    nodes = 0

    def terminal_value(s: DuelState, d: int) -> Optional[float]:
        nonlocal nodes
        if d == 0 or world.terminal(s):
            nodes += 1
            value = world.evaluate_enemy(s)
            return value if perspective == ENEMY else -value
        return None

    def minimax(s: DuelState, d: int) -> float:
        value = terminal_value(s, d)
        if value is not None:
            return value
        actions = world.legal_actions(s)
        if s.turn == perspective:
            return max(minimax(world.apply_action(s, action), d - 1) for action in actions)
        return min(minimax(world.apply_action(s, action), d - 1) for action in actions)

    def alpha_beta(s: DuelState, d: int, alpha: float, beta: float) -> float:
        # Same guaranteed result as minimax(), but branches that can no
        # longer affect the final decision are skipped. Trying strong moves
        # first (via ordered_actions) makes alpha/beta tighten faster,
        # which means more of the remaining branches get cut off early --
        # that's the whole payoff of move ordering here.
        value = terminal_value(s, d)
        if value is not None:
            return value
        actions = ordered_actions(world, s)
        if s.turn == perspective:
            best = -math.inf
            for action in actions:
                best = max(best, alpha_beta(world.apply_action(s, action), d - 1, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break  # beta cutoff: the minimizer above already has a better option
            return best
        best = math.inf
        for action in actions:
            best = min(best, alpha_beta(world.apply_action(s, action), d - 1, alpha, beta))
            beta = min(beta, best)
            if alpha >= beta:
                break  # alpha cutoff: the maximizer above already has a better option
        return best

    def expectimax(s: DuelState, d: int) -> float:
        # Like Minimax, but the non-perspective side is modeled as picking
        # uniformly at random among its legal actions instead of playing
        # optimally against us -- useful when the opponent isn't assumed to
        # be a perfect adversary.
        value = terminal_value(s, d)
        if value is not None:
            return value
        actions = world.legal_actions(s)
        if s.turn == perspective:
            return max(expectimax(world.apply_action(s, action), d - 1) for action in actions)
        return sum(expectimax(world.apply_action(s, action), d - 1) for action in actions) / len(actions)

    best_action = "WAIT"
    best_score = -math.inf
    for action in ordered_actions(world, state):
        child = world.apply_action(state, action)
        if algorithm == "Minimax":
            score = minimax(child, depth - 1)
        elif algorithm == "Expectimax":
            score = expectimax(child, depth - 1)
        else:
            score = alpha_beta(child, depth - 1, -math.inf, math.inf)
        if score > best_score:
            best_action = action
            best_score = score
    elapsed_ms = (time.perf_counter() - begin) * 1000.0
    return AiReport(algorithm, best_action, best_score, nodes, depth, elapsed_ms)


class ChronosDuel:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.world = DuelWorld()
        self.atlas = TextureAtlas(ATLAS_PATH)
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.big_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.state = self.world.initial_state()
        self.player_algorithm = "Minimax"
        self.enemy_algorithm = "Alpha-Beta"
        self.player_control = "Manual"
        self.enemy_control = "AI"
        self.show_guide = True
        self.ai_report: Optional[AiReport] = None
        self.status = "Chronos manual turn"
        self.clock_accum = 0.0
        self.enemy_think_timer = 0.0
        self.auto_step_timer = 0.0
        self.projectiles: List[Projectile] = []
        self.player_motion: Optional[UnitMotion] = None
        self.enemy_motion: Optional[UnitMotion] = None
        self.player_hit_timer = 0.0
        self.enemy_hit_timer = 0.0
        self.last_action = ""
        self.buttons = self.make_buttons()

    def make_buttons(self) -> List[Button]:
        labels = [
            ("P:Man", "P_Manual"),
            ("P:Mini", "P_Minimax"),
            ("P:A-B", "P_Alpha-Beta"),
            ("P:Exp", "P_Expectimax"),
            ("C:Man", "E_Manual"),
            ("C:Mini", "E_Minimax"),
            ("C:A-B", "E_Alpha-Beta"),
            ("C:Exp", "E_Expectimax"),
            ("Guide", "GUIDE"),
            ("Reset", "RESET"),
        ]
        out: List[Button] = []
        x = PANEL_X + 24
        y = 310
        for i, (label, command) in enumerate(labels):
            rect = pygame.Rect(x + (i % 4) * 78, y + (i // 4) * 46, 72, 32)
            out.append(Button(rect, label, command))
        return out

    def reset(self) -> None:
        self.state = self.world.initial_state()
        self.ai_report = None
        self.enemy_think_timer = 0.0
        self.auto_step_timer = 0.0
        self.projectiles.clear()
        self.player_motion = None
        self.enemy_motion = None
        self.player_hit_timer = 0.0
        self.enemy_hit_timer = 0.0
        self.last_action = ""
        self.refresh_turn_status()

    def set_algorithm(self, side: int, algorithm: str) -> None:
        if side == PLAYER:
            self.player_algorithm = algorithm
            self.player_control = "AI"
            self.status = f"Chronos AI: {algorithm}"
        else:
            self.enemy_algorithm = algorithm
            self.enemy_control = "AI"
            self.status = f"Chaos AI: {algorithm}"
        self.enemy_think_timer = 0.12 if self.state.turn == side and not self.world.terminal(self.state) else 0.0

    def set_manual(self, side: int) -> None:
        if side == PLAYER:
            self.player_control = "Manual"
            self.status = "Chronos manual"
        else:
            self.enemy_control = "Manual"
            self.status = "Chaos manual"
        if self.state.turn == side:
            self.enemy_think_timer = 0.0
        self.refresh_turn_status()

    def side_name(self, side: int) -> str:
        return "Chronos" if side == PLAYER else "Chaos"

    def side_control(self, side: int) -> str:
        return self.player_control if side == PLAYER else self.enemy_control

    def side_algorithm(self, side: int) -> str:
        return self.player_algorithm if side == PLAYER else self.enemy_algorithm

    def refresh_turn_status(self) -> None:
        if self.world.terminal(self.state):
            self.status = "Chaos wins" if self.state.php <= 0 else "You win"
            return
        side = self.state.turn
        name = self.side_name(side)
        if self.side_control(side) == "AI":
            self.status = f"{name} AI thinking..."
        else:
            self.status = f"{name} manual turn"

    def player_action(self, action: str) -> None:
        if self.animation_busy():
            return
        side = self.state.turn
        if self.side_control(side) != "Manual" or self.world.terminal(self.state):
            return
        if action not in self.world.legal_actions(self.state):
            self.status = "Action blocked"
            return
        self.apply_action_with_effects(action)
        self.last_action = f"{self.side_name(side)}: {ACTION_LABELS[action]}"
        if not self.world.terminal(self.state) and self.side_control(self.state.turn) == "AI":
            self.enemy_think_timer = 0.42
        self.refresh_turn_status()

    def run_ai_turn(self) -> None:
        if self.animation_busy():
            return
        side = self.state.turn
        if self.side_control(side) != "AI" or self.world.terminal(self.state):
            return
        try:
            report = choose_ai_action(self.world, self.state, self.side_algorithm(side), side)
        except NotImplementedError as exc:
            if side == PLAYER:
                self.player_control = "Manual"
            else:
                self.enemy_control = "Manual"
            self.ai_report = AiReport(self.side_algorithm(side), "WAIT", 0.0, 0, 0, 0.0)
            self.status = str(exc)
            return
        self.ai_report = report
        self.apply_action_with_effects(report.action)
        self.last_action = f"{self.side_name(side)} AI: {ACTION_LABELS[report.action]}"
        self.refresh_turn_status()

    def apply_action_with_effects(self, action: str) -> None:
        old_state = self.state
        owner = old_state.turn
        actor_start = old_state.player_pos if owner == PLAYER else old_state.enemy_pos
        if action == "BEAM":
            start = actor_start
            end = old_state.enemy_pos if owner == PLAYER else old_state.player_pos
            self.projectiles.append(Projectile(start, end, owner))
        new_state = self.world.apply_action(old_state, action)
        if action.startswith("MOVE_"):
            dx, dy = DIRS[action[-1]]
            entered = (actor_start[0] + dx, actor_start[1] + dy)
            actor_end = new_state.player_pos if owner == PLAYER else new_state.enemy_pos
            visual_end = entered if actor_end != entered else actor_end
            motion = UnitMotion(actor_start, visual_end)
            if owner == PLAYER:
                self.player_motion = motion
            else:
                self.enemy_motion = motion
        self.state = new_state
        if action != "BEAM":
            if new_state.php < old_state.php:
                self.player_hit_timer = 0.24
            if new_state.ehp < old_state.ehp:
                self.enemy_hit_timer = 0.24

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.contains(event.pos):
                    if button.command == "RESET":
                        self.reset()
                    elif button.command == "GUIDE":
                        self.show_guide = not self.show_guide
                    elif button.command == "P_Manual":
                        self.set_manual(PLAYER)
                    elif button.command == "E_Manual":
                        self.set_manual(ENEMY)
                    elif button.command.startswith("P_"):
                        self.set_algorithm(PLAYER, button.command[2:])
                    elif button.command.startswith("E_"):
                        self.set_algorithm(ENEMY, button.command[2:])
                    else:
                        self.set_algorithm(ENEMY, button.command)
                    return True
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_1:
                self.set_algorithm(ENEMY, "Minimax")
            elif event.key == pygame.K_2:
                self.set_algorithm(ENEMY, "Alpha-Beta")
            elif event.key == pygame.K_3:
                self.set_algorithm(ENEMY, "Expectimax")
            elif event.key == pygame.K_4:
                self.set_algorithm(PLAYER, "Minimax")
            elif event.key == pygame.K_5:
                self.set_algorithm(PLAYER, "Alpha-Beta")
            elif event.key == pygame.K_6:
                self.set_algorithm(PLAYER, "Expectimax")
            elif event.key == pygame.K_7:
                self.set_manual(PLAYER)
            elif event.key == pygame.K_8:
                self.set_manual(ENEMY)
            elif event.key == pygame.K_h:
                self.show_guide = not self.show_guide
            elif event.key in (pygame.K_UP, pygame.K_w):
                self.player_action("MOVE_U")
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.player_action("MOVE_D")
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.player_action("MOVE_L")
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.player_action("MOVE_R")
            elif event.key in (pygame.K_f, pygame.K_z):
                self.player_action("STRIKE")
            elif event.key in (pygame.K_e, pygame.K_x):
                self.player_action("BEAM")
            elif event.key == pygame.K_m:
                self.player_action("MINE")
            elif event.key == pygame.K_SPACE:
                self.player_action("WAIT")
        return True

    def update(self, dt: float) -> None:
        self.clock_accum += dt
        self.player_hit_timer = max(0.0, self.player_hit_timer - dt)
        self.enemy_hit_timer = max(0.0, self.enemy_hit_timer - dt)
        self.update_unit_motion(dt)
        self.update_projectiles(dt)
        if self.animation_busy():
            return
        if self.enemy_think_timer > 0:
            self.enemy_think_timer -= dt
            if self.enemy_think_timer <= 0:
                self.run_ai_turn()
        elif not self.world.terminal(self.state) and self.side_control(self.state.turn) == "AI":
            self.enemy_think_timer = 0.28

    def animation_busy(self) -> bool:
        return self.player_motion is not None or self.enemy_motion is not None or bool(self.projectiles)

    def update_unit_motion(self, dt: float) -> None:
        if self.player_motion is not None:
            self.player_motion.age += dt
            if self.player_motion.age >= self.player_motion.duration:
                self.player_motion = None
        if self.enemy_motion is not None:
            self.enemy_motion.age += dt
            if self.enemy_motion.age >= self.enemy_motion.duration:
                self.enemy_motion = None

    def update_projectiles(self, dt: float) -> None:
        alive: List[Projectile] = []
        for projectile in self.projectiles:
            projectile.age += dt
            if projectile.age >= projectile.duration:
                if projectile.owner == PLAYER:
                    self.enemy_hit_timer = 0.22
                else:
                    self.player_hit_timer = 0.22
            else:
                alive.append(projectile)
        self.projectiles = alive

    def draw(self) -> None:
        self.screen.fill((11, 17, 27))
        self.draw_header()
        self.draw_board()
        self.draw_panel()
        self.draw_end_overlay()
        pygame.display.flip()

    def draw_header(self) -> None:
        pygame.draw.rect(self.screen, (8, 13, 21), (0, 0, WIDTH, 74))
        title = self.big_font.render("Chronos Duel: Adversarial Search", True, (231, 246, 255))
        self.screen.blit(title, (GRID_X, 25))
        self.screen.blit(self.atlas.get("core"), (GRID_X + 610, 5))

    def draw_board(self) -> None:
        board = pygame.Rect(GRID_X - 8, GRID_Y - 8, GRID_W * TILE + 16, GRID_H * TILE + 16)
        pygame.draw.rect(self.screen, (5, 10, 17), board, border_radius=8)
        pygame.draw.rect(self.screen, (50, 91, 112), board, 2, border_radius=8)
        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = (x, y)
                rect = self.cell_rect(pos)
                terrain = self.world.terrain_at(pos)
                self.screen.blit(self.atlas.tile_floor[(x + y * 2) % len(self.atlas.tile_floor)], rect)
                if terrain == "#":
                    wall = self.atlas.tile_wall[(x * 3 + y) % len(self.atlas.tile_wall)]
                    self.screen.blit(wall, rect)
                    shade = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                    shade.fill((0, 0, 0, 70))
                    self.screen.blit(shade, rect)
                    pygame.draw.rect(self.screen, (102, 132, 152), rect.inflate(-4, -4), 2)
                elif terrain == "~":
                    dist = self.atlas.tile_distortion[(x + y + int(self.clock_accum * 7)) % len(self.atlas.tile_distortion)]
                    self.screen.blit(dist, rect)
                    pygame.draw.rect(self.screen, (199, 74, 255), rect.inflate(-4, -4), 2)
                elif terrain == "P":
                    pad = self.atlas.get("pad")
                    pygame.draw.circle(self.screen, (63, 255, 164), rect.center, 25, 2)
                    self.screen.blit(pad, pad.get_rect(center=rect.center))
                pygame.draw.rect(self.screen, (5, 13, 20), rect, 1)
        self.draw_shards()
        self.draw_core()
        self.draw_units()
        self.draw_mines()
        self.draw_projectiles()

    def draw_shards(self) -> None:
        for i, pos in enumerate(self.world.shards):
            if self.state.shards & (1 << i):
                continue
            rect = self.cell_rect(pos)
            shard = self.atlas.get("shard")
            bob = int(math.sin(self.clock_accum * 4 + i) * 3)
            self.screen.blit(shard, shard.get_rect(center=(rect.centerx, rect.centery + bob)))

    def draw_core(self) -> None:
        rect = self.cell_rect(self.world.core)
        pygame.draw.circle(self.screen, (255, 226, 88), rect.center, 31, 2)
        core = self.atlas.get("core")
        self.screen.blit(core, core.get_rect(center=rect.center))

    def draw_mines(self) -> None:
        for owner, mask, color, asset_name in (
            (PLAYER, self.state.pmines, (78, 238, 255), "mine_player"),
            (ENEMY, self.state.emines, (255, 90, 190), "mine_enemy"),
        ):
            image = self.atlas.get(asset_name)
            for i, pos in enumerate(bit_positions(mask)):
                rect = self.cell_rect(pos)
                pulse = int(math.sin(self.clock_accum * 7 + i + owner) * 3)
                base = (rect.centerx, rect.centery + 12)
                pygame.draw.circle(self.screen, color, base, 13 + pulse, 2)
                self.screen.blit(image, image.get_rect(center=base))

    def draw_units(self) -> None:
        player_center = self.unit_center(self.state.player_pos, self.player_motion)
        player = self.atlas.player_idle[int(self.clock_accum * 2) % len(self.atlas.player_idle)]
        self.screen.blit(player, player.get_rect(midbottom=(player_center[0], player_center[1] + TILE // 2 - 4)))
        pygame.draw.circle(self.screen, (70, 238, 255), player_center, 26, 1)
        enemy_center = self.unit_center(self.state.enemy_pos, self.enemy_motion)
        enemy = self.atlas.enemy_spider[int(self.clock_accum * 8) % len(self.atlas.enemy_spider)]
        self.screen.blit(enemy, enemy.get_rect(center=enemy_center))
        pygame.draw.circle(self.screen, (255, 75, 183), enemy_center, 29, 1)
        if self.player_hit_timer > 0:
            pygame.draw.circle(self.screen, (255, 255, 255), player_center, 32, 3)
        if self.enemy_hit_timer > 0:
            pygame.draw.circle(self.screen, (255, 255, 255), enemy_center, 34, 3)

    def unit_center(self, logical_pos: Vec, motion: Optional[UnitMotion]) -> Vec:
        if motion is None:
            return self.cell_rect(logical_pos).center
        start = self.cell_rect(motion.start).center
        end = self.cell_rect(motion.end).center
        t = min(1.0, motion.age / motion.duration)
        ease = t * t * (3 - 2 * t)
        x = start[0] + (end[0] - start[0]) * ease
        y = start[1] + (end[1] - start[1]) * ease
        y -= math.sin(t * math.pi) * 5
        return int(x), int(y)

    def draw_projectiles(self) -> None:
        blast = self.atlas.get("blast")
        for projectile in self.projectiles:
            start = self.cell_rect(projectile.start).center
            end = self.cell_rect(projectile.end).center
            t = min(1.0, projectile.age / projectile.duration)
            ease = 1 - (1 - t) * (1 - t)
            x = start[0] + (end[0] - start[0]) * ease
            y = start[1] + (end[1] - start[1]) * ease
            for i in range(3):
                trail_t = max(0.0, ease - i * 0.08)
                tx = start[0] + (end[0] - start[0]) * trail_t
                ty = start[1] + (end[1] - start[1]) * trail_t
                alpha = 120 - i * 35
                pygame.draw.circle(self.screen, (82, 232, 255, alpha), (int(tx), int(ty)), 7 - i, 0)
            angle = -math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
            image = pygame.transform.rotate(blast, angle)
            if projectile.owner == ENEMY:
                image = pygame.transform.flip(image, True, False)
            self.screen.blit(image, image.get_rect(center=(int(x), int(y))))

    def draw_panel(self) -> None:
        panel = pygame.Rect(PANEL_X, 0, WIDTH - PANEL_X, HEIGHT)
        pygame.draw.rect(self.screen, (10, 16, 25), panel)
        pygame.draw.line(self.screen, (54, 91, 111), (PANEL_X, 0), (PANEL_X, HEIGHT), 2)
        title = self.big_font.render("Rift Tactics", True, (232, 247, 255))
        self.screen.blit(title, (PANEL_X + 24, 34))
        self.draw_bars(PANEL_X + 24, 96)
        self.draw_buttons()
        self.draw_ai_box()

    def draw_end_overlay(self) -> None:
        if not self.world.terminal(self.state):
            return
        if self.state.ehp <= 0 and self.state.php <= 0:
            title = "DRAW"
            color = (232, 238, 245)
        elif self.state.ehp <= 0:
            title = "CHRONOS WINS"
            color = (96, 246, 255)
        else:
            title = "CHAOS WINS"
            color = (255, 88, 188)
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 155))
        self.screen.blit(veil, (0, 0))
        box = pygame.Rect(0, 0, 520, 170)
        box.center = (WIDTH // 2, HEIGHT // 2)
        pygame.draw.rect(self.screen, (12, 24, 36), box, border_radius=10)
        pygame.draw.rect(self.screen, color, box, 3, border_radius=10)
        title_surf = pygame.font.SysFont("segoeui", 46, bold=True).render(title, True, color)
        self.screen.blit(title_surf, title_surf.get_rect(center=(box.centerx, box.y + 62)))
        sub = self.font.render("Press R to reset", True, (220, 235, 244))
        self.screen.blit(sub, sub.get_rect(center=(box.centerx, box.y + 118)))

    def draw_bars(self, x: int, y: int) -> None:
        self.draw_bar(x, y, "Chronos HP", self.state.php, MAX_HP, (74, 226, 255))
        self.draw_bar(x, y + 46, "Chronos Energy", self.state.penergy, MAX_ENERGY, (94, 255, 170))
        self.draw_bar(x, y + 100, "Chaos HP", self.state.ehp, MAX_HP, (255, 84, 184))
        self.draw_bar(x, y + 146, "Chaos Energy", self.state.eenergy, MAX_ENERGY, (198, 104, 255))
        status = self.font.render(self.status, True, (119, 241, 255))
        self.screen.blit(status, (x, y + 196))

    def draw_bar(self, x: int, y: int, label: str, value: int, total: int, color: Tuple[int, int, int]) -> None:
        text = self.small_font.render(f"{label}: {value}/{total}", True, (214, 230, 240))
        self.screen.blit(text, (x, y))
        rect = pygame.Rect(x, y + 20, 302, 10)
        pygame.draw.rect(self.screen, (36, 49, 63), rect, border_radius=5)
        fill = int(rect.w * max(0, value) / total)
        if fill:
            pygame.draw.rect(self.screen, color, (rect.x, rect.y, fill, rect.h), border_radius=5)

    def draw_buttons(self) -> None:
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            active = False
            if button.command == "P_Manual":
                active = self.player_control == "Manual"
            elif button.command == "E_Manual":
                active = self.enemy_control == "Manual"
            elif button.command.startswith("P_"):
                active = self.player_control == "AI" and button.command[2:] == self.player_algorithm
            elif button.command.startswith("E_"):
                active = self.enemy_control == "AI" and button.command[2:] == self.enemy_algorithm
            elif button.command == "GUIDE":
                active = self.show_guide
            hover = button.contains(mouse)
            fill = (17, 42, 54) if not active else (25, 77, 86)
            if hover:
                fill = (30, 67, 84)
            pygame.draw.rect(self.screen, fill, button.rect, border_radius=7)
            border = (99, 255, 201) if active else (80, 145, 166)
            pygame.draw.rect(self.screen, border, button.rect, 2, border_radius=7)
            label = self.small_font.render(button.label, True, (224, 249, 255))
            self.screen.blit(label, label.get_rect(center=button.rect.center))

    def draw_ai_box(self) -> None:
        box = pygame.Rect(PANEL_X + 24, 438, 318, 246)
        pygame.draw.rect(self.screen, (14, 26, 38), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        if self.show_guide:
            lines = [
                "Guide",
                "Goal: drop enemy HP to zero.",
                "Manual side uses WASD / arrows.",
                "Strike F/Z, Beam E/X, Mine M.",
                "Mine costs 2 and hurts anyone.",
                "Charge: Space, only if energy is not full.",
                "No direct return to your last move cell.",
                "P:Man/C:Man switch that side manual.",
                "P/C algorithm buttons switch that side AI.",
            ]
        else:
            lines = [
                "Tactical log",
                f"Last: {self.last_action or '-'}",
                f"Chronos: {self.player_control if self.player_control == 'Manual' else self.player_algorithm}",
                f"Chaos: {self.enemy_control if self.enemy_control == 'Manual' else self.enemy_algorithm}",
                f"Turn: {self.side_name(self.state.turn)}",
            ]
            if self.ai_report:
                report = self.ai_report
                lines += [
                    f"Action: {ACTION_LABELS[report.action]}",
                    f"Depth: {report.depth}",
                    f"Nodes: {report.nodes}",
                    f"Score: {report.score:.1f}",
                    f"Time: {report.elapsed_ms:.1f} ms",
                ]
            else:
                lines += ["AI: waiting for first turn"]
        for i, line in enumerate(lines[:10]):
            color = (226, 241, 249) if i == 0 else (161, 187, 201)
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (box.x + 16, box.y + 16 + i * 22))

    def cell_rect(self, pos: Vec) -> pygame.Rect:
        x, y = pos
        return pygame.Rect(GRID_X + x * TILE, GRID_Y + y * TILE, TILE, TILE)


def main() -> None:
    pygame.init()
    pygame.joystick.quit()
    pygame.display.set_caption("Chronos Duel: Adversarial Search")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game = ChronosDuel(screen)
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