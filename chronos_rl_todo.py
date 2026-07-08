from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


Vec = Tuple[int, int]
RState = Tuple[int, int, int]

ROOT = Path(__file__).resolve().parent
ATLAS_PATH = ROOT / "Source.png"

TILE = 58
GRID_W = 9
GRID_H = 8
GRID_X = 34
GRID_Y = 96
PANEL_X = GRID_X + GRID_W * TILE + 34
WIDTH = PANEL_X + 390
HEIGHT = 690
FPS = 60

ACTIONS: Dict[str, Vec] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}
ACTION_ORDER = ("N", "E", "S", "W")
KEY_ACTIONS = {
    pygame.K_UP: "N",
    pygame.K_w: "N",
    pygame.K_DOWN: "S",
    pygame.K_s: "S",
    pygame.K_RIGHT: "E",
    pygame.K_d: "E",
    pygame.K_LEFT: "W",
    pygame.K_a: "W",
}


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    command: str
    hotkey: str

    def contains(self, pos: Vec) -> bool:
        return self.rect.collidepoint(pos)


@dataclass
class EpisodeReport:
    algorithm: str
    steps: int
    total_reward: float
    reached_terminal: bool
    message: str


class TextureAtlas:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing texture atlas: {path}")
        self.sheet = pygame.image.load(str(path)).convert_alpha()
        self.tile_floor: List[pygame.Surface] = []
        self.tile_wall: List[pygame.Surface] = []
        self.tile_distortion: List[pygame.Surface] = []
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
        self.tile_floor = [self.crop_scaled(r, (TILE, TILE)) for r in floor_rects]
        self.tile_wall = [self.crop_scaled(r, (TILE, TILE)) for r in wall_rects]
        self.tile_distortion = [self.crop_scaled(r, (TILE, TILE)) for r in distortion_rects]
        self.assets["agent"] = self.crop_fit((42, 25, 106, 158), (42, 54), transparent=True, smooth=False)
        self.assets["goal"] = self.crop_fit((1040, 1594, 160, 160), (62, 62), transparent=True)
        self.assets["trap"] = self.crop_fit((1494, 848, 131, 154), (52, 58), transparent=True)
        self.assets["shard"] = self.crop_fit((1106, 679, 44, 75), (25, 36), transparent=True)

    def get(self, name: str) -> pygame.Surface:
        return self.assets[name]


class RLEnvironment:
    def __init__(self) -> None:
        self.walls = {(1, 1), (3, 1), (6, 1), (3, 2), (6, 3), (1, 5), (4, 5), (6, 6)}
        self.traps = {(2, 5), (5, 2), (7, 4)}
        self.shards = [(2, 2), (4, 6), (7, 2)]
        self.slow_zones = {(3, 4), (5, 5)}
        self.start = (0, 7)
        self.goal = (8, 0)
        self.required_mask = (1 << len(self.shards)) - 1
        self.state: RState = (self.start[0], self.start[1], 0)
        self.slip = 0.20

    def states(self) -> List[RState]:
        states: List[RState] = []
        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x, y) not in self.walls:
                    for mask in range(self.required_mask + 1):
                        states.append((x, y, mask))
        return states

    def pos(self, state: RState) -> Vec:
        return state[0], state[1]

    def in_bounds(self, pos: Vec) -> bool:
        return 0 <= pos[0] < GRID_W and 0 <= pos[1] < GRID_H

    def passable(self, pos: Vec) -> bool:
        return self.in_bounds(pos) and pos not in self.walls

    def terminal(self, state: RState) -> bool:
        pos = self.pos(state)
        return pos in self.traps or (pos == self.goal and state[2] == self.required_mask)

    def reset(self) -> RState:
        self.state = (self.start[0], self.start[1], 0)
        return self.state

    def shard_mask_at(self, pos: Vec) -> int:
        if pos not in self.shards:
            return 0
        return 1 << self.shards.index(pos)

    def move(self, state: RState, action: str) -> RState:
        dx, dy = ACTIONS[action]
        next_state = (state[0] + dx, state[1] + dy)
        if self.passable(next_state):
            mask = state[2] | self.shard_mask_at(next_state)
            return next_state[0], next_state[1], mask
        return state

    def side_actions(self, action: str) -> Tuple[str, str]:
        index = ACTION_ORDER.index(action)
        return ACTION_ORDER[(index - 1) % 4], ACTION_ORDER[(index + 1) % 4]

    def sample_action(self, action: str) -> str:
        roll = random.random()
        if roll < 1.0 - self.slip:
            return action
        left, right = self.side_actions(action)
        return left if random.random() < 0.5 else right

    def reward(self, previous: RState, state: RState) -> float:
        pos = self.pos(state)
        reward = -0.07
        if pos == self.goal:
            return 14.0 if state[2] == self.required_mask else -2.5
        if pos in self.traps:
            return -8.0
        gained = state[2] & ~previous[2]
        if gained:
            reward += 2.5
        if pos in self.slow_zones:
            reward -= 1.0
        return reward

    def step(self, action: str) -> Tuple[RState, float, bool]:
        previous = self.state
        actual = self.sample_action(action)
        self.state = self.move(self.state, actual)
        reward = self.reward(previous, self.state)
        return self.state, reward, self.terminal(self.state)

    def sample_transition(self, state: RState, action: str) -> Tuple[RState, float, bool]:
        actual = self.sample_action(action)
        next_state = self.move(state, actual)
        reward = self.reward(state, next_state)
        return next_state, reward, self.terminal(next_state)


class QAgent:
    def __init__(self, env: RLEnvironment) -> None:
        self.env = env
        self.alpha = 0.45
        self.gamma = 0.92
        self.epsilon = 0.24
        self.q: Dict[Tuple[RState, str], float] = {}
        for state in self.env.states():
            for action in ACTION_ORDER:
                self.q[(state, action)] = 0.0
        self.algorithm = "Q-Learning"
        self.episodes = 0
        self.last_path: List[Vec] = []

    def label(self) -> str:
        return "Q-Path" if self.algorithm == "Q-Learning" else "Trail Mode"

    def value(self, state: RState) -> float:
        return max(self.q[(state, action)] for action in ACTION_ORDER)

    def best_action(self, state: RState) -> str:
        scored = [(self.q[(state, action)], action) for action in ACTION_ORDER]
        _, action = max(scored, key=lambda item: (item[0], item[1]))
        return action

    def choose_action(self, state: RState) -> str:
        # With probability epsilon, explore by selecting a random action uniformly.
        if random.random() < self.epsilon:
            return random.choice(ACTION_ORDER)
        
        # Otherwise, exploit known values by selecting the current greedy action.
        return self.best_action(state)

    def train_episode(self, max_steps: int = 160) -> EpisodeReport:
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        path = [self.env.pos(state)]

        # For SARSA, the initial action must be selected prior to entering the step loop.
        action = self.choose_action(state)

        while not done and steps < max_steps:
            if self.algorithm == "Q-Learning":
                action = self.choose_action(state)

            next_state, reward, done = self.env.step(action)
            path.append(self.env.pos(next_state))
            total_reward += reward
            steps += 1

            if done:
                target = reward
            elif self.algorithm == "Q-Learning":
                # Off-policy target update using the maximum Q-value of the next state.
                target = reward + self.gamma * max(self.q[(next_state, a)] for a in ACTION_ORDER)
            else:
                # On-policy target update using the action selected by the current policy.
                next_action = self.choose_action(next_state)
                target = reward + self.gamma * self.q[(next_state, next_action)]

            # Update the state-action value based on the computed temporal difference target.
            self.q[(state, action)] += self.alpha * (target - self.q[(state, action)])

            state = next_state
            if self.algorithm == "SARSA" and not done:
                action = next_action

        # Update learning tracking statistics and decay the exploration rate exponentially.
        self.episodes += 1
        self.epsilon = max(0.01, self.epsilon * 0.98)
        self.last_path = path

        return EpisodeReport(
            algorithm=self.label(),
            steps=steps,
            total_reward=total_reward,
            reached_terminal=done,
            message=f"Run complete via {self.algorithm}."
        )

    def train_many(self, count: int) -> EpisodeReport:
        report = EpisodeReport(self.label(), 0, 0.0, False, "")
        for _ in range(count):
            report = self.train_episode()
        report.message = f"Trained {count} runs. Last return {report.total_reward:.2f}."
        return report

    def reset(self) -> None:
        for key in self.q:
            self.q[key] = 0.0
        self.episodes = 0
        self.epsilon = 0.24
        self.last_path = []


class ChronosRLGame:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.atlas = TextureAtlas(ATLAS_PATH)
        self.env = RLEnvironment()
        self.agent = QAgent(self.env)
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.big_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.value_font = pygame.font.SysFont("consolas", 13, bold=True)
        self.buttons = self.make_buttons()
        self.report: Optional[EpisodeReport] = None
        self.pilot_state: RState = (self.env.start[0], self.env.start[1], 0)
        self.pilot_return = 0.0
        self.pilot_steps = 0
        self.game_over = False
        self.end_title = ""
        self.end_message = ""
        self.status = "Train a path, or guide Chronos with arrows/WASD."
        self.time = 0.0
        self.display_mask = 0

    def make_buttons(self) -> List[Button]:
        specs = [
            ("Run", "episode", "1"),
            ("Train50", "train", "2"),
            ("Q-Path", "q", "3"),
            ("Trail", "sarsa", "4"),
            ("Layer", "mask", "M"),
            ("Reset", "reset", "R"),
        ]
        buttons: List[Button] = []
        x = PANEL_X + 24
        y = 340
        for i, (label, command, hotkey) in enumerate(specs):
            rect = pygame.Rect(x + (i % 3) * 104, y + (i // 3) * 44, 92, 34)
            buttons.append(Button(rect, label, command, hotkey))
        return buttons

    def reset_pilot(self) -> None:
        self.pilot_state = (self.env.start[0], self.env.start[1], 0)
        self.pilot_return = 0.0
        self.pilot_steps = 0
        self.game_over = False
        self.end_title = ""
        self.end_message = ""
        self.display_mask = 0

    def run_command(self, command: str) -> None:
        try:
            if command == "episode":
                self.report = self.agent.train_episode()
            elif command == "train":
                self.report = self.agent.train_many(50)
            elif command == "q":
                self.agent.algorithm = "Q-Learning"
                self.status = "Learner set to Q-Path."
                return
            elif command == "sarsa":
                self.agent.algorithm = "SARSA"
                self.status = "Learner set to Trail Mode."
                return
            elif command == "mask":
                self.display_mask = (self.display_mask + 1) % (self.env.required_mask + 1)
                self.status = f"Showing shard-mask layer {self.display_mask:03b}."
                return
            elif command == "reset":
                self.agent.reset()
                self.env.reset()
                self.reset_pilot()
                self.report = None
                self.status = "Memory grid and pilot reset."
                return
        except NotImplementedError as exc:
            self.report = None
            self.status = str(exc)
            return
        if self.report is not None:
            self.status = self.report.message

    def move_pilot(self, action: str) -> None:
        if self.game_over:
            self.status = "Shard trial is over. Reset starts a new run."
            return
        old_state = self.pilot_state
        next_state, reward, done = self.env.sample_transition(old_state, action)
        self.pilot_state = next_state
        self.pilot_return += reward
        self.pilot_steps += 1
        self.display_mask = next_state[2]
        old_pos = self.env.pos(old_state)
        new_pos = self.env.pos(next_state)
        gained = next_state[2] & ~old_state[2]
        if done:
            self.finish_trial(reward)
        elif new_pos == old_pos:
            self.status = f"Chronos held position. Reward {reward:+.2f}."
        elif gained:
            self.status = f"Shard recovered at {new_pos}. Reward {reward:+.2f}."
        elif new_pos == self.env.goal:
            self.status = "The core is locked. Recover every shard first."
        else:
            self.status = f"Chronos moved to {new_pos}. Reward {reward:+.2f}."

    def finish_trial(self, reward: float) -> None:
        self.game_over = True
        pos = self.env.pos(self.pilot_state)
        if pos == self.env.goal and self.pilot_state[2] == self.env.required_mask:
            self.end_title = "Shard Trial Complete"
            self.end_message = "Chronos restored the core with every shard."
        else:
            self.end_title = "Chronos Lost"
            self.end_message = "A rift trap collapsed the run."
        self.status = f"{self.end_title} at {pos}. Final return {self.pilot_return:.2f}."

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.contains(event.pos):
                    if not self.game_over or button.command == "reset":
                        self.run_command(button.command)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if self.game_over and event.key != pygame.K_r:
                self.status = "Shard trial is over. Reset starts a new run."
                return True
            if event.key == pygame.K_1:
                self.run_command("episode")
            elif event.key == pygame.K_2:
                self.run_command("train")
            elif event.key == pygame.K_3:
                self.run_command("q")
            elif event.key == pygame.K_4:
                self.run_command("sarsa")
            elif event.key == pygame.K_m:
                self.run_command("mask")
            elif event.key == pygame.K_r:
                self.run_command("reset")
            elif event.key in KEY_ACTIONS:
                self.move_pilot(KEY_ACTIONS[event.key])
        return True

    def update(self, dt: float) -> None:
        self.time += dt

    def draw(self) -> None:
        self.screen.fill((7, 11, 18))
        self.draw_header()
        self.draw_board()
        self.draw_panel()
        if self.game_over:
            self.draw_end_overlay()
        pygame.display.flip()

    def draw_header(self) -> None:
        pygame.draw.rect(self.screen, (8, 13, 21), (0, 0, WIDTH, 76))
        title = self.big_font.render("Chronos Shard Trial", True, (231, 246, 255))
        self.screen.blit(title, (28, 18))
        subtitle = self.font.render("Learn, collect, survive.", True, (133, 186, 205))
        self.screen.blit(subtitle, (352, 25))

    def draw_board(self) -> None:
        board = pygame.Rect(GRID_X - 8, GRID_Y - 8, GRID_W * TILE + 16, GRID_H * TILE + 16)
        pygame.draw.rect(self.screen, (5, 10, 17), board, border_radius=8)
        pygame.draw.rect(self.screen, (50, 91, 112), board, 2, border_radius=8)
        max_abs = max(1.0, max(abs(self.agent.value(s)) for s in self.env.states()))
        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = (x, y)
                state = (x, y, self.display_mask)
                rect = self.cell_rect(pos)
                if pos in self.env.walls:
                    self.screen.blit(self.atlas.tile_wall[(x + y) % len(self.atlas.tile_wall)], rect)
                    pygame.draw.rect(self.screen, (7, 12, 18), rect, 1)
                    continue
                self.screen.blit(self.atlas.tile_floor[(x * 2 + y * 3) % len(self.atlas.tile_floor)], rect)
                value = self.agent.value(state)
                if value >= 0:
                    shade_color = (38, 171, 151, int(28 + 92 * value / max_abs))
                else:
                    shade_color = (220, 64, 120, int(36 + 92 * abs(value) / max_abs))
                shade = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                shade.fill(shade_color)
                self.screen.blit(shade, rect)
                if pos in self.env.slow_zones:
                    self.screen.blit(self.atlas.tile_distortion[(x + y) % len(self.atlas.tile_distortion)], rect)
                if pos in self.env.traps:
                    self.screen.blit(self.atlas.get("trap"), self.atlas.get("trap").get_rect(center=rect.center))
                if pos in self.env.shards:
                    shard_bit = self.env.shard_mask_at(pos)
                    if not self.display_mask & shard_bit:
                        self.screen.blit(self.atlas.get("shard"), self.atlas.get("shard").get_rect(center=rect.center))
                    else:
                        pygame.draw.circle(self.screen, (108, 255, 184), rect.center, 12, 2)
                if pos == self.env.goal:
                    self.screen.blit(self.atlas.get("goal"), self.atlas.get("goal").get_rect(center=rect.center))
                    if self.display_mask != self.env.required_mask:
                        lock = self.small_font.render("LOCK", True, (255, 120, 144))
                        self.screen.blit(lock, lock.get_rect(center=(rect.centerx, rect.y + 12)))
                else:
                    self.draw_policy(state, rect)
                self.draw_q_values(state, rect)
                pygame.draw.rect(self.screen, (5, 13, 20), rect, 1)
        self.draw_last_path()
        self.draw_pilot()

    def draw_policy(self, state: RState, rect: pygame.Rect) -> None:
        action = self.agent.best_action(state)
        dx, dy = ACTIONS[action]
        start = pygame.Vector2(rect.center)
        end = start + pygame.Vector2(dx, dy) * 16
        pygame.draw.line(self.screen, (82, 238, 255), start, end, 3)
        left = pygame.Vector2(-dy, dx) * 5
        back = pygame.Vector2(dx, dy) * -6
        pygame.draw.polygon(self.screen, (220, 254, 255), [end, end + back + left, end + back - left])

    def draw_q_values(self, state: RState, rect: pygame.Rect) -> None:
        pos = self.env.pos(state)
        if pos in self.env.traps or pos == self.env.goal:
            return
        top = self.value_font.render(f"{self.agent.q[(state, 'N')]:.1f}", True, (215, 234, 242))
        bottom = self.value_font.render(f"{self.agent.q[(state, 'S')]:.1f}", True, (215, 234, 242))
        left = self.value_font.render(f"{self.agent.q[(state, 'W')]:.1f}", True, (215, 234, 242))
        right = self.value_font.render(f"{self.agent.q[(state, 'E')]:.1f}", True, (215, 234, 242))
        self.screen.blit(top, top.get_rect(center=(rect.centerx, rect.y + 10)))
        self.screen.blit(bottom, bottom.get_rect(center=(rect.centerx, rect.bottom - 10)))
        self.screen.blit(left, left.get_rect(center=(rect.x + 16, rect.centery)))
        self.screen.blit(right, right.get_rect(center=(rect.right - 16, rect.centery)))

    def draw_last_path(self) -> None:
        if len(self.agent.last_path) < 2:
            return
        points = [self.cell_rect(pos).center for pos in self.agent.last_path[-28:]]
        for a, b in zip(points, points[1:]):
            pygame.draw.line(self.screen, (255, 225, 90), a, b, 4)
        for point in points:
            pygame.draw.circle(self.screen, (255, 246, 154), point, 4)

    def draw_panel(self) -> None:
        panel = pygame.Rect(PANEL_X, 0, WIDTH - PANEL_X, HEIGHT)
        pygame.draw.rect(self.screen, (10, 16, 25), panel)
        pygame.draw.line(self.screen, (54, 91, 111), (PANEL_X, 0), (PANEL_X, HEIGHT), 2)
        icon = self.atlas.get("agent")
        self.screen.blit(icon, (PANEL_X + 24, 24))
        title = self.big_font.render("Shard Console", True, (231, 246, 255))
        self.screen.blit(title, (PANEL_X + 94, 34))
        self.draw_legend(PANEL_X + 24, 108)
        self.draw_buttons()
        self.draw_report()

    def draw_legend(self, x: int, y: int) -> None:
        box = pygame.Rect(x, y, 330, 206)
        pygame.draw.rect(self.screen, (12, 24, 35), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        title = self.small_font.render("Shard trial brief", True, (118, 220, 242))
        self.screen.blit(title, (x + 12, y + 10))
        rows = [
            "Goal: collect all shards, then reach core.",
            "Blue crystals are missing shard fragments.",
            "Purple rifts end the run with -8.",
            "Swirl floors drain extra reward.",
            "Slip can skew a move sideways.",
            "Arrows/WASD move Chronos manually.",
            "Q-Path learns off-route; Trail learns on-route.",
            f"Layer {self.display_mask:03b}/{self.env.required_mask:03b} | slip {self.env.slip:.2f}",
        ]
        for i, row in enumerate(rows):
            text = self.small_font.render(row, True, (201, 225, 235))
            self.screen.blit(text, (x + 12, y + 34 + i * 19))

    def draw_buttons(self) -> None:
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            hover = button.contains(mouse)
            active = (button.command == "q" and self.agent.algorithm == "Q-Learning") or (
                button.command == "sarsa" and self.agent.algorithm == "SARSA"
            )
            fill = (36, 78, 82) if active else (29, 59, 76) if hover else (19, 37, 51)
            border = (104, 255, 177) if active else (95, 232, 255) if hover else (58, 101, 121)
            pygame.draw.rect(self.screen, fill, button.rect, border_radius=7)
            pygame.draw.rect(self.screen, border, button.rect, 2, border_radius=7)
            text = self.font.render(button.label, True, (221, 246, 255))
            self.screen.blit(text, text.get_rect(center=(button.rect.centerx - 8, button.rect.centery)))
            key = self.small_font.render(button.hotkey, True, (139, 186, 204))
            self.screen.blit(key, (button.rect.right - 18, button.rect.y + 2))

    def draw_report(self) -> None:
        x = PANEL_X + 24
        y = 454
        box = pygame.Rect(x, y, 330, 190)
        pygame.draw.rect(self.screen, (14, 26, 38), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        status = self.small_font.render(self.status[:44], True, (95, 255, 176))
        self.screen.blit(status, (x + 14, y + 14))
        rows = [
            f"Learner: {self.agent.label()}",
            f"Training runs: {self.agent.episodes}",
            f"Pilot tile: {self.env.pos(self.pilot_state)}",
            f"Pilot shards: {self.pilot_state[2]:03b}/{self.env.required_mask:03b}",
            f"Pilot return: {self.pilot_return:.2f}",
        ]
        if self.report is not None:
            rows.extend(
                [
                    f"Last steps: {self.report.steps}",
                    f"Training return: {self.report.total_reward:.2f}",
                    f"Terminal: {self.report.reached_terminal}",
                ]
            )
        for i, row in enumerate(rows):
            text = self.small_font.render(row, True, (199, 222, 232))
            self.screen.blit(text, (x + 14, y + 36 + i * 17))

    def draw_pilot(self) -> None:
        pos = self.env.pos(self.pilot_state)
        rect = self.cell_rect(pos)
        center = rect.center
        pulse = 2 + int(2 * (1.0 + pygame.math.Vector2(1, 0).rotate(self.time * 120).x))
        complete = self.pilot_state[2] == self.env.required_mask
        ring = (95, 255, 176) if complete else (91, 231, 255)
        pygame.draw.circle(self.screen, ring, center, 26 + pulse, 2)
        image = self.atlas.get("agent")
        self.screen.blit(image, image.get_rect(center=(center[0], center[1] - 4)))

    def draw_end_overlay(self) -> None:
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 8, 14, 178))
        self.screen.blit(veil, (0, 0))
        won = self.env.pos(self.pilot_state) == self.env.goal and self.pilot_state[2] == self.env.required_mask
        border = (95, 255, 176) if won else (255, 105, 143)
        box = pygame.Rect(0, 0, 530, 184)
        box.center = (WIDTH // 2, HEIGHT // 2)
        pygame.draw.rect(self.screen, (12, 24, 36), box, border_radius=10)
        pygame.draw.rect(self.screen, border, box, 3, border_radius=10)
        title = pygame.font.SysFont("segoeui", 40, bold=True).render(self.end_title, True, border)
        self.screen.blit(title, title.get_rect(center=(box.centerx, box.y + 48)))
        details = [
            self.end_message,
            f"Final tile: {self.env.pos(self.pilot_state)}",
            f"Shards: {self.pilot_state[2]:03b}/{self.env.required_mask:03b}",
            f"Steps: {self.pilot_steps}   Return: {self.pilot_return:.2f}",
            "Use Reset for a new run.",
        ]
        for i, row in enumerate(details):
            text = self.font.render(row, True, (218, 238, 246))
            self.screen.blit(text, text.get_rect(center=(box.centerx, box.y + 84 + i * 20)))

    def cell_rect(self, pos: Vec) -> pygame.Rect:
        x, y = pos
        return pygame.Rect(GRID_X + x * TILE, GRID_Y + y * TILE, TILE, TILE)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Chronos Shard Trial")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game = ChronosRLGame(screen)
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            running = game.handle_event(event)
        game.update(dt)
        game.draw()
    pygame.quit()


if __name__ == "__main__":
    main()
