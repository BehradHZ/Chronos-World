from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


Vec = Tuple[int, int]
State = Tuple[int, int, bool]

ROOT = Path(__file__).resolve().parent
ATLAS_PATH = ROOT / "Source.png"

TILE = 62
GRID_W = 8
GRID_H = 7
GRID_X = 36
GRID_Y = 104
PANEL_X = GRID_X + GRID_W * TILE + 34
WIDTH = PANEL_X + 390
HEIGHT = 670
FPS = 60

GAMMA = 0.92
STEP_REWARD = -0.04
NOISE = 0.18

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
class MdpReport:
    algorithm: str
    iterations: int
    delta: float
    stable: bool
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
        self.assets["core"] = self.crop_fit((1040, 1594, 160, 160), (68, 68), transparent=True)
        self.assets["bad_core"] = self.crop_fit((1494, 848, 131, 154), (54, 58), transparent=True)
        self.assets["agent"] = self.crop_fit((42, 25, 106, 158), (42, 54), transparent=True, smooth=False)

    def get(self, name: str) -> pygame.Surface:
        return self.assets[name]


class GridMDP:
    def __init__(self) -> None:
        self.walls = {(1, 1), (3, 1), (3, 2), (5, 3), (1, 5), (4, 5)}
        self.terminals: Dict[Vec, float] = {(7, 0): 12.0, (7, 6): -12.0, (0, 6): 4.0}
        self.hazards: Dict[Vec, float] = {(2, 4): -2.0, (6, 4): -3.0, (5, 1): -1.5, (4, 3): -2.5}
        self.chargers = {(2, 6), (4, 0)}
        self.gates = {(6, 2), (6, 5)}
        self.start: State = (0, 0, False)
        self.values: Dict[State, float] = {s: 0.0 for s in self.states()}
        self.policy: Dict[State, str] = {s: "E" for s in self.states() if not self.terminal(s)}
        self.iterations = 0
        self.last_delta = 0.0
        self.noise = NOISE
        self.display_charged = False

    def states(self) -> List[State]:
        states: List[State] = []
        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x, y) not in self.walls:
                    states.append((x, y, False))
                    states.append((x, y, True))
        return states

    def pos(self, state: State) -> Vec:
        return state[0], state[1]

    def in_bounds(self, pos: Vec) -> bool:
        return 0 <= pos[0] < GRID_W and 0 <= pos[1] < GRID_H

    def passable(self, pos: Vec) -> bool:
        return self.in_bounds(pos) and pos not in self.walls

    def terminal(self, state: State) -> bool:
        return self.pos(state) in self.terminals

    def reward(self, state: State, next_state: State) -> float:
        next_pos = self.pos(next_state)
        if next_pos in self.terminals:
            return self.terminals[next_pos]
        reward = STEP_REWARD + self.hazards.get(next_pos, 0.0)
        if next_pos in self.chargers and not state[2]:
            reward += 1.2
        if self.pos(state) != next_pos and next_pos in self.gates:
            reward += 0.4
        return reward

    def move(self, state: State, action: str) -> State:
        dx, dy = ACTIONS[action]
        next_pos = (state[0] + dx, state[1] + dy)
        if self.passable(next_pos):
            if next_pos in self.gates and not state[2]:
                return state
            charged = state[2] or next_pos in self.chargers
            return next_pos[0], next_pos[1], charged
        return state

    def side_actions(self, action: str) -> Tuple[str, str]:
        index = ACTION_ORDER.index(action)
        return ACTION_ORDER[(index - 1) % 4], ACTION_ORDER[(index + 1) % 4]

    def transitions(self, state: State, action: str) -> List[Tuple[State, float, float]]:
            # Terminal states transition to themselves indefinitely with zero future reward.
            if self.terminal(state):
                return [(state, 1.0, 0.0)]

            # Collect the intended movement along with left/right stochastic drift probabilities.
            left_slip, right_slip = self.side_actions(action)
            candidates = [
                (action, 1.0 - self.noise),
                (left_slip, self.noise / 2.0),
                (right_slip, self.noise / 2.0)
            ]

            # Aggregate probabilities for outcomes that resolve to the same destination state.
            merged_transitions: Dict[State, float] = {}
            for act, prob in candidates:
                if prob > 0.0:
                    next_state = self.move(state, act)
                    merged_transitions[next_state] = merged_transitions.get(next_state, 0.0) + prob

            # Construct transition tuples mapping destination, probability, and environmental step reward.
            return [(nxt, prob, self.reward(state, nxt)) for nxt, prob in merged_transitions.items()]

    def sample_transition(self, state: State, action: str) -> Tuple[State, float]:
        if self.terminal(state):
            return state, 0.0
        roll = random.random()
        actual_action = action
        if roll >= 1.0 - self.noise:
            left, right = self.side_actions(action)
            actual_action = left if random.random() < 0.5 else right
        next_state = self.move(state, actual_action)
        return next_state, self.reward(state, next_state)

    def q_value(self, state: State, action: str, values: Optional[Dict[State, float]] = None) -> float:
        if values is None:
            values = self.values

        # Apply the Bellman expectation backup equation across all possible stochastic outcomes.
        q = 0.0
        for next_state, prob, reward in self.transitions(state, action):
            q += prob * (reward + GAMMA * values[next_state])
        return q

    def best_action(self, state: State, values: Optional[Dict[State, float]] = None) -> Tuple[str, float]:
        scored = [(self.q_value(state, action, values), action) for action in ACTION_ORDER]
        value, action = max(scored, key=lambda item: (item[0], item[1]))
        return action, value

    def value_iteration_step(self) -> MdpReport:
        new_values = {}
        delta = 0.0

        # Synchronously calculate maximum expected utility for each state across the grid.
        for state in self.states():
            if self.terminal(state):
                new_values[state] = 0.0
            else:
                _, best_val = self.best_action(state, self.values)
                new_values[state] = best_val
                delta = max(delta, abs(new_values[state] - self.values[state]))

        self.values = new_values
        self.iterations += 1
        self.last_delta = delta
        self.update_policy()

        # Check for convergence using standard Bellman residual threshold criteria.
        stable = delta < 1e-5
        message = "Route values converged." if stable else "Value iteration step completed."
        return MdpReport("Value Sweep", self.iterations, delta, stable, message)

    def run_value_iteration(self, limit: int = 100) -> MdpReport:
        report = MdpReport("Value Sweep", self.iterations, 0.0, False, "")
        for _ in range(limit):
            report = self.value_iteration_step()
            if report.stable:
                break
        report.message = "Route values converged." if report.stable else f"Ran {limit} route-value sweeps."
        return report

    def update_policy(self) -> None:
        for state in self.states():
            if self.terminal(state):
                continue
            action, _ = self.best_action(state)
            self.policy[state] = action

    def evaluate_policy(self, sweeps: int = 32) -> None:
        for _ in range(sweeps):
            new_values = dict(self.values)
            for state in self.states():
                if self.terminal(state):
                    continue
                action = self.policy.get(state, "E")
                new_values[state] = self.q_value(state, action, self.values)
            self.values = new_values

    def policy_iteration(self, limit: int = 20) -> MdpReport:
        stable = False
        
        # Alternate between policy evaluation sweeps and greedy policy improvement step routines.
        for _ in range(limit):
            self.evaluate_policy()
            policy_stable = True

            for state in self.states():
                if self.terminal(state):
                    continue
                old_action = self.policy.get(state)
                best_act, _ = self.best_action(state, self.values)
                
                if best_act != old_action:
                    policy_stable = False
                self.policy[state] = best_act

            self.iterations += 1
            if policy_stable:
                stable = True
                break

        message = "Route policy stabilized." if stable else f"Ran {limit} policy iteration updates."
        return MdpReport("Policy Iteration", self.iterations, self.last_delta, stable, message)

    def reset(self) -> None:
        self.values = {s: 0.0 for s in self.states()}
        self.policy = {s: "E" for s in self.states() if not self.terminal(s)}
        self.iterations = 0
        self.last_delta = 0.0

class ChronosMdpGame:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.atlas = TextureAtlas(ATLAS_PATH)
        self.mdp = GridMDP()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.big_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.value_font = pygame.font.SysFont("consolas", 16, bold=True)
        self.buttons = self.make_buttons()
        self.report: Optional[MdpReport] = None
        self.pilot_state = self.mdp.start
        self.pilot_return = 0.0
        self.pilot_steps = 0
        self.game_over = False
        self.end_title = ""
        self.end_message = ""
        self.status = "Plan a route, or move Chronos with arrows/WASD."
        self.time = 0.0

    def make_buttons(self) -> List[Button]:
        specs = [
            ("Solve", "value", "1"),
            ("Step", "step", "2"),
            ("Route", "policy", "3"),
            ("Layer", "layer", "L"),
            ("Drift", "noise", "N"),
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
        self.pilot_state = self.mdp.start
        self.pilot_return = 0.0
        self.pilot_steps = 0
        self.game_over = False
        self.end_title = ""
        self.end_message = ""

    def run_command(self, command: str) -> None:
        try:
            if command == "value":
                self.report = self.mdp.run_value_iteration()
            elif command == "step":
                self.report = self.mdp.value_iteration_step()
            elif command == "policy":
                self.report = self.mdp.policy_iteration()
            elif command == "layer":
                self.mdp.display_charged = not self.mdp.display_charged
                mode = "charged" if self.mdp.display_charged else "uncharged"
                self.report = None
                self.status = f"Showing {mode} value layer."
                return
            elif command == "noise":
                self.mdp.noise = 0.08 if self.mdp.noise > 0.12 else 0.22
                self.report = None
                self.status = f"Rift drift set to {self.mdp.noise:.2f}."
                return
            elif command == "reset":
                self.mdp.reset()
                self.reset_pilot()
                self.report = None
                self.status = "Route values and pilot position reset."
                return
        except NotImplementedError as exc:
            self.report = None
            self.status = str(exc)
            return
        if self.report is not None:
            self.status = self.report.message

    def move_pilot(self, action: str) -> None:
        if self.game_over:
            self.status = "Gatewalk is sealed. Reset starts a new run."
            return
        if self.mdp.terminal(self.pilot_state):
            self.finish_gatewalk(self.mdp.terminals[self.mdp.pos(self.pilot_state)])
            return
        old_state = self.pilot_state
        try:
            next_state, reward = self.mdp.sample_transition(old_state, action)
        except NotImplementedError as exc:
            self.status = str(exc)
            return
        self.pilot_state = next_state
        self.pilot_return += reward
        self.pilot_steps += 1
        old_pos = self.mdp.pos(old_state)
        new_pos = self.mdp.pos(next_state)
        if new_pos == old_pos:
            self.status = f"Chronos held position. Reward {reward:+.2f}."
        elif next_state[2] and not old_state[2]:
            self.status = f"Chronos charged at {new_pos}. Reward {reward:+.2f}."
        elif self.mdp.terminal(next_state):
            self.status = f"Terminal reached at {new_pos}. Reward {reward:+.2f}."
            self.finish_gatewalk(reward)
        else:
            self.status = f"Chronos moved to {new_pos}. Reward {reward:+.2f}."
        self.mdp.display_charged = next_state[2]

    def finish_gatewalk(self, reward: float) -> None:
        self.game_over = True
        pos = self.mdp.pos(self.pilot_state)
        if reward > 0:
            if reward >= 10:
                self.end_title = "Gatewalk Complete"
                self.end_message = "Chronos reached the golden core."
            else:
                self.end_title = "Safe Extraction"
                self.end_message = "Chronos found the side exit."
        else:
            self.end_title = "Timeline Broken"
            self.end_message = "Chronos fell into the unstable rift."
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
                self.status = "Gatewalk is sealed. Reset starts a new run."
                return True
            if event.key == pygame.K_1:
                self.run_command("value")
            elif event.key == pygame.K_2:
                self.run_command("step")
            elif event.key == pygame.K_3:
                self.run_command("policy")
            elif event.key == pygame.K_l:
                self.run_command("layer")
            elif event.key == pygame.K_n:
                self.run_command("noise")
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
        pygame.draw.rect(self.screen, (8, 13, 21), (0, 0, WIDTH, 78))
        title = self.big_font.render("Chronos Gatewalk", True, (231, 246, 255))
        self.screen.blit(title, (28, 20))
        subtitle = self.font.render("Plan under drift; pilot the route.", True, (133, 186, 205))
        self.screen.blit(subtitle, (320, 27))

    def draw_board(self) -> None:
        board = pygame.Rect(GRID_X - 8, GRID_Y - 8, GRID_W * TILE + 16, GRID_H * TILE + 16)
        pygame.draw.rect(self.screen, (5, 10, 17), board, border_radius=8)
        pygame.draw.rect(self.screen, (50, 91, 112), board, 2, border_radius=8)
        min_value = min(self.mdp.values.values()) if self.mdp.values else 0.0
        max_value = max(self.mdp.values.values()) if self.mdp.values else 1.0
        spread = max(1.0, max_value - min_value)
        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = (x, y)
                state = (x, y, self.mdp.display_charged)
                rect = self.cell_rect(pos)
                if pos in self.mdp.walls:
                    self.screen.blit(self.atlas.tile_wall[(x + y) % len(self.atlas.tile_wall)], rect)
                    pygame.draw.rect(self.screen, (7, 12, 18), rect, 1)
                    continue
                tile = self.atlas.tile_floor[(x * 2 + y * 3) % len(self.atlas.tile_floor)]
                self.screen.blit(tile, rect)
                value = self.mdp.values.get(state, 0.0)
                normalized = (value - min_value) / spread
                if value >= 0:
                    overlay = (40, 180, 160, int(28 + 88 * normalized))
                else:
                    overlay = (210, 57, 120, int(36 + 92 * (1.0 - normalized)))
                shade = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                shade.fill(overlay)
                self.screen.blit(shade, rect)
                if pos in self.mdp.hazards:
                    self.screen.blit(self.atlas.tile_distortion[(x + y) % len(self.atlas.tile_distortion)], rect)
                if pos in self.mdp.chargers:
                    pygame.draw.circle(self.screen, (95, 255, 176), rect.center, 23, 3)
                    charge = self.small_font.render("C", True, (218, 255, 232))
                    self.screen.blit(charge, charge.get_rect(center=(rect.centerx, rect.centery - 3)))
                if pos in self.mdp.gates:
                    gate_color = (109, 231, 255) if self.mdp.display_charged else (255, 119, 145)
                    pygame.draw.rect(self.screen, gate_color, rect.inflate(-15, -15), 3, border_radius=5)
                if pos in self.mdp.terminals:
                    self.draw_terminal(pos)
                else:
                    self.draw_policy_arrow(state, rect)
                if pos not in self.mdp.terminals:
                    value_text = self.value_font.render(f"{value: .2f}", True, (235, 247, 252))
                    self.screen.blit(value_text, value_text.get_rect(center=(rect.centerx, rect.centery + 18)))
                pygame.draw.rect(self.screen, (5, 13, 20), rect, 1)
        self.draw_pilot()

    def draw_terminal(self, state: Vec) -> None:
        rect = self.cell_rect(state)
        reward = self.mdp.terminals[state]
        asset = "core" if reward > 0 else "bad_core"
        image = self.atlas.get(asset)
        self.screen.blit(image, image.get_rect(center=rect.center))
        label = self.font.render(f"{reward:+.0f}", True, (255, 232, 128) if reward > 0 else (255, 132, 150))
        self.screen.blit(label, label.get_rect(center=(rect.centerx, rect.y + 12)))

    def draw_policy_arrow(self, state: State, rect: pygame.Rect) -> None:
        action = self.mdp.policy.get(state)
        if action is None:
            return
        dx, dy = ACTIONS[action]
        start = pygame.Vector2(rect.center)
        end = start + pygame.Vector2(dx, dy) * 18
        pygame.draw.line(self.screen, (78, 238, 255), start, end, 4)
        left = pygame.Vector2(-dy, dx) * 6
        back = pygame.Vector2(dx, dy) * -7
        pygame.draw.polygon(self.screen, (216, 253, 255), [end, end + back + left, end + back - left])

    def draw_panel(self) -> None:
        panel = pygame.Rect(PANEL_X, 0, WIDTH - PANEL_X, HEIGHT)
        pygame.draw.rect(self.screen, (10, 16, 25), panel)
        pygame.draw.line(self.screen, (54, 91, 111), (PANEL_X, 0), (PANEL_X, HEIGHT), 2)
        icon = self.atlas.get("agent")
        self.screen.blit(icon, (PANEL_X + 24, 24))
        title = self.big_font.render("Gate Console", True, (231, 246, 255))
        self.screen.blit(title, (PANEL_X + 94, 34))
        self.draw_legend(PANEL_X + 24, 108)
        self.draw_buttons()
        self.draw_report()

    def draw_legend(self, x: int, y: int) -> None:
        box = pygame.Rect(x, y, 330, 206)
        pygame.draw.rect(self.screen, (12, 24, 35), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        title = self.small_font.render("Gatewalk brief", True, (118, 220, 242))
        self.screen.blit(title, (x + 12, y + 10))
        rows = [
            "Goal: reach the golden core (+12).",
            "Green pads charge Chronos.",
            "Red gates block uncharged Chronos.",
            "Purple rift is a terminal loss (-12).",
            "Swirl floors are risky negative zones.",
            "Drift may slide moves sideways.",
            "Arrows/WASD move the pilot manually.",
            f"View layer: {'charged' if self.mdp.display_charged else 'uncharged'} | drift {self.mdp.noise:.2f}",
        ]
        for i, row in enumerate(rows):
            text = self.small_font.render(row, True, (201, 225, 235))
            self.screen.blit(text, (x + 12, y + 34 + i * 19))

    def draw_buttons(self) -> None:
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            hover = button.contains(mouse)
            fill = (29, 59, 76) if hover else (19, 37, 51)
            border = (95, 232, 255) if hover else (58, 101, 121)
            pygame.draw.rect(self.screen, fill, button.rect, border_radius=7)
            pygame.draw.rect(self.screen, border, button.rect, 2, border_radius=7)
            text = self.font.render(button.label, True, (221, 246, 255))
            self.screen.blit(text, text.get_rect(center=button.rect.center))
            key = self.small_font.render(button.hotkey, True, (139, 186, 204))
            self.screen.blit(key, (button.rect.right - 18, button.rect.y + 2))

    def draw_report(self) -> None:
        x = PANEL_X + 24
        y = 454
        box = pygame.Rect(x, y, 330, 180)
        pygame.draw.rect(self.screen, (14, 26, 38), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        status = self.small_font.render(self.status[:44], True, (95, 255, 176))
        self.screen.blit(status, (x + 14, y + 14))
        rows = [
            f"Planning sweeps: {self.mdp.iterations}",
            f"Last value shift: {self.mdp.last_delta:.5f}",
            f"Pilot tile: {self.mdp.pos(self.pilot_state)}",
            f"Pilot charge: {self.pilot_state[2]}",
            f"Pilot return: {self.pilot_return:.2f}",
        ]
        if self.report is not None:
            rows.extend(
                [
                    f"Planner: {self.report.algorithm}",
                    f"Stable: {self.report.stable}",
                ]
            )
        for i, row in enumerate(rows):
            text = self.small_font.render(row, True, (199, 222, 232))
            self.screen.blit(text, (x + 14, y + 44 + i * 20))

    def draw_pilot(self) -> None:
        pos = self.mdp.pos(self.pilot_state)
        rect = self.cell_rect(pos)
        center = rect.center
        pulse = 2 + int(2 * (1.0 + pygame.math.Vector2(1, 0).rotate(self.time * 120).x))
        ring = (95, 255, 176) if self.pilot_state[2] else (91, 231, 255)
        pygame.draw.circle(self.screen, ring, center, 26 + pulse, 2)
        image = self.atlas.get("agent")
        self.screen.blit(image, image.get_rect(center=(center[0], center[1] - 4)))

    def draw_end_overlay(self) -> None:
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 8, 14, 178))
        self.screen.blit(veil, (0, 0))
        reward = self.mdp.terminals.get(self.mdp.pos(self.pilot_state), 0.0)
        good = reward > 0
        border = (95, 255, 176) if good else (255, 105, 143)
        box = pygame.Rect(0, 0, 500, 178)
        box.center = (WIDTH // 2, HEIGHT // 2)
        pygame.draw.rect(self.screen, (12, 24, 36), box, border_radius=10)
        pygame.draw.rect(self.screen, border, box, 3, border_radius=10)
        title = pygame.font.SysFont("segoeui", 42, bold=True).render(self.end_title, True, border)
        self.screen.blit(title, title.get_rect(center=(box.centerx, box.y + 48)))
        details = [
            self.end_message,
            f"Final tile: {self.mdp.pos(self.pilot_state)}",
            f"Steps: {self.pilot_steps}   Return: {self.pilot_return:.2f}",
            "Use Reset for a new run.",
        ]
        for i, row in enumerate(details):
            text = self.font.render(row, True, (218, 238, 246))
            self.screen.blit(text, text.get_rect(center=(box.centerx, box.y + 88 + i * 22)))

    def cell_rect(self, pos: Vec) -> pygame.Rect:
        x, y = pos
        return pygame.Rect(GRID_X + x * TILE, GRID_Y + y * TILE, TILE, TILE)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Chronos Gatewalk")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game = ChronosMdpGame(screen)
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
