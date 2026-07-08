from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


Vec = Tuple[int, int]
BState = Tuple[int, int, bool]

ROOT = Path(__file__).resolve().parent
ATLAS_PATH = ROOT / "Source.png"

TILE = 58
GRID_W = 9
GRID_H = 8
GRID_X = 34
GRID_Y = 96
PANEL_X = GRID_X + GRID_W * TILE + 34
WIDTH = PANEL_X + 398
HEIGHT = 690
FPS = 60


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    command: str
    hotkey: str

    def contains(self, pos: Vec) -> bool:
        return self.rect.collidepoint(pos)


@dataclass(frozen=True)
class Observation:
    beacon: Vec
    kind: str
    reading: int


@dataclass
class InferenceReport:
    observations: int
    map_state: Vec
    map_prob: float
    entropy: float
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
        self.assets["ghost"] = self.crop_fit((21, 630, 137, 191), (48, 58), transparent=True, smooth=False)
        self.assets["beacon"] = self.crop_fit((1040, 1594, 160, 160), (48, 48), transparent=True)
        self.assets["sensor"] = self.crop_fit((427, 1047, 162, 132), (54, 44), transparent=True)
        self.assets["rift"] = self.crop_fit((1494, 848, 131, 154), (54, 58), transparent=True)

    def get(self, name: str) -> pygame.Surface:
        return self.assets[name]


def manhattan(a: Vec, b: Vec) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class BayesSensorNet:
    def __init__(self) -> None:
        self.walls = {(1, 1), (4, 1), (6, 1), (3, 2), (1, 4), (6, 4), (3, 6), (7, 6)}
        self.beacons: List[Vec] = [(0, 0), (8, 0), (0, 7), (8, 7)]
        self.rift_zones = {(2, 2), (5, 5), (7, 3)}
        positions = [(x, y) for y in range(GRID_H) for x in range(GRID_W) if (x, y) not in self.walls]
        self.states: List[BState] = [(x, y, phased) for x, y in positions for phased in (False, True)]
        self.hidden = random.choice([s for s in self.states if self.pos(s) not in self.beacons])
        self.prior: Dict[BState, float] = {state: self.prior_weight(state) for state in self.states}
        total = sum(self.prior.values())
        self.prior = {state: probability / total for state, probability in self.prior.items()}
        self.posterior: Dict[BState, float] = dict(self.prior)
        self.observations: List[Observation] = []
        self.reveal = False

    def pos(self, state: BState) -> Vec:
        return state[0], state[1]

    def prior_weight(self, state: BState) -> float:
        pos = self.pos(state)
        center_bias = 1.25 if 2 <= pos[0] <= 6 and 2 <= pos[1] <= 5 else 1.0
        phase_bias = 0.58 if state[2] else 0.42
        rift_bias = 1.35 if pos in self.rift_zones else 1.0
        return center_bias * phase_bias * rift_bias

    def reset(self, new_target: bool = True) -> None:
        if new_target:
            self.hidden = random.choice([s for s in self.states if self.pos(s) not in self.beacons])
        self.posterior = dict(self.prior)
        self.observations = []
        self.reveal = False

    def passable(self, pos: Vec) -> bool:
        return 0 <= pos[0] < GRID_W and 0 <= pos[1] < GRID_H and pos not in self.walls

    def likelihood(self, obs: Observation, state: BState) -> float:
        if obs.kind == "flux":
            # Check proximity to any active rift zone within a Manhattan distance of 2.
            near_rift = any(manhattan(self.pos(state), zone) <= 2 for zone in self.rift_zones)
            high_prob = 0.78 if state[2] or near_rift else 0.18
            return high_prob if obs.reading == 1 else 1.0 - high_prob

        # Calculate expected distance, incorporating phase distortion offset if applicable.
        true_dist = manhattan(obs.beacon, self.pos(state)) + (1 if state[2] else 0)
        error = abs(obs.reading - true_dist)

        # Map the absolute sensor measurement error to its corresponding conditional probability.
        if error == 0:
            return 0.62
        elif error == 1:
            return 0.22
        elif error == 2:
            return 0.10
        elif error == 3:
            return 0.04
        return 0.02

    def sample_reading(self, beacon: Vec, kind: str) -> int:
        if kind == "flux":
            near_rift = min(manhattan(self.pos(self.hidden), zone) for zone in self.rift_zones) <= 2
            high_probability = 0.78 if self.hidden[2] or near_rift else 0.18
            return 1 if random.random() < high_probability else 0
        true_distance = manhattan(beacon, self.pos(self.hidden)) + (1 if self.hidden[2] else 0)
        choices = [true_distance - 2, true_distance - 1, true_distance, true_distance + 1, true_distance + 2, true_distance + 3]
        weights = [0.06, 0.16, 0.56, 0.14, 0.06, 0.02]
        reading = random.choices(choices, weights=weights, k=1)[0]
        return max(0, reading)

    def observe(self, beacon: Optional[Vec] = None, kind: str = "distance") -> InferenceReport:
        if beacon is None:
            beacon = self.beacons[len(self.observations) % len(self.beacons)]
        reading = self.sample_reading(beacon, kind)
        self.observations.append(Observation(beacon, kind, reading))
        self.posterior = self.infer(self.observations)
        if kind == "flux":
            label = "high" if reading else "low"
            return self.report(f"Flux near beacon {self.beacons.index(beacon) + 1}: {label}.")
        return self.report(f"Ping from beacon {self.beacons.index(beacon) + 1}: range {reading}.")

    def infer(self, observations: List[Observation]) -> Dict[BState, float]:
        unnormalized = {}
        total = 0.0

        # Compute the joint probability of the prior and the accumulated evidence for each state.
        for state in self.states:
            prob = self.prior[state]
            for obs in observations:
                prob *= self.likelihood(obs, state)
            unnormalized[state] = prob
            total += prob

        # Normalize the posterior distribution, defaulting to the prior if total likelihood is zero.
        if total == 0.0:
            return dict(self.prior)

        return {state: prob / total for state, prob in unnormalized.items()}

    def position_posterior(self) -> Dict[Vec, float]:
        marginal: Dict[Vec, float] = {}
        for state, probability in self.posterior.items():
            marginal[self.pos(state)] = marginal.get(self.pos(state), 0.0) + probability
        return marginal

    def map_state(self) -> Tuple[Vec, float]:
        marginal = self.position_posterior()
        state, probability = max(marginal.items(), key=lambda item: item[1])
        return state, probability

    def map_phase(self) -> Tuple[bool, float]:
        phase_true = sum(probability for state, probability in self.posterior.items() if state[2])
        if phase_true >= 0.5:
            return True, phase_true
        return False, 1.0 - phase_true

    def entropy(self) -> float:
        return -sum(p * math.log(p, 2) for p in self.posterior.values() if p > 0.0)

    def report(self, message: str) -> InferenceReport:
        state, probability = self.map_state()
        return InferenceReport(len(self.observations), state, probability, self.entropy(), message)


class ChronosBayesGame:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.atlas = TextureAtlas(ATLAS_PATH)
        self.net = BayesSensorNet()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.big_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.value_font = pygame.font.SysFont("consolas", 14, bold=True)
        self.buttons = self.make_buttons()
        self.report: InferenceReport = self.net.report("Signal map initialized.")
        self.status = self.report.message
        self.game_over = False
        self.end_title = ""
        self.end_message = ""
        self.guess: Optional[Vec] = None
        self.time = 0.0

    def make_buttons(self) -> List[Button]:
        specs = [
            ("Ping", "sense", "1"),
            ("Flux", "flux", "2"),
            ("Sweep", "sense4", "3"),
            ("Peek", "reveal", "4"),
            ("Reset", "reset", "R"),
            ("New", "new", "N"),
        ]
        buttons: List[Button] = []
        x = PANEL_X + 24
        y = 340
        for i, (label, command, hotkey) in enumerate(specs):
            rect = pygame.Rect(x + (i % 3) * 104, y + (i // 3) * 44, 92, 34)
            buttons.append(Button(rect, label, command, hotkey))
        return buttons

    def run_command(self, command: str) -> None:
        if self.game_over and command not in {"reset", "new"}:
            self.status = "Investigation is closed. Reset or New starts again."
            return
        try:
            if command == "sense":
                self.report = self.net.observe(kind="distance")
            elif command == "flux":
                self.report = self.net.observe(kind="flux")
            elif command == "sense4":
                for _ in range(4):
                    kind = "flux" if len(self.net.observations) % 2 else "distance"
                    self.report = self.net.observe(kind=kind)
            elif command == "reveal":
                self.net.reveal = not self.net.reveal
                self.report = self.net.report("Hidden anomaly revealed." if self.net.reveal else "Hidden anomaly concealed.")
            elif command == "reset":
                self.net.reset(new_target=False)
                self.game_over = False
                self.end_title = ""
                self.end_message = ""
                self.guess = None
                self.report = self.net.report("Evidence cleared. Signal map restored.")
            elif command == "new":
                self.net.reset(new_target=True)
                self.game_over = False
                self.end_title = ""
                self.end_message = ""
                self.guess = None
                self.report = self.net.report("New hidden anomaly seeded.")
        except NotImplementedError as exc:
            self.status = str(exc)
            return
        self.status = self.report.message

    def make_guess(self, pos: Vec) -> None:
        if self.game_over:
            self.status = "Investigation is closed. Reset or New starts again."
            return
        self.guess = pos
        hidden_pos = self.net.pos(self.net.hidden)
        self.net.reveal = True
        self.game_over = True
        if pos == hidden_pos:
            self.end_title = "Anomaly Found"
            self.end_message = "Chronos locked the hidden signal."
        else:
            self.end_title = "Signal Lost"
            self.end_message = f"The anomaly was at {hidden_pos}, not {pos}."
        self.report = self.net.report(self.end_title)
        self.status = f"{self.end_title}: guessed {pos}."

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.contains(event.pos):
                    self.run_command(button.command)
                    return True
            clicked = self.grid_pos(event.pos)
            if clicked in self.net.beacons:
                if not self.game_over:
                    try:
                        self.report = self.net.observe(clicked)
                        self.status = self.report.message
                    except NotImplementedError as exc:
                        self.status = str(exc)
            elif clicked is not None:
                self.make_guess(clicked)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if self.game_over and event.key not in (pygame.K_r, pygame.K_n):
                self.status = "Investigation is closed. Reset or New starts again."
                return True
            if event.key == pygame.K_1:
                self.run_command("sense")
            elif event.key == pygame.K_2:
                self.run_command("flux")
            elif event.key == pygame.K_3:
                self.run_command("sense4")
            elif event.key == pygame.K_4:
                self.run_command("reveal")
            elif event.key == pygame.K_r:
                self.run_command("reset")
            elif event.key == pygame.K_n:
                self.run_command("new")
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
        title = self.big_font.render("Chronos Signal Hunt", True, (231, 246, 255))
        self.screen.blit(title, (28, 18))
        subtitle = self.font.render("Scan beacons, click a suspect.", True, (133, 186, 205))
        self.screen.blit(subtitle, (374, 25))

    def draw_board(self) -> None:
        board = pygame.Rect(GRID_X - 8, GRID_Y - 8, GRID_W * TILE + 16, GRID_H * TILE + 16)
        pygame.draw.rect(self.screen, (5, 10, 17), board, border_radius=8)
        pygame.draw.rect(self.screen, (50, 91, 112), board, 2, border_radius=8)
        marginal = self.net.position_posterior()
        max_probability = max(marginal.values())
        map_state, _ = self.net.map_state()
        for y in range(GRID_H):
            for x in range(GRID_W):
                state = (x, y)
                rect = self.cell_rect(state)
                if state in self.net.walls:
                    self.screen.blit(self.atlas.tile_wall[(x + y) % len(self.atlas.tile_wall)], rect)
                    pygame.draw.rect(self.screen, (7, 12, 18), rect, 1)
                    continue
                self.screen.blit(self.atlas.tile_floor[(x * 2 + y * 3) % len(self.atlas.tile_floor)], rect)
                probability = marginal[state]
                alpha = int(32 + 190 * probability / max_probability) if max_probability > 0 else 32
                shade = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                shade.fill((81, 224, 255, alpha))
                self.screen.blit(shade, rect)
                if state == map_state:
                    pygame.draw.rect(self.screen, (255, 226, 90), rect.inflate(-5, -5), 3)
                if self.guess == state:
                    pygame.draw.rect(self.screen, (255, 105, 143), rect.inflate(-9, -9), 3)
                if state in self.net.rift_zones:
                    self.screen.blit(self.atlas.tile_distortion[(x + y) % len(self.atlas.tile_distortion)], rect)
                if state in self.net.beacons:
                    image = self.atlas.get("beacon")
                    self.screen.blit(image, image.get_rect(center=rect.center))
                    index = self.net.beacons.index(state) + 1
                    label = self.font.render(str(index), True, (255, 247, 180))
                    self.screen.blit(label, label.get_rect(center=(rect.centerx, rect.centery + 1)))
                probability_text = self.value_font.render(f"{probability:.2f}", True, (235, 248, 252))
                self.screen.blit(probability_text, probability_text.get_rect(center=(rect.centerx, rect.bottom - 11)))
                pygame.draw.rect(self.screen, (5, 13, 20), rect, 1)
        if self.net.reveal:
            rect = self.cell_rect(self.net.pos(self.net.hidden))
            image = self.atlas.get("ghost")
            self.screen.blit(image, image.get_rect(center=rect.center))
            phase = "P" if self.net.hidden[2] else "S"
            text = self.font.render(phase, True, (255, 232, 128))
            self.screen.blit(text, text.get_rect(center=(rect.centerx, rect.y + 12)))

    def draw_panel(self) -> None:
        panel = pygame.Rect(PANEL_X, 0, WIDTH - PANEL_X, HEIGHT)
        pygame.draw.rect(self.screen, (10, 16, 25), panel)
        pygame.draw.line(self.screen, (54, 91, 111), (PANEL_X, 0), (PANEL_X, HEIGHT), 2)
        icon = self.atlas.get("sensor")
        self.screen.blit(icon, (PANEL_X + 28, 30))
        title = self.big_font.render("Signal Console", True, (231, 246, 255))
        self.screen.blit(title, (PANEL_X + 94, 34))
        self.draw_legend(PANEL_X + 24, 108)
        self.draw_buttons()
        self.draw_report()

    def draw_legend(self, x: int, y: int) -> None:
        box = pygame.Rect(x, y, 340, 206)
        pygame.draw.rect(self.screen, (12, 24, 35), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        title = self.small_font.render("Signal hunt brief", True, (118, 220, 242))
        self.screen.blit(title, (x + 12, y + 10))
        rows = [
            "Goal: find the hidden anomaly cell.",
            "Ping gives noisy distance from a beacon.",
            "Flux says whether phase energy is high.",
            "Bright cells are more likely locations.",
            "Yellow outline is the current best guess.",
            "Click beacons to scan from them.",
            "Click any other cell to make your guess.",
            "Peek reveals the answer for debugging.",
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
        y = 436
        box = pygame.Rect(x, y, 340, 218)
        pygame.draw.rect(self.screen, (14, 26, 38), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        status = self.small_font.render(self.status[:46], True, (95, 255, 176))
        self.screen.blit(status, (x + 14, y + 14))
        rows = [
            f"Scans: {self.report.observations}",
            f"Best cell: {self.report.map_state} ({self.report.map_prob:.3f})",
            f"Uncertainty: {self.report.entropy:.3f} bits",
            f"Phase read: {self.net.map_phase()[0]} ({self.net.map_phase()[1]:.2f})",
            f"Guess: {self.guess if self.guess else 'none'}",
            f"Hidden: {self.net.hidden if self.net.reveal else 'concealed'}",
        ]
        for i, row in enumerate(rows):
            text = self.small_font.render(row, True, (199, 222, 232))
            self.screen.blit(text, (x + 14, y + 44 + i * 20))
        if self.net.observations:
            title = self.small_font.render("Latest readings", True, (118, 220, 242))
            self.screen.blit(title, (x + 14, y + 166))
            latest = self.net.observations[-3:]
            for i, obs in enumerate(latest):
                idx = self.net.beacons.index(obs.beacon) + 1
                reading = "high" if obs.kind == "flux" and obs.reading else "low" if obs.kind == "flux" else str(obs.reading)
                text = self.small_font.render(f"B{idx} {obs.kind[:1]}:{reading}", True, (174, 201, 214))
                self.screen.blit(text, (x + 14 + i * 102, y + 184))

    def draw_end_overlay(self) -> None:
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 8, 14, 178))
        self.screen.blit(veil, (0, 0))
        won = self.guess == self.net.pos(self.net.hidden)
        border = (95, 255, 176) if won else (255, 105, 143)
        box = pygame.Rect(0, 0, 540, 184)
        box.center = (WIDTH // 2, HEIGHT // 2)
        pygame.draw.rect(self.screen, (12, 24, 36), box, border_radius=10)
        pygame.draw.rect(self.screen, border, box, 3, border_radius=10)
        title = pygame.font.SysFont("segoeui", 40, bold=True).render(self.end_title, True, border)
        self.screen.blit(title, title.get_rect(center=(box.centerx, box.y + 48)))
        hidden = self.net.pos(self.net.hidden)
        details = [
            self.end_message,
            f"Guess: {self.guess}   Hidden: {hidden}",
            f"Scans used: {len(self.net.observations)}",
            f"Final confidence: {self.report.map_prob:.3f}",
            "Use Reset or New for another hunt.",
        ]
        for i, row in enumerate(details):
            text = self.font.render(row, True, (218, 238, 246))
            self.screen.blit(text, text.get_rect(center=(box.centerx, box.y + 84 + i * 20)))

    def cell_rect(self, pos: Vec) -> pygame.Rect:
        x, y = pos
        return pygame.Rect(GRID_X + x * TILE, GRID_Y + y * TILE, TILE, TILE)

    def grid_pos(self, pixel: Vec) -> Optional[Vec]:
        x = (pixel[0] - GRID_X) // TILE
        y = (pixel[1] - GRID_Y) // TILE
        pos = (x, y)
        return pos if self.net.passable(pos) else None


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Chronos Signal Hunt")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game = ChronosBayesGame(screen)
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
