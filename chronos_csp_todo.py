from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pygame


Vec = Tuple[int, int]
Color = str

ROOT = Path(__file__).resolve().parent
ATLAS_PATH = ROOT / "Source.png"

TILE = 58
GRID_W = 10
GRID_H = 8
GRID_X = 34
GRID_Y = 96
PANEL_X = GRID_X + GRID_W * TILE + 34
WIDTH = PANEL_X + 390
HEIGHT = 670
FPS = 60

FREQUENCIES: Tuple[Color, ...] = ("Cyan", "Green", "Violet", "Gold")
FREQ_COLORS: Dict[Color, Tuple[int, int, int]] = {
    "Cyan": (76, 231, 255),
    "Green": (85, 255, 167),
    "Violet": (203, 103, 255),
    "Gold": (255, 220, 92),
}
FREQ_COSTS: Dict[Color, int] = {"Cyan": 3, "Green": 2, "Violet": 4, "Gold": 5}
FREQ_QUOTAS: Dict[Color, int] = {"Cyan": 3, "Green": 2, "Violet": 2, "Gold": 3}
RELAY_NAMES: Dict[str, str] = {
    "A": "Astra",
    "B": "Boreal",
    "C": "Cyra",
    "D": "Dusk",
    "E": "Echo",
    "F": "Fable",
    "G": "Gale",
    "H": "Halo",
    "I": "Ion",
    "J": "Juno",
}
RULE_LINES = [
    "Light all ten relays with signal runes.",
    "Linked relays cannot share the same rune.",
    "Every rune must meet its exact ledger quota.",
    "Cyra-Halo rejects Violet/Gold together.",
    "Dusk-Juno rejects Cyan/Gold together.",
]


@dataclass(frozen=True)
class Variable:
    name: str
    pos: Vec
    forbidden: Tuple[Color, ...] = ()


@dataclass
class CspReport:
    algorithm: str
    solved: bool
    assignment: Dict[str, Color]
    nodes: int
    backtracks: int
    checks: int
    cost: int
    elapsed_ms: float
    message: str
    log: List[str] = field(default_factory=list)


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
        self.assets["relay"] = self.crop_fit((427, 1047, 162, 132), (54, 44), transparent=True)
        self.assets["crystal"] = self.crop_fit((1106, 679, 44, 75), (26, 40), transparent=True)
        self.assets["rift"] = self.crop_fit((1494, 848, 131, 154), (54, 58), transparent=True)
        self.assets["core"] = self.crop_fit((1040, 1594, 160, 160), (72, 72), transparent=True)
        self.assets["reset"] = self.crop_scaled((1296, 1773, 109, 114), (42, 34), transparent=True)

    def get(self, name: str) -> pygame.Surface:
        return self.assets[name]


class ChronosCSP:
    def __init__(self) -> None:
        self.variables: Dict[str, Variable] = {
            "A": Variable("A", (1, 1)),
            "B": Variable("B", (4, 1), forbidden=("Violet",)),
            "C": Variable("C", (7, 1)),
            "D": Variable("D", (2, 3), forbidden=("Gold",)),
            "E": Variable("E", (5, 3)),
            "F": Variable("F", (8, 3), forbidden=("Cyan",)),
            "G": Variable("G", (3, 6)),
            "H": Variable("H", (7, 6), forbidden=("Green",)),
            "I": Variable("I", (1, 5), forbidden=("Cyan",)),
            "J": Variable("J", (6, 5), forbidden=("Violet",)),
        }
        self.edges: Set[Tuple[str, str]] = {
            self.edge("A", "B"),
            self.edge("A", "D"),
            self.edge("A", "I"),
            self.edge("B", "C"),
            self.edge("B", "D"),
            self.edge("B", "E"),
            self.edge("C", "E"),
            self.edge("C", "F"),
            self.edge("D", "E"),
            self.edge("D", "G"),
            self.edge("D", "I"),
            self.edge("D", "J"),
            self.edge("E", "F"),
            self.edge("E", "G"),
            self.edge("E", "H"),
            self.edge("E", "J"),
            self.edge("F", "H"),
            self.edge("F", "J"),
            self.edge("G", "H"),
            self.edge("G", "I"),
            self.edge("G", "J"),
            self.edge("H", "J"),
            self.edge("I", "J"),
            self.edge("C", "H"),
        }
        self.neighbors: Dict[str, Set[str]] = {name: set() for name in self.variables}
        for a, b in self.edges:
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)
        self.base_domains: Dict[str, List[Color]] = {
            name: [value for value in FREQUENCIES if value not in var.forbidden]
            for name, var in self.variables.items()
        }

    @staticmethod
    def edge(a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))  # type: ignore[return-value]

    def constraint_ok(self, left: str, left_value: Color, right: str, right_value: Color) -> bool:
        if left_value == right_value:
            return False
        pair = self.edge(left, right)
        if pair == ("C", "H") and {left_value, right_value} == {"Violet", "Gold"}:
            return False
        if pair == ("D", "J") and {left_value, right_value} == {"Cyan", "Gold"}:
            return False
        return True

    def total_cost(self, assignment: Dict[str, Color]) -> int:
        return sum(FREQ_COSTS[value] for value in assignment.values())

    def quota_counts(self, assignment: Dict[str, Color]) -> Dict[Color, int]:
        return {color: sum(1 for value in assignment.values() if value == color) for color in FREQUENCIES}

    def quota_possible(self, assignment: Dict[str, Color], domains: Optional[Dict[str, List[Color]]] = None) -> bool:
        counts = self.quota_counts(assignment)
        remaining = [name for name in self.variables if name not in assignment]
        for color, quota in FREQ_QUOTAS.items():
            if counts[color] > quota:
                return False
            if domains is None:
                possible = sum(1 for name in remaining if color in self.base_domains[name])
            else:
                possible = sum(1 for name in remaining if color in domains[name])
            if counts[color] + possible < quota:
                return False
        return True

    def assignment_conflicts(self, assignment: Dict[str, Color]) -> List[Tuple[str, str]]:
        conflicts: List[Tuple[str, str]] = []
        for a, b in self.edges:
            if a in assignment and b in assignment and not self.constraint_ok(a, assignment[a], b, assignment[b]):
                conflicts.append((a, b))
        return conflicts

    def unary_ok(self, var: str, value: Color) -> bool:
        return value not in self.variables[var].forbidden

    def complete(self, assignment: Dict[str, Color]) -> bool:
        return len(assignment) == len(self.variables) and self.quota_counts(assignment) == FREQ_QUOTAS

    def consistent(
        self,
        assignment: Dict[str, Color],
        var: str,
        value: Color,
        stats: Optional[Dict[str, int]] = None,
    ) -> bool:
        if not self.unary_ok(var, value):
            return False
        for neighbor in self.neighbors[var]:
            if neighbor not in assignment:
                continue
            if stats is not None:
                stats["checks"] += 1
            if not self.constraint_ok(var, value, neighbor, assignment[neighbor]):
                return False
        return True

    def select_unassigned(
        self,
        assignment: Dict[str, Color],
        domains: Dict[str, List[Color]],
        mode: str,
    ) -> str:
        remaining = []
        for name in self.variables:
            if name not in assignment:
                remaining.append(name)

        if mode == "plain":
            return remaining[0]
        

        # Minimum Remaining Values (MRV): pick the variable with the tightest constraints
        # (smallest remaining valid domain) to fail early if no solution exists.
        if mode == "mrv":
            best = remaining[0]
            for name in remaining:
                if len(domains[name]) < len(domains[best]):
                    best = name
            return best

        # Degree Heuristic: break MRV ties by selecting the variable that imposes constraints
        # on the largest number of remaining unassigned neighbors.
        if mode == "degree":
            min_size = len(domains[remaining[0]])
            for name in remaining:
                if len(domains[name]) < min_size:
                    min_size = len(domains[name])

            tied = []
            for name in remaining:
                if len(domains[name]) == min_size:
                    tied.append(name)

            best = tied[0]
            best_degree = -1
            for name in tied:
                degree = 0
                for neighbor in self.neighbors[name]:
                    if neighbor not in assignment:
                        degree += 1
                if degree > best_degree:
                    best_degree = degree
                    best = name
            return best

    def ordered_values(
        self,
        var: str,
        assignment: Dict[str, Color],
        domains: Dict[str, List[Color]],
        use_lcv: bool,
    ) -> List[Color]:
        if not use_lcv:
            return domains[var]
        
        # Least Constraining Value (LCV): sort frequencies by how many valid options
        # they eliminate from neighboring domains, prioritizing choices that maximize flexibility.
        def count_conflicts(value: Color) -> int:
            total = 0
            for neighbor in self.neighbors[var]:
                if neighbor in assignment:
                    continue
                for other in domains[neighbor]:
                    if not self.constraint_ok(var, value, neighbor, other):
                        total += 1
            return total

        return sorted(domains[var], key=count_conflicts)

    def forward_check(
        self,
        assignment: Dict[str, Color],
        domains: Dict[str, List[Color]],
        var: str,
        value: Color,
    ) -> Optional[Dict[str, List[Color]]]:
        next_domains = {name: values[:] for name, values in domains.items()}
        next_domains[var] = [value]
        for neighbor in self.neighbors[var]:
            if neighbor in assignment:
                continue
            filtered = [other for other in next_domains[neighbor] if self.constraint_ok(var, value, neighbor, other)]
            if not filtered:
                return None
            next_domains[neighbor] = filtered
        return next_domains

    def backtracking_search(
        self,
        algorithm: str,
        variable_mode: str = "plain",
        use_lcv: bool = False,
    ) -> CspReport:
        start_time = time.perf_counter()

        stats = {"nodes": 0, "backtracks": 0, "checks": 0}
        best = {"assignment": None, "cost": float("inf")}

        def backtrack(assignment: Dict[str, Color], domains: Dict[str, List[Color]]) -> None:
            stats["nodes"] += 1

            # Base case: if all variables are assigned, evaluate global ledger rules and cost.
            if len(assignment) == len(self.variables):
                if self.complete(assignment):
                    cost = self.total_cost(assignment)
                    if cost < best["cost"]:
                        best["cost"] = cost
                        best["assignment"] = dict(assignment)   
                return

            # Cost bounding pruning: abandon branches that already meet or exceed the cost 
            # of the optimal signal configuration found so far.
            current_cost = self.total_cost(assignment)
            if current_cost >= best["cost"]:
                return

            # Ledger admissibility pruning: verify if remaining domain slots can satisfy exactly 
            # the frequency usage requirements established by the global quotas.            
            if not self.quota_possible(assignment, domains):
                return

            var = self.select_unassigned(assignment, domains, variable_mode)

            values = self.ordered_values(var, assignment, domains, use_lcv)

            for value in values:
                if self.consistent(assignment, var, value, stats):
                    assignment[var] = value
                    next_domains = self.forward_check(assignment, domains, var, value)
                    if next_domains is not None:
                        backtrack(assignment, next_domains)   
                    else:
                        stats["backtracks"] += 1
                    del assignment[var] 

        initial_domains = {name: list(values) for name, values in self.base_domains.items()}
        backtrack({}, initial_domains)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        solved = best["assignment"] is not None
        message = "Signal plan sealed." if solved else "No valid signal pمهمه!lan found."

        return CspReport(
            algorithm=algorithm,
            solved=solved,
            assignment=best["assignment"] if solved else {},
            nodes=stats["nodes"],
            backtracks=stats["backtracks"],
            checks=stats["checks"],
            cost=best["cost"] if solved else 0,
            elapsed_ms=elapsed_ms,
            message=message,
        )

def line_points(a: Vec, b: Vec) -> Tuple[Vec, Vec]:
    return (
        (GRID_X + a[0] * TILE + TILE // 2, GRID_Y + a[1] * TILE + TILE // 2),
        (GRID_X + b[0] * TILE + TILE // 2, GRID_Y + b[1] * TILE + TILE // 2),
    )


class ChronosCspGame:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.atlas = TextureAtlas(ATLAS_PATH)
        self.problem = ChronosCSP()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.big_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.title_font = pygame.font.SysFont("segoeui", 21, bold=True)
        self.buttons = self.make_buttons()
        self.assignment: Dict[str, Color] = {}
        self.report: Optional[CspReport] = None
        self.status = "The forge is waiting for a signal plan."
        self.time = 0.0

    def make_buttons(self) -> List[Button]:
        specs = [
            ("Solve", "solve", "1"),
            ("LCV", "lcv", "2"),
            ("MRV", "mrv", "3"),
            ("Degree", "degree", "4"),
            ("Clear", "clear", "R"),
        ]
        buttons: List[Button] = []
        x = PANEL_X + 25
        y = 378
        for i, (label, command, hotkey) in enumerate(specs):
            rect = pygame.Rect(x + (i % 3) * 104, y + (i // 3) * 44, 92, 34)
            buttons.append(Button(rect, label, command, hotkey))
        return buttons

    def reset(self) -> None:
        self.assignment.clear()
        self.report = None
        self.status = "Signal runes cleared."

    def run_solver(self, command: str) -> None:
        try:
            if command == "solve":
                self.report = self.problem.backtracking_search("Solve")
            elif command == "lcv":
                self.report = self.problem.backtracking_search("LCV", use_lcv=True)
            elif command == "mrv":
                self.report = self.problem.backtracking_search("MRV", variable_mode="mrv")
            elif command == "degree":
                self.report = self.problem.backtracking_search("MRV + Degree", variable_mode="degree")
            else:
                return
        except NotImplementedError as exc:
            self.assignment.clear()
            self.report = None
            self.status = str(exc)
            return
        self.assignment = dict(self.report.assignment)
        self.status = self.report.message

    def cycle_variable(self, name: str) -> None:
        values = self.problem.base_domains[name]
        current = self.assignment.get(name)
        if current is None:
            self.assignment[name] = values[0]
        else:
            index = values.index(current)
            if index == len(values) - 1:
                del self.assignment[name]
            else:
                self.assignment[name] = values[index + 1]
        self.report = None
        self.update_manual_status(name)

    def clear_variable(self, name: str) -> None:
        if name in self.assignment:
            del self.assignment[name]
        self.report = None
        self.status = f"{RELAY_NAMES[name]} cleared."

    def update_manual_status(self, name: str) -> None:
        conflicts = self.problem.assignment_conflicts(self.assignment)
        counts = self.problem.quota_counts(self.assignment)
        if conflicts:
            self.status = f"{RELAY_NAMES[name]} tuned; {len(conflicts)} clash detected."
            return
        if len(self.assignment) == len(self.problem.variables):
            if counts == FREQ_QUOTAS:
                self.status = "Manual signal plan sealed. No clashes."
            else:
                self.status = "All relays lit; ledger quota still wrong."
            return
        remaining = len(self.problem.variables) - len(self.assignment)
        self.status = f"{RELAY_NAMES[name]} tuned. {remaining} relays remain."

    def variable_at(self, pos: Vec) -> Optional[str]:
        point = pygame.Vector2(pos)
        for name, var in self.problem.variables.items():
            center = line_points(var.pos, var.pos)[0]
            hitbox = pygame.Rect(0, 0, 86, 88)
            hitbox.center = center
            if hitbox.collidepoint(pos) or point.distance_to(center) <= 44:
                return name
        return None

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            for button in self.buttons:
                if event.button == 1 and button.contains(event.pos):
                    if button.command == "clear":
                        self.reset()
                    else:
                        self.run_solver(button.command)
                    return True
            var = self.variable_at(event.pos)
            if var is not None:
                if event.button == 3:
                    self.clear_variable(var)
                else:
                    self.cycle_variable(var)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_1:
                self.run_solver("solve")
            elif event.key == pygame.K_2:
                self.run_solver("lcv")
            elif event.key == pygame.K_3:
                self.run_solver("mrv")
            elif event.key == pygame.K_4:
                self.run_solver("degree")
            elif event.key == pygame.K_r:
                self.reset()
        return True

    def update(self, dt: float) -> None:
        self.time += dt

    def draw(self) -> None:
        self.screen.fill((7, 11, 18))
        self.draw_header()
        self.draw_board()
        self.draw_panel()
        pygame.display.flip()

    def draw_header(self) -> None:
        pygame.draw.rect(self.screen, (8, 13, 21), (0, 0, WIDTH, 74))
        title = self.big_font.render("Chronos Relay Forge", True, (231, 246, 255))
        self.screen.blit(title, (28, 18))
        subtitle = self.font.render("Seal linked relays; obey the signal ledger.", True, (133, 186, 205))
        self.screen.blit(subtitle, (330, 25))

    def draw_board(self) -> None:
        board = pygame.Rect(GRID_X - 8, GRID_Y - 8, GRID_W * TILE + 16, GRID_H * TILE + 16)
        pygame.draw.rect(self.screen, (5, 10, 17), board, border_radius=8)
        pygame.draw.rect(self.screen, (50, 91, 112), board, 2, border_radius=8)
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = pygame.Rect(GRID_X + x * TILE, GRID_Y + y * TILE, TILE, TILE)
                tile = self.atlas.tile_floor[(x * 3 + y * 5) % len(self.atlas.tile_floor)]
                self.screen.blit(tile, rect)
                if x in (0, GRID_W - 1) or y in (0, GRID_H - 1):
                    self.screen.blit(self.atlas.tile_wall[(x + y) % len(self.atlas.tile_wall)], rect)
                pygame.draw.rect(self.screen, (5, 13, 20), rect, 1)

        conflicts = set(self.problem.assignment_conflicts(self.assignment))
        for a, b in sorted(self.problem.edges):
            start, end = line_points(self.problem.variables[a].pos, self.problem.variables[b].pos)
            edge = self.problem.edge(a, b)
            color = (255, 92, 120) if edge in conflicts else (70, 137, 159)
            width = 5 if edge in conflicts else 3
            pygame.draw.line(self.screen, color, start, end, width)
            pygame.draw.line(self.screen, (10, 20, 27), start, end, 1)

        for name, var in self.problem.variables.items():
            self.draw_variable(name, var)

    def draw_variable(self, name: str, var: Variable) -> None:
        center = line_points(var.pos, var.pos)[0]
        value = self.assignment.get(name)
        pulse = int(4 * (1.0 + pygame.math.Vector2(1, 0).rotate(self.time * 90).x))
        if value is None:
            color = (93, 125, 148)
            pygame.draw.circle(self.screen, (31, 45, 58), center, 30)
            pygame.draw.circle(self.screen, color, center, 30, 2)
        else:
            color = FREQ_COLORS[value]
            glow = pygame.Surface((92, 92), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*color, 70), (46, 46), 34 + pulse)
            self.screen.blit(glow, (center[0] - 46, center[1] - 46))
            pygame.draw.circle(self.screen, (20, 26, 38), center, 32)
            pygame.draw.circle(self.screen, color, center, 32, 3)
        image = self.atlas.get("relay") if value is None else self.atlas.get("crystal")
        self.screen.blit(image, image.get_rect(center=center))
        label = self.small_font.render(RELAY_NAMES[name], True, (235, 248, 255))
        self.screen.blit(label, label.get_rect(center=(center[0], center[1] + 40)))
        if var.forbidden:
            text = self.small_font.render("rejects " + "/".join(var.forbidden), True, (255, 156, 144))
            self.screen.blit(text, text.get_rect(center=(center[0], center[1] - 42)))

    def draw_panel(self) -> None:
        panel = pygame.Rect(PANEL_X, 0, WIDTH - PANEL_X, HEIGHT)
        pygame.draw.rect(self.screen, (10, 16, 25), panel)
        pygame.draw.line(self.screen, (54, 91, 111), (PANEL_X, 0), (PANEL_X, HEIGHT), 2)
        icon = self.atlas.get("core")
        self.screen.blit(icon, (PANEL_X + 24, 24))
        title = self.big_font.render("Forge Console", True, (231, 246, 255))
        self.screen.blit(title, (PANEL_X + 104, 34))
        self.draw_domains(PANEL_X + 25, 108)
        self.draw_rules(PANEL_X + 25, 252)
        self.draw_buttons()
        self.draw_report()

    def draw_domains(self, x: int, y: int) -> None:
        header = self.title_font.render("Signal Ledger", True, (194, 231, 242))
        self.screen.blit(header, (x, y))
        for i, value in enumerate(FREQUENCIES):
            row_y = y + 32 + i * 28
            pygame.draw.circle(self.screen, FREQ_COLORS[value], (x + 12, row_y + 10), 9)
            label = self.font.render(f"{value}: {FREQ_QUOTAS[value]} slots, power {FREQ_COSTS[value]}", True, (225, 240, 245))
            self.screen.blit(label, (x + 30, row_y))

    def draw_rules(self, x: int, y: int) -> None:
        box = pygame.Rect(x, y, 330, 130) 
        pygame.draw.rect(self.screen, (12, 24, 35), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        title = self.small_font.render("Forge brief", True, (118, 220, 242))
        self.screen.blit(title, (x + 12, y + 9))
        for i, row in enumerate(RULE_LINES[:5]): 
            text = self.small_font.render(row, True, (190, 215, 225))
            self.screen.blit(text, (x + 12, y + 30 + i * 14))
        hint = self.small_font.render("Modes: Solve, LCV, MRV, MRV+Degree.", True, (118, 220, 242))
        self.screen.blit(hint, (x + 12, y + 106))

    def draw_buttons(self) -> None:
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            hover = button.contains(mouse)
            fill = (29, 59, 76) if hover else (19, 37, 51)
            border = (95, 232, 255) if hover else (58, 101, 121)
            pygame.draw.rect(self.screen, fill, button.rect, border_radius=7)
            pygame.draw.rect(self.screen, border, button.rect, 2, border_radius=7)
            text = self.font.render(button.label, True, (221, 246, 255))
            self.screen.blit(text, text.get_rect(center=(button.rect.centerx - 8, button.rect.centery)))
            key = self.small_font.render(button.hotkey, True, (139, 186, 204))
            self.screen.blit(key, (button.rect.right - 18, button.rect.y + 2))

    def draw_report(self) -> None:
        x = PANEL_X + 25
        y = 476
        box = pygame.Rect(x, y, 330, 185)
        pygame.draw.rect(self.screen, (14, 26, 38), box, border_radius=8)
        pygame.draw.rect(self.screen, (53, 88, 107), box, 1, border_radius=8)
        status_color = (95, 255, 176) if not self.problem.assignment_conflicts(self.assignment) else (255, 120, 130)
        status = self.small_font.render(self.status[:44], True, status_color)
        self.screen.blit(status, (x + 14, y + 14))
        if self.report is not None:
            rows = [
                f"Rite: {self.report.algorithm}",
                f"Sealed: {self.report.solved} | Power: {self.report.cost}",
                f"Nodes: {self.report.nodes}",
                f"Backtracks: {self.report.backtracks}",
                f"Checks: {self.report.checks}",
            ]
        else:
            conflicts = self.problem.assignment_conflicts(self.assignment)
            counts = self.problem.quota_counts(self.assignment)
            rows = [
                f"Lit: {len(self.assignment)}/{len(self.problem.variables)}",
                f"Clashes: {len(conflicts)}",
                "Ledger " + " ".join(f"{c[0]}:{counts[c]}/{FREQ_QUOTAS[c]}" for c in FREQUENCIES),
                f"Power draw: {self.problem.total_cost(self.assignment)}",
            ]
        for i, row in enumerate(rows):
            text = self.small_font.render(row, True, (199, 222, 232))
            self.screen.blit(text, (x + 14, y + 46 + i * 22))
        if self.report and self.report.log:
            log_y = y + 158 
            title = self.small_font.render("Trace", True, (118, 220, 242))
            self.screen.blit(title, (x + 14, log_y))
            row = self.report.log[-1]
            text = self.small_font.render(row[:42], True, (174, 201, 214))
            self.screen.blit(text, (x + 62, log_y))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Chronos Relay Forge")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game = ChronosCspGame(screen)
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
