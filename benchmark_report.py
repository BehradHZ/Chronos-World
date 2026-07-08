"""
Benchmark script: compares the completed (mine) implementation against the
original baseline implementation for both parts of the Chronos project.

Part 1 (Weaver): runs BFS, DFS, UCS, Greedy, A* on the default map and
reports cost / visited states / frontier peak / runtime for both versions.

Part 2 (Duel): runs Minimax, Alpha-Beta, Expectimax from the initial duel
state (both as ENEMY and PLAYER perspective) and reports nodes / depth /
runtime for both versions. Also isolates the effect of move ordering on
Alpha-Beta at increasing depths.

Usage:
    python3 benchmark_report.py

Requires chronos_weaver_todo.py, chronos_weaver.py (baseline),
chronos_duel_todo.py, chronos_duel.py (baseline), and Source.png
all in the same folder as this script.
"""

import os
import math

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

import chronos_weaver_todo as weaver_mine  # noqa: E402
import chronos_weaver as weaver_base  # noqa: E402
import chronos_duel_todo as duel_mine  # noqa: E402
import chronos_duel as duel_base  # noqa: E402


def line(char="-", width=100):
    print(char * width)


def title(text):
    print()
    line("=")
    print(text)
    line("=")


# ---------------------------------------------------------------------------
# PART 1: Chronos Weaver
# ---------------------------------------------------------------------------

def run_weaver_benchmark():
    title("PART 1: CHRONOS WEAVER  (search algorithms)")

    world_mine = weaver_mine.World()
    world_base = weaver_base.World()
    start_mine = world_mine.initial_state()
    start_base = world_base.initial_state()

    algorithms = ["BFS", "DFS", "UCS", "Greedy", "A*"]
    rows = []

    for algo in algorithms:
        rm = weaver_mine.solve(world_mine, start_mine, algo)
        rb = weaver_base.solve(world_base, start_base, algo)
        rows.append((algo, rm, rb))

    header = f"{'Algorithm':10s} | {'MINE cost':>10s} {'MINE visited':>13s} {'MINE peak':>10s} {'MINE ms':>9s} | {'BASE cost':>10s} {'BASE visited':>13s} {'BASE peak':>10s} {'BASE ms':>9s}"
    print(header)
    line()
    for algo, rm, rb in rows:
        print(
            f"{algo:10s} | {rm.cost:10.1f} {rm.visited:13d} {rm.frontier_peak:10d} {rm.elapsed_ms:9.2f} | "
            f"{rb.cost:10.1f} {rb.visited:13d} {rb.frontier_peak:10d} {rb.elapsed_ms:9.2f}"
        )

    print()
    print("Heuristic value at the initial state:")
    print(f"  mine heuristic:     {world_mine.heuristic(start_mine):.2f}")
    print(f"  baseline heuristic: {world_base.heuristic(start_base):.2f}")

    print()
    print("Headline comparison (A* vs baseline A*):")
    a_mine = next(rm for algo, rm, _ in rows if algo == "A*")
    a_base = next(rb for algo, _, rb in rows if algo == "A*")
    reduction = (1 - a_mine.visited / a_base.visited) * 100 if a_base.visited else 0.0
    same_cost = "YES" if abs(a_mine.cost - a_base.cost) < 1e-9 else "NO -- CHECK THIS"
    print(f"  optimal cost preserved: {same_cost}  (mine={a_mine.cost:.1f}, baseline={a_base.cost:.1f})")
    print(f"  visited-states reduction: {reduction:.1f}%  (mine={a_mine.visited}, baseline={a_base.visited})")


# ---------------------------------------------------------------------------
# PART 2: Chronos Duel
# ---------------------------------------------------------------------------

def run_duel_benchmark():
    title("PART 2: CHRONOS DUEL  (adversarial search)")

    world_mine = duel_mine.DuelWorld()
    world_base = duel_base.DuelWorld()
    state_mine = world_mine.initial_state()
    state_base = world_base.initial_state()

    algorithms = ["Minimax", "Alpha-Beta", "Expectimax"]

    for perspective_name, persp_mine, persp_base in [
        ("ENEMY (Chaos)", duel_mine.ENEMY, duel_base.ENEMY),
        ("PLAYER (Chronos)", duel_mine.PLAYER, duel_base.PLAYER),
    ]:
        print()
        print(f"-- perspective: {perspective_name} --")
        header = f"{'Algorithm':12s} | {'MINE action':>12s} {'MINE score':>11s} {'MINE nodes':>11s} {'MINE depth':>10s} {'MINE ms':>9s} | {'BASE action':>12s} {'BASE score':>11s} {'BASE nodes':>11s} {'BASE ms':>9s}"
        print(header)
        line()
        for algo in algorithms:
            rm = duel_mine.choose_ai_action(world_mine, state_mine, algo, persp_mine)
            rb = duel_base.choose_ai_action(world_base, state_base, algo, persp_base)
            print(
                f"{algo:12s} | {rm.action:>12s} {rm.score:11.2f} {rm.nodes:11d} {rm.depth:10d} {rm.elapsed_ms:9.2f} | "
                f"{rb.action:>12s} {rb.score:11.2f} {rb.nodes:11d} {rb.elapsed_ms:9.2f}"
            )

    print()
    print("Headline comparison (Alpha-Beta vs baseline Alpha-Beta, depth=5, ENEMY perspective):")
    rm = duel_mine.choose_ai_action(world_mine, state_mine, "Alpha-Beta", duel_mine.ENEMY)
    rb = duel_base.choose_ai_action(world_base, state_base, "Alpha-Beta", duel_base.ENEMY)
    reduction = (1 - rm.nodes / rb.nodes) * 100 if rb.nodes else 0.0
    print(f"  nodes reduction: {reduction:.1f}%  (mine={rm.nodes}, baseline={rb.nodes})")


def run_correctness_check():
    title("CORRECTNESS CHECK: Alpha-Beta must match Minimax at the same depth")

    world = duel_mine.DuelWorld()
    state = world.initial_state()
    perspective = duel_mine.ENEMY

    def minimax(s, d):
        if d == 0 or world.terminal(s):
            v = world.evaluate_enemy(s)
            return v if perspective == duel_mine.ENEMY else -v
        actions = world.legal_actions(s)
        if s.turn == perspective:
            return max(minimax(world.apply_action(s, a), d - 1) for a in actions)
        return min(minimax(world.apply_action(s, a), d - 1) for a in actions)

    def alpha_beta(s, d, alpha, beta):
        if d == 0 or world.terminal(s):
            v = world.evaluate_enemy(s)
            return v if perspective == duel_mine.ENEMY else -v
        actions = duel_mine.ordered_actions(world, s)
        if s.turn == perspective:
            best = -math.inf
            for a in actions:
                best = max(best, alpha_beta(world.apply_action(s, a), d - 1, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        best = math.inf
        for a in actions:
            best = min(best, alpha_beta(world.apply_action(s, a), d - 1, alpha, beta))
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    print(f"{'depth':6s} {'minimax score':15s} {'alpha-beta score':18s} {'match?':8s}")
    line()
    for depth in [2, 3, 4, 5]:
        best_mm, best_ab = -math.inf, -math.inf
        for a in duel_mine.ordered_actions(world, state):
            child = world.apply_action(state, a)
            best_mm = max(best_mm, minimax(child, depth - 1))
            best_ab = max(best_ab, alpha_beta(child, depth - 1, -math.inf, math.inf))
        match = "YES" if abs(best_mm - best_ab) < 1e-9 else "NO -- BUG"
        print(f"{depth:6d} {best_mm:15.2f} {best_ab:18.2f} {match:8s}")


def run_ordering_effect():
    title("MOVE-ORDERING EFFECT ON ALPHA-BETA (mine implementation, ENEMY perspective)")

    world = duel_mine.DuelWorld()
    state = world.initial_state()
    perspective = duel_mine.ENEMY

    def run(depth, use_ordering):
        nodes = 0

        def terminal_value(s, d):
            nonlocal nodes
            if d == 0 or world.terminal(s):
                nodes += 1
                v = world.evaluate_enemy(s)
                return v if perspective == duel_mine.ENEMY else -v
            return None

        def get_actions(s):
            return duel_mine.ordered_actions(world, s) if use_ordering else world.legal_actions(s)

        def ab(s, d, alpha, beta):
            v = terminal_value(s, d)
            if v is not None:
                return v
            actions = get_actions(s)
            if s.turn == perspective:
                best = -math.inf
                for a in actions:
                    best = max(best, ab(world.apply_action(s, a), d - 1, alpha, beta))
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        break
                return best
            best = math.inf
            for a in actions:
                best = min(best, ab(world.apply_action(s, a), d - 1, alpha, beta))
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

        best_score = -math.inf
        for a in get_actions(state):
            child = world.apply_action(state, a)
            best_score = max(best_score, ab(child, depth - 1, -math.inf, math.inf))
        return nodes

    print(f"{'depth':6s} {'nodes WITH ordering':20s} {'nodes WITHOUT ordering':23s} {'reduction':10s}")
    line()
    for depth in [1, 2, 3, 4, 5, 6, 7]:
        n_with = run(depth, True)
        n_without = run(depth, False)
        reduction = (1 - n_with / n_without) * 100 if n_without else 0.0
        print(f"{depth:6d} {n_with:20d} {n_without:23d} {reduction:9.1f}%")


if __name__ == "__main__":
    run_weaver_benchmark()
    run_duel_benchmark()
    run_correctness_check()
    run_ordering_effect()
    print()
    line("=")
    print("Done. Copy the tables above directly into the project report.")
    line("=")