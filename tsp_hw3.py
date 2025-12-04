#!/usr/bin/env python3
"""
Student ID: 925631693
"""

import sys
import time
import random
from typing import List, Tuple

def read_tsp_file(path: str) -> List[List[float]]:
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()

    n = int(lines[0].strip())
    dist = [[0.0] * n for _ in range(n)]

    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        i, j, d = parts
        i = int(i) - 1
        j = int(j) - 1
        d = float(d)
        dist[i][j] = d
        dist[j][i] = d

    return dist


def tour_length(tour: List[int], dist: List[List[float]]) -> float:
    n = len(tour)
    cost = 0.0
    for k in range(n - 1):
        cost += dist[tour[k]][tour[k + 1]]
    cost += dist[tour[-1]][tour[0]] 
    return cost


def nearest_neighbor_tour(start: int, dist: List[List[float]]) -> List[int]:
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_node = min(unvisited, key=lambda j: dist[current][j])
        unvisited.remove(next_node)
        tour.append(next_node)
        current = next_node
    return tour


# 2.5-opt

def two_point_five_opt_improvement(
    tour: List[int],
    dist: List[List[float]],
    current_cost: float,
    end_time: float,
    cycles_evaluated: int,
    max_tries_without_improve: int = 20000,
) -> Tuple[List[int], float, int]:
    n = len(tour)
    if n < 4:
        return tour, current_cost, cycles_evaluated

    no_improve = 0

    while time.time() < end_time and no_improve < max_tries_without_improve:
        n = len(tour)
        if n < 4:
            break

        # pick a node index k to relocate
        k = random.randrange(n)
        v = tour[k]

        prev = tour[k - 1]
        nxt = tour[(k + 1) % n]

        insert_pos = random.randrange(n - 1)

        if insert_pos == k or insert_pos == (k - 1) % n:
            no_improve += 1
            continue

        reduced = tour[:]
        reduced.pop(k) 

        if insert_pos >= len(reduced):
            insert_pos = len(reduced) - 1

        a = reduced[insert_pos]
        b = reduced[(insert_pos + 1) % len(reduced)]

        delta = -dist[prev][v] - dist[v][nxt] + dist[prev][nxt]
        delta += -dist[a][b] + dist[a][v] + dist[v][b]

        cycles_evaluated += 1

        if delta < -1e-12:
            new_tour = reduced[:]
            new_tour.insert(insert_pos + 1, v)
            tour = new_tour
            current_cost += delta
            no_improve = 0
        else:
            no_improve += 1

    return tour, current_cost, cycles_evaluated

def solve_tsp_25opt(
    dist: List[List[float]],
    time_limit_sec: float = 60.0,
    num_random_starts: int = 100,
) -> Tuple[List[int], float, int]:
    n = len(dist)
    start_time = time.time()
    end_time = start_time + time_limit_sec

    random.seed(0)

    best_tour = list(range(n))
    best_cost = tour_length(best_tour, dist)
    cycles_evaluated = 1 

    print(f"Solving graph with fixed time limit {time_limit_sec} seconds using NN + 2.5-opt...")

    while time.time() < end_time:
        if num_random_starts is not None and num_random_starts <= 0:
            break

        num_random_starts -= 1

        if random.random() < 0.5:
            start_node = random.randrange(n)
            tour = nearest_neighbor_tour(start_node, dist)
        else:
            tour = list(range(n))
            random.shuffle(tour)

        current_cost = tour_length(tour, dist)
        cycles_evaluated += 1 

        tour, current_cost, cycles_evaluated = two_point_five_opt_improvement(
            tour,
            dist,
            current_cost,
            end_time,
            cycles_evaluated,
        )

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = tour[:]
            print(
                f"[{time.time() - start_time:6.2f}s] "
                f"New best cost: {best_cost:.2f}   cycles≈{cycles_evaluated}"
            )

    return best_tour, best_cost, cycles_evaluated


def write_cycle_line(tour: List[int], path: str, mode: str = "w"):

    tour_1_based = [v + 1 for v in tour]
    cycle = tour_1_based + [tour_1_based[0]]

    with open(path, mode) as f:
        f.write(", ".join(str(v) for v in cycle) + "\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python tsp_hw3.py <euclid_tsp_file> <random_tsp_file> [ignored_solution_name] [ignored_time]")
        print("Note: This script always uses a 60-second limit and writes solution_925631693.txt")
        sys.exit(1)

    euclid_file = sys.argv[1]
    random_file = sys.argv[2]

    time_limit = 60.0
    out_file = "solution_925631693.txt"

    print(f"Reading Euclidean instance from {euclid_file} ...")
    dist_euclid = read_tsp_file(euclid_file)
    print(f"Loaded Euclidean complete graph with {len(dist_euclid)} nodes.\n")

    best_tour_e, best_cost_e, cycles_eval_e = solve_tsp_25opt(
        dist_euclid, time_limit_sec=time_limit
    )

    print("\n===== EUCLIDEAN RESULT (NN + 2.5-opt) =====")
    print(f"Best tour cost (Euclidean): {best_cost_e:.2f}")
    print(f"Approx. number of cycles evaluated (Euclidean): {cycles_eval_e:e}")

    write_cycle_line(best_tour_e, out_file, mode="w")

    print(f"\nReading random instance from {random_file} ...")
    dist_rand = read_tsp_file(random_file)
    print(f"Loaded random complete graph with {len(dist_rand)} nodes.\n")

    best_tour_r, best_cost_r, cycles_eval_r = solve_tsp_25opt(
        dist_rand, time_limit_sec=time_limit
    )

    print("\n===== RANDOM RESULT (NN + 2.5-opt) =====")
    print(f"Best tour cost (Random): {best_cost_r:.2f}")
    print(f"Approx. number of cycles evaluated (Random): {cycles_eval_r:e}")

    write_cycle_line(best_tour_r, out_file, mode="a")

    print(f"\nWrote solution cycles to: {out_file}")
    print("Line 1: Euclidean graph cycle (NN + 2.5-opt)")
    print("Line 2: Random graph cycle    (NN + 2.5-opt)")


if __name__ == "__main__":
    main()