#!/usr/bin/env python3
"""Benchmark harness for hnswlib: recall, QPS, and memory.

Modes:
  static  Build an index once, sweep ef, and report recall@k, QPS,
          build time, index file size, and process RSS.
  churn   Sliding-window workload of interleaved deletes, inserts, and
          queries (e.g. 5% of the index replaced per round) to measure
          recall drift, write/query throughput, and memory growth under
          realistic traffic.

Examples:
  python3 bench/benchmark.py static --space l2 --dim 128 --num-elements 100000 \
      --ef 10 20 50 100 200
  python3 bench/benchmark.py churn --space cosine --dim 128 --num-elements 50000 \
      --rounds 20 --churn-fraction 0.05 --delete-mode mark --replace-deleted
"""

import argparse
import json
import resource
import sys
import time

import numpy as np

import hnswlib


def rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def make_data(kind, n, dim, rng):
    if kind == "gaussian":
        return rng.standard_normal((n, dim)).astype(np.float32)
    if kind == "clustered":
        num_clusters = 64
        centers = rng.standard_normal((num_clusters, dim)).astype(np.float32) * 3.0
        assignment = rng.integers(0, num_clusters, n)
        return (centers[assignment] + rng.standard_normal((n, dim)).astype(np.float32)).astype(np.float32)
    # otherwise treat as a path to a .npy file of float32 vectors
    data = np.load(kind)
    if data.shape[0] < n:
        raise ValueError(f"dataset {kind} has {data.shape[0]} rows, need {n}")
    return np.ascontiguousarray(data[:n], dtype=np.float32)


def recall_at_k(found_labels, true_labels):
    hits = sum(len(set(f) & set(t)) for f, t in zip(found_labels, true_labels))
    return hits / float(true_labels.shape[0] * true_labels.shape[1])


def measure_qps(index, queries, k, threads, min_seconds=0.5):
    """Return (qps, labels_of_first_batch)."""
    labels, _ = index.knn_query(queries, k=k, num_threads=threads)
    batches = 1
    start = time.perf_counter()
    while True:
        index.knn_query(queries, k=k, num_threads=threads)
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            break
        batches += 1
    return batches * len(queries) / elapsed, labels


def print_table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def output(args, results):
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


def bench_static(args):
    rng = np.random.default_rng(args.seed)
    print(f"# static: space={args.space} dim={args.dim} n={args.num_elements} "
          f"k={args.k} M={args.M} ef_construction={args.ef_construction} dataset={args.dataset}")

    data = make_data(args.dataset, args.num_elements + args.num_queries, args.dim, rng)
    queries = data[args.num_elements:]
    data = data[:args.num_elements]

    rss_before = rss_mb()
    index = hnswlib.Index(space=args.space, dim=args.dim)
    index.init_index(max_elements=args.num_elements, M=args.M, ef_construction=args.ef_construction)
    t0 = time.perf_counter()
    index.add_items(data, num_threads=args.build_threads)
    build_s = time.perf_counter() - t0
    rss_after = rss_mb()

    bf = hnswlib.BFIndex(space=args.space, dim=args.dim)
    bf.init_index(max_elements=args.num_elements)
    bf.add_items(data)
    true_labels, _ = bf.knn_query(queries, k=args.k)

    print(f"build: {build_s:.2f}s ({args.num_elements / build_s:.0f} vec/s, {args.build_threads} threads)")
    print(f"index_file_size: {index.index_file_size() / 1e6:.1f} MB, peak RSS: {rss_after:.0f} MB (+{rss_after - rss_before:.0f} MB)")
    print()

    rows = []
    results = {"mode": "static", "space": args.space, "dim": args.dim,
               "num_elements": args.num_elements, "k": args.k, "M": args.M,
               "ef_construction": args.ef_construction, "build_seconds": build_s,
               "index_file_size_bytes": index.index_file_size(), "peak_rss_mb": rss_after,
               "sweep": []}
    for ef in args.ef:
        index.set_ef(ef)
        qps, labels = measure_qps(index, queries, args.k, args.query_threads)
        rec = recall_at_k(labels, true_labels)
        rows.append([ef, f"{rec:.4f}", f"{qps:.0f}"])
        results["sweep"].append({"ef": ef, "recall": rec, "qps": qps})
    print_table(rows, ["ef", f"recall@{args.k}", f"QPS ({args.query_threads} thr)"])
    output(args, results)


def bench_churn(args):
    rng = np.random.default_rng(args.seed)
    churn = int(args.num_elements * args.churn_fraction)
    total_needed = args.num_elements + args.rounds * churn + args.num_queries
    print(f"# churn: space={args.space} dim={args.dim} n={args.num_elements} "
          f"k={args.k} rounds={args.rounds} churn={churn}/round "
          f"delete_mode={args.delete_mode} replace_deleted={args.replace_deleted} dataset={args.dataset}")

    pool = make_data(args.dataset, total_needed, args.dim, rng)
    queries = pool[-args.num_queries:]
    pool = pool[:-args.num_queries]

    # capacity: with tombstone deletes and no replacement the index must grow
    if args.delete_mode == "mark" and not args.replace_deleted:
        capacity = args.num_elements + args.rounds * churn
    else:
        capacity = args.num_elements

    index = hnswlib.Index(space=args.space, dim=args.dim)
    index.init_index(max_elements=capacity, M=args.M,
                     ef_construction=args.ef_construction,
                     allow_replace_deleted=(args.delete_mode == "mark" and args.replace_deleted))
    index.set_ef(args.ef[0])

    bf = hnswlib.BFIndex(space=args.space, dim=args.dim)
    bf.init_index(max_elements=args.num_elements)

    def delete_label(label):
        if args.delete_mode == "mark":
            index.mark_deleted(label)
        elif args.delete_mode == "remove":
            index.remove_item(label)
        else:
            raise ValueError(f"unknown delete mode {args.delete_mode}")

    t0 = time.perf_counter()
    index.add_items(pool[:args.num_elements], np.arange(args.num_elements),
                    num_threads=args.build_threads)
    build_s = time.perf_counter() - t0
    bf.add_items(pool[:args.num_elements], np.arange(args.num_elements))
    print(f"initial build: {build_s:.2f}s, peak RSS {rss_mb():.0f} MB\n")

    oldest = 0       # sliding window start (label of oldest live element)
    next_label = args.num_elements
    rows = []
    results = {"mode": "churn", "space": args.space, "dim": args.dim,
               "num_elements": args.num_elements, "k": args.k, "M": args.M,
               "ef": args.ef[0], "rounds": args.rounds, "churn_per_round": churn,
               "delete_mode": args.delete_mode, "replace_deleted": args.replace_deleted,
               "build_seconds": build_s, "rounds_data": []}

    for rnd in range(1, args.rounds + 1):
        # delete oldest `churn` elements
        t0 = time.perf_counter()
        for label in range(oldest, oldest + churn):
            delete_label(label)
        delete_s = time.perf_counter() - t0
        for label in range(oldest, oldest + churn):
            bf.delete_vector(label)
        oldest += churn

        # insert `churn` new elements
        new_labels = np.arange(next_label, next_label + churn)
        new_data = pool[next_label:next_label + churn]
        t0 = time.perf_counter()
        index.add_items(new_data, new_labels, num_threads=args.build_threads,
                        replace_deleted=(args.delete_mode == "mark" and args.replace_deleted))
        insert_s = time.perf_counter() - t0
        bf.add_items(new_data, new_labels)
        next_label += churn

        # query and score against exact ground truth over the live set
        true_labels, _ = bf.knn_query(queries, k=args.k)
        qps, labels = measure_qps(index, queries, args.k, args.query_threads)
        rec = recall_at_k(labels, true_labels)

        rows.append([rnd, next_label - oldest, f"{rec:.4f}", f"{qps:.0f}",
                     f"{churn / delete_s:.0f}", f"{churn / insert_s:.0f}",
                     f"{index.index_file_size() / 1e6:.1f}", f"{rss_mb():.0f}"])
        results["rounds_data"].append({
            "round": rnd, "live": next_label - oldest, "recall": rec, "qps": qps,
            "deletes_per_s": churn / delete_s, "inserts_per_s": churn / insert_s,
            "index_file_size_bytes": index.index_file_size(), "peak_rss_mb": rss_mb()})

    print_table(rows, ["round", "live", f"recall@{args.k}", "QPS",
                       "del/s", "ins/s", "file MB", "RSS MB"])
    output(args, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--space", default="l2", help="l2, ip, or cosine (default l2)")
    common.add_argument("--dim", type=int, default=128)
    common.add_argument("--num-elements", type=int, default=50000)
    common.add_argument("--num-queries", type=int, default=500)
    common.add_argument("--k", type=int, default=10)
    common.add_argument("--M", type=int, default=16)
    common.add_argument("--ef-construction", type=int, default=200)
    common.add_argument("--ef", type=int, nargs="+", default=[10, 20, 50, 100, 200],
                        help="ef values to sweep (static) or single search ef (churn, first value used)")
    common.add_argument("--dataset", default="clustered",
                        help="'gaussian', 'clustered', or path to a .npy file (default clustered)")
    common.add_argument("--build-threads", type=int, default=-1)
    common.add_argument("--query-threads", type=int, default=1)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--json", help="write results to this JSON file")

    sub.add_parser("static", parents=[common], help="recall/QPS sweep on a fixed index")

    churn_p = sub.add_parser("churn", parents=[common],
                             help="interleaved delete/insert/query traffic")
    churn_p.add_argument("--rounds", type=int, default=20)
    churn_p.add_argument("--churn-fraction", type=float, default=0.05,
                         help="fraction of the index deleted+reinserted per round (default 0.05)")
    churn_p.add_argument("--delete-mode", default="mark", choices=["mark", "remove"],
                         help="mark: mark_deleted tombstones; remove: true removal (requires remove_item support)")
    churn_p.add_argument("--replace-deleted", action="store_true",
                         help="with --delete-mode mark, reuse tombstoned slots for new inserts")

    args = parser.parse_args()
    if args.mode == "static":
        bench_static(args)
    else:
        bench_churn(args)


if __name__ == "__main__":
    main()
