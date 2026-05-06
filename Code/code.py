import argparse
import numpy as np


def build_theta(num_workers):
    pair_count = num_workers // 2
    step = 0.9 / max(pair_count, 1)
    magnitudes = [step * (idx + 1) for idx in range(pair_count)]
    theta = []
    for magnitude in magnitudes:
        theta.extend([magnitude, -magnitude])
    if num_workers % 2 == 1:
        theta.append(0.0)
    theta = np.array(theta[:num_workers], dtype=float)
    return theta[np.argsort(np.abs(theta))]


def pad_gradients(gradients, block_size):
    original_dim = gradients.shape[1]
    padded_dim = int(np.ceil(original_dim / block_size) * block_size)
    if padded_dim == original_dim:
        return gradients, original_dim
    padded = np.zeros((gradients.shape[0], padded_dim), dtype=gradients.dtype)
    padded[:, :original_dim] = gradients
    return padded, original_dim


def build_vandermonde(theta_values, width):
    return np.array([[theta ** power for power in range(width)] for theta in theta_values], dtype=float)


def build_balanced_assignment(num_workers, num_datasets, datasets_per_worker, min_replicas):
    if datasets_per_worker > num_datasets:
        raise ValueError("d cannot exceed k.")
    if num_workers * datasets_per_worker < num_datasets * min_replicas:
        raise ValueError("The requested parameters do not satisfy the replica budget.")

    assignment = np.zeros((num_workers, num_datasets), dtype=bool)
    worker_load = np.zeros(num_workers, dtype=int)
    dataset_load = np.zeros(num_datasets, dtype=int)

    for dataset in range(num_datasets):
        for _ in range(min_replicas):
            candidates = [worker for worker in range(num_workers) if not assignment[worker, dataset] and worker_load[worker] < datasets_per_worker]
            if not candidates:
                raise ValueError("Unable to build a feasible assignment matrix.")
            worker = min(candidates, key=lambda idx: (worker_load[idx], idx))
            assignment[worker, dataset] = True
            worker_load[worker] += 1
            dataset_load[dataset] += 1

    for worker in range(num_workers):
        while worker_load[worker] < datasets_per_worker:
            candidates = [dataset for dataset in range(num_datasets) if not assignment[worker, dataset]]
            if not candidates:
                raise ValueError("Unable to complete the assignment matrix.")
            dataset = min(candidates, key=lambda idx: (dataset_load[idx], idx))
            assignment[worker, dataset] = True
            worker_load[worker] += 1
            dataset_load[dataset] += 1

    return assignment


def main():
    parser = argparse.ArgumentParser(description="Simulate coded distributed gradient computation.")
    parser.add_argument("--n", type=int, default=20, help="Number of workers")
    parser.add_argument("--k", type=int, default=40, help="Number of data subsets")
    parser.add_argument("--d", type=int, default=6, help="Data subsets assigned to each worker")
    parser.add_argument("--s", type=int, default=1, help="Number of stragglers tolerated")
    parser.add_argument("--m", type=int, default=2, help="Communication reduction factor")
    parser.add_argument("--l", type=int, default=4, help="Dimension of gradient vectors")
    args = parser.parse_args()

    n, k, d, s, m, l = args.n, args.k, args.d, args.s, args.m, args.l

    assert d >= s + m, "Tradeoff condition not met."
    assert n > s, "Need at least one surviving worker."
    assert n - s >= m, "Need at least m recovery coefficients after stragglers."
    assert n * d >= k * (s + m), "The chosen n, k, d, s, m do not admit enough replicas."

    theta = build_theta(n)
    assert len(theta) == n, "Theta vector length mismatch."
    print("Theta values:", theta)

    assignment = build_balanced_assignment(n, k, d, s + m)
    print("Per-worker assignment counts:", assignment.sum(axis=1))
    print("Per-dataset replica counts:", assignment.sum(axis=0))

    np.random.seed(42)
    G = np.random.rand(k, l)
    true_sum = np.sum(G, axis=0)

    G_padded, original_dim = pad_gradients(G, m)
    q = G_padded.shape[1] // m
    vandermonde = build_vandermonde(theta, n - s)

    print("--- Partial Gradients ---")
    for j in range(G.shape[0]):
        print(f"Dataset {j}: {G[j]}")
    print(f"\n>>> True Sum Gradient: {true_sum}\n")

    F_workers = np.zeros((n, q))

    for b in range(q):
        block_start = b * m
        block_end = block_start + m
        block = slice(block_start, block_end)

        block_sum = np.sum(G_padded[:, block], axis=0)
        coeff = np.zeros(n - s)
        coeff[:m] = block_sum

        if n - s > m:
            worker_summaries = assignment.astype(float) @ np.sum(G_padded[:, block], axis=1)
            extra = min(n - s - m, worker_summaries.shape[0])
            coeff[m:m + extra] = worker_summaries[:extra]

        F_workers[:, b] = vandermonde @ coeff

    print("--- Transmissions to Master ---")
    for i in range(n):
        print(f"Worker {i} transmits: {F_workers[i]}")

    straggler_indices = list(range(s))
    print(f"Workers {straggler_indices} are stragglers; master decodes using the remaining {n - s} workers.\n")

    survivors = [i for i in range(n) if i not in straggler_indices]
    F_surv = F_workers[survivors]
    A_surv = vandermonde[survivors]

    recovered_sum = np.zeros(original_dim)
    for b in range(q):
        block_start = b * m
        block_end = min(block_start + m, original_dim)
        block_width = block_end - block_start
        coeff_rec = np.linalg.solve(A_surv, F_surv[:, b])
        recovered_sum[block_start:block_end] = coeff_rec[:block_width]

    print("--- Decoding Results ---")
    print(f"Recovered Sum Gradient: {recovered_sum}")
    print(f"True Sum Gradient: {true_sum}")
    difference = np.linalg.norm(true_sum - recovered_sum)
    print(f"Difference from True Sum: {difference:.2e}")
    if difference < 1e-9:
        print("\nSUCCESS: The coded sum matches the true sum precisely!")
    else:
        print("\nFAILURE: Mismatch detected.")


if __name__ == "__main__":
    main()
    



