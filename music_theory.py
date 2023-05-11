import fractions
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pathlib

sieve_max = 1000
sieve = np.ones(sieve_max, dtype=bool)
sieve[0] = sieve[1] = 0
for i in range(2, int(np.sqrt(sieve_max))):
    sieve[i * 2::i] = 0
primes = np.squeeze(np.argwhere(sieve)).tolist()


def _generate_all_partitions_core(total, steps, min_size, max_size):
    if steps <= 0 or total < min_size:
        return
    if steps == 1:
        if min_size <= total <= max_size:
            yield [total]
        return
    for step in range(min_size, max_size + 1):
        for partition in generate_all_partitions(
                total - step, steps - 1, min_size, max_size
        ):
            yield [step] + partition


_memoization_table = {}


def generate_all_partitions(total, steps, min_size, max_size):
    args = (total, steps, min_size, max_size)
    if args in _memoization_table.keys():
        yield from _memoization_table[args]
        return
    partitions = []
    for partition in _generate_all_partitions_core(*args):
        partitions.append(partition)
        yield partition
    _memoization_table[args] = partitions


def rotations(x):
    for rotate in range(len(x)):
        yield x[rotate:] + x[:rotate]


def normalize(partition):
    return min(rotations(partition))


def highest_prime_factor(n):
    if n == 1:
        return 1
    for prime in primes[::-1]:
        if n % prime == 0:
            return prime
    return np.inf


def tenney_height(n, d):
    return n * d


def nice_height(n, d):
    return tenney_height(n, d) * highest_prime_factor(n * d)


def max_interval(edo):
    return int(np.ceil(edo / 2 + 1))


class ConsonanceModel:

    def __init__(self, height=nice_height, use_unison=False):
        self.spread = 17.0
        fractions_ = {
            fractions.Fraction(n, d)
            for n in range(1, 100)
            for d in range(1, 100)
            if 1 <= n / d < np.sqrt(2)
        }
        if not use_unison:
            fractions_ = {f for f in fractions_ if f != 1}
        ratios = np.array([float(x) for x in fractions_])
        self.cents = 1200 * np.log2(ratios)
        self.weights = 1 / np.array([height(x.numerator, x.denominator) for x in fractions_])

    def calculate(self, cents):
        normalized_cents = cents % 1200.0
        if normalized_cents > 600.0:
            normalized_cents = 1200.0 - normalized_cents
        diff = (normalized_cents - self.cents) / self.spread
        return np.sum(np.exp(-diff * diff) * self.weights)

    def make_table(self, edo):
        result = []
        for i in range(max_interval(edo) + 1):
            result.append(self.calculate(1200.0 * i / edo))
        return result

    def plot(self, axes):
        cents = np.linspace(0.0, 1200.0, 10_000)
        axes.set_xlabel("Interval (cents)")
        axes.set_ylabel("Consonance (arb.)")
        axes.plot(cents, [self.calculate(x) for x in cents])
        return axes


def generate_scale_classes(edo, scale_size, min_step, max_step):
    partition_classes = set()
    for partition in generate_all_partitions(edo, scale_size, min_step, max_step):
        normalized = tuple(normalize(partition))
        partition_classes.add(normalized)
    return [np.concatenate([[0], np.cumsum(x)[:-1]]) for x in partition_classes]


def generate_scales(edo, scale_classes):
    binary_scales = []
    scale_class_indices = []
    for index, scale_class in enumerate(scale_classes):
        transpositions = set()
        for transposition in range(edo):
            scale = [0] * edo
            for pitch_class in scale_class:
                scale[(pitch_class + transposition) % edo] = 1
            transpositions.add(tuple(scale))
        for scale in transpositions:
            binary_scales.append(scale)
            scale_class_indices.append(index)
    return np.array(binary_scales), scale_class_indices


def normalize_interval(interval, edo):
    abs_interval = abs(interval)
    if abs_interval > edo / 2:
        return edo - abs_interval
    return abs_interval


def interval_vector(scale, edo):
    vector = [0 for __ in range(max_interval(edo) + 1)]
    for note_1, note_2 in itertools.combinations(scale, 2):
        interval = normalize_interval(note_1 - note_2, edo)
        vector[interval] += 1
    return vector


def smallest_third(scale, edo):
    roll = 2
    rotation = np.concatenate([scale[roll:], scale[:roll]])
    return min(normalize_interval(x - y, edo) for x, y in zip(rotation, scale))


def select_best_scale_classes(scale_classes, edo, consonance_table, count):
    dissonances = [
        -np.dot(consonance_table, interval_vector(scale, edo)) for scale in scale_classes
    ]
    partition = np.argpartition(dissonances, count)
    np_scales = np.array(scale_classes)
    return np_scales[partition[:count], :]


def main():
    out_dir = pathlib.Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)

    specs = [
        ("A. Tenney height", ConsonanceModel(height=tenney_height, use_unison=True)),
        ("B. Unisons removed", ConsonanceModel(height=tenney_height)),
        ("C. Prime limit penalty", ConsonanceModel(height=nice_height)),
    ]

    figure, axes = plt.subplots(1, len(specs), figsize=(6.0 * len(specs), 6.0))

    for i, (title, model) in enumerate(specs):
        ax = axes[i]
        ax.set_title(title)
        model.plot(ax)

    figure.savefig(str(out_dir / "figures.png"))


if __name__ == "__main__":
    main()