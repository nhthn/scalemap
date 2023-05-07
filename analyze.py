import fractions
import json
import itertools
import logging

import numpy as np
import sklearn.preprocessing
import umap

logger = logging.getLogger("analyze")

primes = [2, 3, 5, 7, 11, 13]


def generate_all_partitions(total, steps, min_size, max_size):
    if steps <= 0 or total < min_size:
        return
    if steps == 1:
        if min_size <= total <= max_size:
            yield [total]
        return
    for step in range(min_size, max_size + 1):
        for partition in generate_all_partitions(total - step, steps - 1, min_size, max_size):
            yield [step] + partition


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


def weight(fraction):
    n, d = fraction.numerator, fraction.denominator
    return 1 / (np.sqrt(n * d) * highest_prime_factor(n) * highest_prime_factor(d))


def max_interval(edo):
    return int(np.ceil(edo / 2 + 1))


class Consonance:

    def __init__(self):
        self.spread = 17.0
        fractions_ = {
            fractions.Fraction(n, d)
            for n in range(1, 100)
            for d in range(1, 100)
            if 1 < n / d < 2
        }
        cutoff = np.max([weight(f) for f in fractions_]) * 0.03
        fractions_ = {f for f in fractions_ if weight(f) > cutoff}
        ratios = np.array([float(x) for x in fractions_])
        self.cents = 1200 * np.log2(ratios)
        self.weights = np.array([weight(x) for x in fractions_])

    def calculate(self, cents):
        cents = np.abs(cents)
        diff = (cents - self.cents) / self.spread
        return np.sum(np.exp(-diff * diff) * self.weights)

    def make_table(self, edo):
        result = []
        for i in range(max_interval(edo) + 1):
            result.append(self.calculate(1200 * i / edo))
        return result


def generate_scale_classes(edo, scale_size, min_step, max_step):
    partition_classes = set()
    for partition in generate_all_partitions(edo, scale_size, min_step, max_step):
        normalized = tuple(normalize(partition))
        partition_classes.add(normalized)
    return [np.concatenate([[0], np.cumsum(x)[:-1]]) for x in partition_classes]


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


def select_best_scales(scales, edo, consonance_table, count):
    dissonances = [
        -np.dot(interval_vector(scale, edo), consonance_table) for scale in scales
    ]
    partition = np.argpartition(dissonances, count)
    np_scales = np.array(scales)
    return np_scales[partition[:count], :]


def analyze():
    edo = 72
    scale_size = 7
    min_step_12edo = 0.5
    max_step_12edo = 2.5
    max_scales = 8_000
    consonance_quantile = 0.9

    min_step = int(np.ceil(min_step_12edo * edo / 12))
    max_step = int(np.floor(max_step_12edo * edo / 12))

    consonance_table = Consonance().make_table(edo)
    logger.debug("Generating scale classes...")
    scales = generate_scale_classes(edo, scale_size, min_step, max_step)
    logger.info(f"Generated {len(scales)} scales.")
    new_size = min(int(len(scales) * (1 - consonance_quantile)), max_scales)
    if len(scales) < max_scales:
        logger.info(f"No filtering, as {len(scales)} < max_scales.")
    else:
        logger.debug("Filtering scale classes...")
        scales = select_best_scales(scales, edo, consonance_table, new_size)
        logger.info(f"Filtered down to {new_size} scales.")

    logger.debug("Computing features...")
    interval_vectors = np.array([interval_vector(scale, edo) for scale in scales])
    consonances = np.array([
        np.dot(vector, consonance_table) for vector in interval_vectors
    ])
    # Given a set of IV that sum to constants, the L2 norm of sqrt(IV) will sum to a constant.
    interval_vectors_l2 = np.sqrt(interval_vectors)
    # The actual L2 norms of each IV will become unity in a later step.

    balance = np.array([
        1 - np.abs(np.sum(np.exp(1j * 2 * np.pi * scale / edo))) / scale_size
        for scale in scales
    ])
    smallest_thirds = np.array([smallest_third(scale, edo) for scale in scales])

    grouped_features = [
        {"data": interval_vectors_l2, "preprocessing": "l2_norm"},
        {"data": consonances, "weight": 6.0},
        {"data": balance},
        {"data": smallest_thirds},
    ]
    for group in grouped_features:
        data = group["data"]
        if data.ndim == 1:
            data = data[:, None]
        preprocessing = group.get("preprocess", "standard_scaler")
        if preprocessing == "standard_scaler":
            scaler = sklearn.preprocessing.StandardScaler()
            data = scaler.fit_transform(data)
        elif preprocessing == "l2_norm":
            data = sklearn.preprocessing.normalize(data, norm="l2")
        else:
            raise ValueError("Incorrect preprocessing step")
        data = data * group.get("weight", 1.0)
        group["data"] = data
    features = np.concatenate([group["data"] for group in grouped_features], axis=1)

    n_neighbors = int(len(scales) ** 0.5 / 2)
    logger.debug("Running UMAP...")
    reducer = umap.UMAP(random_state=0, n_neighbors=n_neighbors)
    points = reducer.fit_transform(features)

    # Nonlinearly map the consonances to a uniform-ish distribution for better visualization.
    # Experimentally I have found that consonances have a bell-shaped distribution, but musically
    # only the far outliers are all that interesting.
    quantile_transformer = sklearn.preprocessing.QuantileTransformer(
        random_state=0,
        n_quantiles=10,
    )
    normalized_consonances = np.squeeze(
        quantile_transformer.fit_transform(consonances[:, np.newaxis])
    )

    logger.debug("Exporting data...")
    json_export = {
        "edo": edo,
        "scale_size": scale_size,
        "min_step": min_step,
        "max_step": max_step,
        "scales": [x.tolist() for x in scales],
        "consonances": consonances.tolist(),
        "normalized_consonances": normalized_consonances.tolist(),
        "points": points.tolist(),
    }
    out_file_name = "out.json"
    with open(out_file_name, "w") as file:
        json.dump(json_export, file, indent=4)
    logger.info(f"Data exported to {out_file_name}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_obj is not logger:
            log_obj.disabled = True
    analyze()
