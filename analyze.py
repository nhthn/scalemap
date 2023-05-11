import json
import logging

import numpy as np
import sklearn.preprocessing
import umap

import music_theory

logger = logging.getLogger("analyze")


def analyze(
        edo=31,
        scale_size=7,
        min_step_12edo=0.5,
        max_step_12edo=3.5,
        max_scales=8_000,
        consonance_quantile=0.9,
        scale_weight=5.0,
        interval_vector_weight=5.0,
        consonance_weight=5.0,
        balance_weight=1.0,
        smallest_third_weight=1.0,
        neighbors=20,
):

    max_scale_classes = max_scales // edo
    min_step = int(np.ceil(min_step_12edo * edo / 12))
    max_step = int(np.floor(max_step_12edo * edo / 12))

    consonance_table = music_theory.ConsonanceModel().make_table(edo)
    logger.debug("Generating scale classes...")
    scale_classes = music_theory.generate_scale_classes(edo, scale_size, min_step, max_step)
    logger.info(f"Generated {len(scale_classes)} scale classes.")
    new_size = min(int(len(scale_classes) * (1 - consonance_quantile)), max_scale_classes)
    if len(scale_classes) < max_scale_classes:
        logger.info(f"No filtering, as {len(scale_classes)} < max_scale_classes.")
    else:
        logger.debug("Filtering scale classes...")
        scale_classes = music_theory.select_best_scale_classes(scale_classes, edo, consonance_table, new_size)
        logger.info(f"Filtered down to {new_size} scale classes.")

    logger.debug("Computing scales...")
    binary_scales, scale_class_indices = music_theory.generate_scales(edo, scale_classes)
    logger.info(f"{len(binary_scales)} scales generated.")

    logger.debug("Computing features...")
    interval_vectors = np.array([music_theory.interval_vector(scale, edo) for scale in scale_classes])
    consonances = np.array([
        np.dot(vector, consonance_table) for vector in interval_vectors
    ])
    # Given a set of IV that sum to constants, the L2 norm of sqrt(IV) will sum to a constant.
    interval_vectors_l2 = np.sqrt(interval_vectors)
    # The actual L2 norms of each IV will become unity in a later step.

    balance = np.array([
        1 - np.abs(np.sum(np.exp(1j * 2 * np.pi * scale / edo))) / scale_size
        for scale in scale_classes
    ])
    smallest_thirds = np.array([music_theory.smallest_third(scale, edo) for scale in scale_classes])

    grouped_features = [
        {"data": interval_vectors_l2, "preprocessing": "l2_norm", "weight": interval_vector_weight},
        {"data": consonances, "weight": consonance_weight},
        {"data": balance, "weight": balance_weight},
        {"data": smallest_thirds, "weight": smallest_third_weight},
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
    # Dimensions: (scale_class, feature)
    scale_class_features = np.concatenate([group["data"] for group in grouped_features], axis=1)

    scales = [np.argwhere(binary_scale).squeeze().tolist() for binary_scale in binary_scales]

    scale_features = []
    for binary_scale, scale_class_index in zip(binary_scales, scale_class_indices):
        # Dividing by sqrt(edo) ensures that the L2-norm of the scale is 0.
        normalized_scale = binary_scale / np.sqrt(edo) * scale_weight
        row = np.concatenate([normalized_scale, scale_class_features[scale_class_index, :]])
        scale_features.append(row)
    scale_features = np.array(scale_features)
    np.savetxt("out.txt", scale_features)

    logger.debug("Running UMAP...")
    reducer = umap.UMAP(random_state=0, n_neighbors=neighbors)
    points = reducer.fit_transform(scale_features)

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

    return {
        "edo": edo,
        "scale_size": scale_size,
        "min_step": min_step,
        "max_step": max_step,
        "scales": scales,
        "scale_class_indices": scale_class_indices,
        "scale_classes": [x.tolist() for x in scale_classes],
        "consonances": consonances.tolist(),
        "normalized_consonances": normalized_consonances.tolist(),
        "points": points.tolist(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_obj is not logger:
            log_obj.disabled = True

    in_file_name = "in.json"
    try:
        with open(in_file_name) as file:
            in_json = json.load(file)
    except FileNotFoundError:
        in_json = {}

    out_json = analyze(**in_json)
    out_json["parameters"] = in_file_name
    out_file_name = "out.json"
    with open(out_file_name, "w") as file:
        json.dump(out_json, file, indent=4)
    logger.info(f"Results exported to {out_file_name}.")
