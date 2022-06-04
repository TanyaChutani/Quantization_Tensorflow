import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bbox(
    image: np.ndarray,
    scores: np.ndarray,
    classname: np.ndarray,
    coordinates: np.ndarray,
    min_score_threshold: float,
    normalised_coordiantes: bool,
):
    width, height, channels = image.shape

    if normalised_coordiantes:
        coordinates[:, 0] = (coordinates[:, 0] * width).astype(int)
        coordinates[:, 1] = (coordinates[:, 1] * height).astype(int)
        coordinates[:, 2] = (coordinates[:, 2] * width).astype(int)
        coordinates[:, 3] = (coordinates[:, 3] * height).astype(int)

    assert (
        len(scores) == len(classname) == len(coordinates)
    ), "Length of scores, classes and coordinates are not equal"

    fig, ax = plt.subplots()
    ax.imshow(image)

    for idx, i in enumerate(coordinates):
        if scores[idx] > min_score_threshold:
            rect = patches.Rectangle(
                (i[1], i[0]), i[3], i[2], edgecolor="c", linewidth=1, fill=False
            )
            plt.text(
                i[1],
                i[0],
                classname[idx],
                bbox=dict(fill=True, facecolor="c", edgecolor="c", linewidth=1),
            )
            ax.add_patch(rect)
    plt.show()

