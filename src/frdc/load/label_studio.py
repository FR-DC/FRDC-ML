from __future__ import annotations

import logging
from pathlib import Path
from warnings import warn

from label_studio_sdk.data_manager import Filters, Column, Type, Operator

from frdc.conf import LABEL_STUDIO_CLIENT

logger = logging.getLogger(__name__)


class Task(dict):
    def get_bounds_and_labels(self) -> tuple[list[tuple[int, int]], list[str]]:
        bounds = []
        labels = []

        # Each annotation is an entire image labelled by a single person.
        # By selecting the 0th annotation, we are usually selecting the main
        # annotation.
        annotation = self["annotations"][0]

        # There are some metadata in `annotation`, but we just want the results
        results = annotation["result"]

        for bbox_ix, bbox in enumerate(results):
            # 'id' = {str} 'jr4EXAKAV8'
            # 'type' = {str} 'polygonlabels'
            # 'value' = {dict: 3} {
            #       'closed': True,
            #       'points': [[x0, y0], [x1, y1], ... [xn, yn]],
            #       'polygonlabels': ['label']
            # }
            # 'origin' = {str} 'manual'
            # 'to_name' = {str} 'image'
            # 'from_name' = {str} 'label'
            # 'image_rotation' = {int} 0
            # 'original_width' = {int} 450
            # 'original_height' = {int} 600
            bbox: dict

            # See Issue FRML-78: Somehow some labels are actually just metadata
            if bbox["from_name"] != "label":
                continue

            # We flatten the value dict into the result dict
            v = bbox.pop("value")
            bbox = {**bbox, **v}

            # Points are in percentage, we need to convert them to pixels
            bbox["points"] = [
                (
                    int(x * bbox["original_width"] / 100),
                    int(y * bbox["original_height"] / 100),
                )
                for x, y in bbox["points"]
            ]

            # Only take the first label as this is not a multi-label task
            bbox["label"] = bbox.pop("polygonlabels")[0]
            if not bbox["closed"]:
                logger.warning(
                    f"Label for {bbox['label']} @ {bbox['points']} not closed. "
                    f"Skipping"
                )
                continue

            bounds.append(bbox["points"])
            labels.append(bbox["label"])

        return bounds, labels


def get_task(
    file_name: Path | str = "chestnut_nature_park/20201218/result.jpg",
    project_id: int = 1,
):
    proj = LABEL_STUDIO_CLIENT.get_project(project_id)
    task_ids = [
        task["id"]
        for task in proj.get_tasks()
        if file_name.as_posix() in task["storage_filename"]
    ]

    if len(task_ids) > 1:
        warn(f"More than 1 task found for {file_name}, using the first one")
    elif len(task_ids) == 0:
        raise ValueError(f"No task found for {file_name}")

    return Task(proj.get_task(task_ids[0]))
