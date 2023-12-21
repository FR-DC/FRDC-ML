from frdc.load.label_studio import get_task
from utils import requires_label_studio


@requires_label_studio
def test_get_bounds_and_labels():
    task = get_task("DEBUG/0/result.jpg")
    bounds, labels = task.get_bounds_and_labels()
