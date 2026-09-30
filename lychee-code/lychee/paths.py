"""Where the rendered data live.

``DATA`` (environment variable ``LYCHEE_DATA``, default ``D:/lychee_data``) holds the rendered scenes, training sets and checkpoints of the paper.  The table scripts need only the
``meta.npz`` of the evaluation sets (fruit tables, commands, target masks; no images); these ship with the repository under ``data_meta/`` so that every table of the paper regenerates
without the rendered scenes (level 1 of REPRODUCE.md).  Scripts that need images (rendering, training, evaluation of networks) take explicit paths.
"""
import os

DATA = os.environ.get("LYCHEE_DATA", "D:/lychee_data")
SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_meta")          # lychee-code/data_meta


def eval_meta(name: str) -> str:
    """Path of ``meta.npz`` of the evaluation set ``name`` (for example ``iid_600``): the rendered data if present, else the copy that ships with the repository."""
    tried = []
    for base in (f"{DATA}/select_eval", f"{SHIPPED}/select_eval"):
        p = f"{base}/{name}/meta.npz"
        if os.path.exists(p):
            return p
        tried.append(p)
    raise FileNotFoundError("meta.npz of evaluation set %r not found; tried %s" % (name, tried))
