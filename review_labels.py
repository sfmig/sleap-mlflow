"""Quality-control checks for a SLEAP labels file, using sleap-io.

Runs a set of diagnostics over a .slp / .pkg.slp file and prints a report
flagging things that are usually labeling mistakes:

  * frames with more instances than expected  (drives "Max animals = N")
  * duplicate / overlapping instances on the same animal
  * empty instances        (no visible nodes)
  * incomplete instances   (some nodes missing)
  * points outside the image bounds
  * per-node visibility     (nodes that are rarely labeled)

Each check is a standalone function taking a loaded `Labels` object and
returning a list of records, so they can be reused independently.
"""

import argparse

import numpy as np
import sleap_io as sio


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def source_name(video):
    """Original image/video filename behind an (embedded) video, if any."""
    src = video.source_video or video
    fn = src.filename
    return fn[0] if isinstance(fn, list) else fn


def frame_location(labels, lf):
    """A compact identifier for a labeled frame."""
    return {
        "video": labels.videos.index(lf.video),
        "frame_idx": lf.frame_idx,
        "source": source_name(lf.video),
    }


def video_hw(video):
    """(height, width) of a video, or None if unknown (e.g. unopened externals)."""
    shape = video.shape or video.backend_metadata.get("shape")
    if shape is None:
        return None
    return int(shape[1]), int(shape[2])


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def multi_instance_frames(labels, max_instances_expected=1):
    """Frames with more user instances than expected (sets SLEAP's max animals)."""
    out = []
    for lf in labels:
        n_user = len(lf.user_instances)
        if n_user > max_instances_expected:
            out.append(
                {
                    **frame_location(labels, lf),
                    "n_instances": len(lf.instances),
                    "n_user": n_user,
                    "n_pred": len(lf.predicted_instances),
                }
            )
    return out


def duplicate_instance_pairs(labels, iou_threshold=0.5):
    """Pairs of instances in the same frame whose bounding boxes overlap.

    A high-overlap pair on single-animal data is almost always the same animal
    labeled twice (an accidental double-click), not a genuine second animal.
    Also reports the mean distance between matching nodes as a magnitude.
    """
    out = []
    for lf in labels:
        insts = lf.user_instances
        for i in range(len(insts)):
            for j in range(i + 1, len(insts)):
                if not insts[i].overlaps_with(insts[j], iou_threshold):
                    continue
                a, b = insts[i].numpy(), insts[j].numpy()
                shared = ~np.isnan(a).any(axis=1) & ~np.isnan(b).any(axis=1)
                mean_dist = (
                    float(np.linalg.norm(a[shared] - b[shared], axis=1).mean())
                    if shared.any()
                    else float("nan")
                )
                out.append(
                    {
                        **frame_location(labels, lf),
                        "instances": (i, j),
                        "mean_node_dist_px": round(mean_dist, 2),
                    }
                )
    return out


def empty_instances(labels):
    """Instances with no visible nodes at all (placeholder / stray instances)."""
    out = []
    for lf in labels:
        for k, inst in enumerate(lf.user_instances):
            if inst.n_visible == 0:
                out.append({**frame_location(labels, lf), "instance": k})
    return out


def incomplete_instances(labels):
    """Instances missing some (but not all) nodes."""
    n_nodes = len(labels.skeleton.nodes)
    out = []
    for lf in labels:
        for k, inst in enumerate(lf.user_instances):
            if 0 < inst.n_visible < n_nodes:
                out.append(
                    {
                        **frame_location(labels, lf),
                        "instance": k,
                        "n_visible": inst.n_visible,
                        "n_missing": n_nodes - inst.n_visible,
                    }
                )
    return out


def out_of_bounds_instances(labels):
    """Instances with at least one visible point outside the image bounds.

    Frames whose video dimensions are unknown (external videos that were not
    opened) are skipped; their count is reported under the "_skipped" key.
    """
    out = []
    for lf in labels:
        hw = video_hw(lf.video)
        if hw is None:
            continue
        height, width = hw
        for k, inst in enumerate(lf.user_instances):
            pts = inst.numpy()
            vis = ~np.isnan(pts).any(axis=1)
            xy = pts[vis]
            oob = (
                (xy[:, 0] < 0)
                | (xy[:, 0] >= width)
                | (xy[:, 1] < 0)
                | (xy[:, 1] >= height)
            )
            if oob.any():
                out.append(
                    {
                        **frame_location(labels, lf),
                        "instance": k,
                        "n_oob_points": int(oob.sum()),
                    }
                )
    return out


def instances_with_unknown_size(labels):
    """Count user instances whose video dimensions are unknown (unopened externals)."""
    return sum(len(lf.user_instances) for lf in labels if video_hw(lf.video) is None)


def node_visibility(labels):
    """Per-node count of how often each node is visible vs missing."""
    nodes = [n.name for n in labels.skeleton.nodes]
    visible = np.zeros(len(nodes), dtype=int)
    total = 0
    for lf in labels:
        for inst in lf.user_instances:
            visible += ~np.isnan(inst.numpy()).any(axis=1)
            total += 1
    return [
        {"node": name, "visible": int(v), "missing": total - int(v), "total": total}
        for name, v in zip(nodes, visible)
    ]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def print_rows(rows, fmt, limit=None):
    shown = rows if limit is None else rows[:limit]
    for r in shown:
        print("  " + fmt.format(**r))
    if limit is not None and len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more")


def run_report(slp_file, max_instances_expected=1, iou_threshold=0.5, limit=20):
    labels = sio.load_slp(slp_file, open_videos=False)

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"  input file:         {slp_file}")
    print(f"  videos:             {len(labels.videos)}")
    print(f"  labeled frames:     {len(labels)}")
    print(f"  user instances:     {labels.n_user_instances}")
    print(f"  predicted instances:{labels.n_pred_instances}")
    print(f"  tracks:             {len(labels.tracks)}")
    print(f"  suggestions:        {len(labels.suggestions)}")
    sk = labels.skeleton
    print(
        f"  skeleton:           {sk.name} "
        f"({len(sk.nodes)} nodes, {len(sk.edges)} edges)"
    )

    checks = [
        (
            f"FRAMES WITH > {max_instances_expected} USER INSTANCE(S)",
            lambda: multi_instance_frames(labels, max_instances_expected),
            "video {video:>4}  frame {frame_idx:>5}  "
            "n={n_instances} (user={n_user}, pred={n_pred})  {source}",
        ),
        (
            f"DUPLICATE / OVERLAPPING USER INSTANCES (IoU > {iou_threshold})",
            lambda: duplicate_instance_pairs(labels, iou_threshold),
            "video {video:>4}  frame {frame_idx:>5}  "
            "user instances {instances}  mean_node_dist={mean_node_dist_px}px  {source}",
        ),
        (
            "EMPTY USER INSTANCES (no visible nodes)",
            lambda: empty_instances(labels),
            "video {video:>4}  frame {frame_idx:>5}  user instance {instance}  {source}",
        ),
        (
            "INCOMPLETE USER INSTANCES (some nodes missing)",
            lambda: incomplete_instances(labels),
            "video {video:>4}  frame {frame_idx:>5}  user instance {instance}  "
            "visible={n_visible} missing={n_missing}  {source}",
        ),
        (
            "POINTS OUTSIDE IMAGE BOUNDS (user instances)",
            lambda: out_of_bounds_instances(labels),
            "video {video:>4}  frame {frame_idx:>5}  user instance {instance}  "
            "oob_points={n_oob_points}  {source}",
        ),
    ]

    for title, check, fmt in checks:
        print()
        print("=" * 70)
        rows = check()
        print(f"{title}: {len(rows)}")
        print("=" * 70)

        print_rows(rows, fmt, limit)
        if title.startswith("POINTS OUTSIDE"):
            n_unknown = instances_with_unknown_size(labels)
            if n_unknown:
                print(
                    f"  (note: {n_unknown} user instance(s) not checked — video "
                    "size unknown; load with open_videos=True or use a .pkg.slp)"
                )

    print()
    print("=" * 70)
    print("NODE VISIBILITY (across user instances)")
    print("=" * 70)
    for r in node_visibility(labels):
        pct = 100 * r["visible"] / r["total"] if r["total"] else 0
        print(
            f"  {r['node']:<14} visible {r['visible']:>5}/{r['total']:<5} "
            f"({pct:5.1f}%)  missing {r['missing']}"
        )


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "slp_path",
        help="Path to the .slp / .pkg.slp labels file.",
    )
    parser.add_argument(
        "-n",
        "--max-instances-expected",
        type=int,
        default=1,
        help="Frames with more than this many instances are flagged "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Bounding-box IoU above which two instances are called duplicates "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max rows to print per check; 0 or less means no limit "
        "(default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # open_videos=False: we only need the labels, not the pixels, so skip
    # probing the (possibly missing) video backends and the warnings it emits.

    run_report(
        args.slp_path,
        max_instances_expected=args.max_instances_expected,
        iou_threshold=args.iou_threshold,
        limit=args.limit if args.limit > 0 else None,
    )
