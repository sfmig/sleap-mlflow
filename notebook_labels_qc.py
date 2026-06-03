# %%
import sleap_io as sio
from sleap.qc import LabelQCDetector, QCConfig

# %%
config = QCConfig(
    instance_threshold=0.7,      # Score threshold for flagging
    gmm_n_components=3,          # Number of GMM components
    duplicate_iou_threshold=0.5, # IoU threshold for duplicate detection
)



# %%
labels_file = "/home/sminano/swc/project_sleap_dome/labels_Kostas/dome_1male_v6.slp"

# %%
# Load labels
labels = sio.load_file(labels_file)

# Workaround for a sleap.qc bug: single-image videos store `filename` as a
# 1-element list, which the InstanceCountChecker later uses as an (unhashable)
# dict key. Unwrap them to plain strings so the id is hashable. Only mutates
# the in-memory labels; don't save these labels expecting list filenames back.
for _video in labels.videos:
    if isinstance(_video.filename, list) and len(_video.filename) == 1:
        _video.filename = _video.filename[0]

# %%%%%%%%%%%%%%%%%%
# SLEAP LABEL QC

# Create detector with default config
detector = LabelQCDetector(config=config)

# Fit on labels (learns what "normal" looks like from your data)
detector.fit(labels)

# Score all instances
results = detector.score(labels)

# Get flagged instances above threshold (0.0-1.0, higher = more anomalous)
flagged = results.get_flagged()

# Inspect flagged instances
for flag in flagged:
    print(f"Video {flag.video_idx}, Frame {flag.frame_idx}, Instance {flag.instance_idx}")
    print(f"  Score: {flag.score:.2f}")
    print(f"  Issue: {flag.top_issue}")
# %%
