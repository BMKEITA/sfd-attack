import os
import shutil

val_dir = "data/imagenet/val"
gt_file = "ILSVRC2012_validation_ground_truth.txt"
wnids_file = "imagenet_class_index.txt"

# Load class ids
with open(wnids_file) as f:
    wnids = [line.strip() for line in f]

# Load labels
with open(gt_file) as f:
    labels = [int(x.strip()) for x in f]

images = sorted(os.listdir(val_dir))

for img, label in zip(images, labels):
    class_id = wnids[label - 1]
    class_dir = os.path.join(val_dir, class_id)

    os.makedirs(class_dir, exist_ok=True)
    shutil.move(os.path.join(val_dir, img), os.path.join(class_dir, img))