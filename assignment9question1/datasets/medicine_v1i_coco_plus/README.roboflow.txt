
medicine - v1 2026-11-11 2:27pm
==============================

This dataset was exported via roboflow.com on November 11, 2026 at 8:58 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 449 images (447 original + 2 added).
Parol are annotated in COCO format.

The following pre-processing was applied to each image:

No image augmentation techniques were applied.



----------------------------------------------------------------------
medicine_v1i_coco_plus  (modified copy)
----------------------------------------------------------------------
Adds 2 Indian-market pack photos to the train split (1280 px wide, much higher
resolution than the original ~200-300 px Turkish images):
  daplin-trio_jpg.custom.jpg  -> class DAPLIN-TRIO  (Daplin Trio cream, 10 g)
  nor-co-q10_jpg.custom.jpg   -> class NOR-CO-Q10   (Nor Co-Q10 softgel capsules)
Boxes cover the whole pack, matching the original annotations. Extra text-level
ground truth for these two images is in extra_ground_truth.csv.
Dataset now has 449 images and 19 classes. Original data is licensed CC BY 4.0.
