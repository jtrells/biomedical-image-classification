# biomedical-image-classification
Where we evaluate different deep learning approaches for biomedical image classification.

## Microscopy Experiments
src/experiments

## Captions
### Dataset
For modality classification, we used the subfigures from the ImageCLEF 2013 and 2016 sub-figure classification task. We obtained the figure captions from the Compound Figure Detection task (currently only for the 2016 dataset). The caveat is that the captions match the compound figures; not further post-processing was done to match the sub-sentence to the subfigure.

While working on matching the subfigures to the captions, we noticed some error in the image names. For instance, some images did not follow the pattern *ImageName-{subfigure_number}*, while a couple use a *-* instead of a *.* in the figure name. Therefore, we corrected the subfigure names and the CSV files in our data folder reflect that. The changes were as follows:

ImageClef 2016 Test Set
* 1471-2105-6-S2-S11-7`_`X.jpg -> 1471-2105-6-S2-S11-7`-`X.jpg
* IJBI2010-308627`-`003-8.jpg -> IJBI2010-308627`.`003-8.jpg
* IJBI2010-535329`-X.015`.jpg -> IJBI2010-535329`.015-X`.jpg

We did not include the images or the bounding boxes in this repository. For access to the dataset, please contact the [ImageCLEFmed: The Medical Task 2016 organizers](https://www.imageclef.org/2016/medical).

```python

```
