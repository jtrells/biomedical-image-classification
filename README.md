# Active Learning

## File Structure

Parquet files and trained models are kept on /mnt/vil-al-interface, the directory has the following structure:

```
vil-al-interface
-- files
    -- <tax-name> (e.g. cord19)
        -- <tax-name_classifier_version.parquet>
        -- cord19_gel_v1.parquet
        -- cord19_radiology_v1.parquet
        ...
        -- all.parquet (special file)
-- models
    -- <tax-name> (e.g. cord19)
        -- <classifier> (e.g gel)
            -- <classifier_version.pt>
            -- gel_1.pt
```

## Taxonomy

To reflect the hierarchical structure, the classes have the format parent_class.child_class. For higher-modality it is:
exp, mic, rad, pho, oth, mol and gra.
For a sub-classifier like gel: exp.gel, exp.pla.

We are using a smaller string name, but we have a mapping dictionary for the front-end so users see the full name.

While some parquet files may classes like: mic.ele.sca, mic.ele.tra, mic.ele.oth, the training logic only consider classes with more than 100 samples (see logic on ImageDataSet, ImageDataModule and Trainers). To make it easier on the front-end, I'm storing the used classes on the database. tldr; let's be careful with applying the label encoder over the whole dataset directly.

## Workflow

0. Get a first version of the classifiers on the available data (Done => all versions 1 of data and classifiers are using the available labeled data).
1. Process a load of unlabeled data

   For any incoming collection of documents, create a CSV file with the following rows:

   - img -> unique image name
   - source -> 'cord-19' for the current unlabeled set
   - img_path -> cord19-uic/... (files are on /mnt but path can start from cord19-uic without a / at the beginning)
   - caption -> every image has a .txt file with the caption for the figure, that same caption should apply for every subfigure
   - width -> calculate with PIL
   - height -> calculate with PIL

2. Apply classifiers to every image to get pseudo-labels. An approach may be applying the high-modality classifier, then dividing each subset accordingly and applying the other classifiers. We would get the column 'label' but it may be worth it to add a column 'al' -> true to indicate the prediction came from active learning.
3. Divide the data into training, validation and test sets to complement existing datasets (column split_set: TRAIN, VAL, TEST).

   - For existing datasets, the split was done by starting from the more specific classifiers (e.g. mic.ele.tra). If an image was part of that group for TRAIN, then it also belongs to the TRAIN group for mic.ele. This design decision simplifies a bit the partition.
   - The partition was done using the stratify method from utils/datasets.py.
   - I'm not sure if we want to get data for TEST or persist the first TEST partition that we originally gather. I guess it makes sense to add data to TEST as the distribution may be different. However, we have to consider this for displaying later the performance metrics.

4. We have to save in some column whether that sample needs to labeling or not.

   - It may be worth keeping the values for the three type of strategies you identified or shall we use one? One should be fine for the prototype.

5. For the existing datasets, I used the wrapper.py extract_and_update_features method calculate the prediction and pred_probs columns.

   - For these cases, I guess the prediction column would be the same as the label column, or shall we leave the label column empty and use the prediction column for a pseudo label?
   - pred-probs has the probabilies per class based on the active classes used (label encoder sorts them, that's how I know which one belongs to which class).

6. Save the data as parquet
7. Update the existing dataset files -> 'al' column will differentiate the labeled data with this set. We need to version them accordingly using the file name.

Human interaction comes here where we label the images and then once done we send them back to training. Ideally, we should be able to track if the process is working or not, maybe we can save that information in the database so the front-end knows what's going on.

Steps for updating classifiers:

1. Gather the data from the images table in the database. Remove elements marked for delete, update elements based on the new label. This step produces an updated version of the parquet files -> update file version.

   - A possible scenario may create data that do not follow the split that we want for TRAIN/VAL/TEST but I believe we should not care about this edge case now.

2. Use wrapper train function to update classifiers. The method should version the weights file but double check that.

## Thumbnails

After finishing with everything above, we can focus on creating thumbnails to speed up network transmissions.
