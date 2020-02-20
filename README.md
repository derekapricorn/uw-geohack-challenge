# uw-geohack-challenge

## Project Description
Produce an automated method to mask out __trees__ in high resolution drone imagery. To do so, I create an algorithm to segment trees in overhead imagery and I demonstrated this in the ipynb files contained in this repo.

I also used the algorithm to produce the mask output for the eval data. The mask file name is __eval_data_mask.png__. The output data is in raster png format, and contain 0 values for the ‘background’ class and 255 values for ‘tree’ class.

## Method
Because the original trainig and eval images are of very large size (5160x7068), directly loading the image without significant downsampling will cause memory issues. However, aggresive downsampling will cause loss of image details. As such, it was deemed reasonable to divide the original images into smaller patches (of size 672×448), segment them, and then later stitch the segmentation results together. 

Another issue to overcome is the lack of ground truth. After spliting the original training image into patches, I used __hasty.ai__ to assist manual labelling. Specifically I only labelled patches with bright pixels and did not label patches that were entirely blank which are the trivial case. 

After labelling the training images, I trained a u-net with a resnet-32 backbone which was pretrained on imagenet data. I used fastai and pytorch libraries to do so.

I then validated the model by segmenting the eval patches. The segmentation results were resampled to the input patch size. Lastly, I stitched up the mask patches into the final result file.

## Files in the repo
- preprocess.ipynb: functions used to split image into patches
- training.ipynb: main training script
- postprocess.ipynb: resample and merge prediction mask patches into the final result
- utils.py: utility functions shared with other notebooks
- training_external_not_in_use.ipynb: this was sandbox code used to experiment with external data but they didn't yield good result.

## Future improvement
It was observed that the network made false positive prediction on the small bushes on the left of the eval image, presumably due to their resemblance of the true tree textures. To mitigate this issue, future work can be done synthesize training data by croping the bushes and randomly paste them to a generic grassland. 


