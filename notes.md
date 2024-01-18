###What technique did I choose and why
The guideline provides 3 face detection methods: SSD, RetinaFace, and InsightFace. In order to decide what method to use, I analyzed the distributions and sizes of faces in the provided validation image set. It turns out that the number of faces in each image varies in a wide range, from 0 face to 24 faces. Also, the face sizes also vary from 24x24 to 200x200 pixels. This means we need a method to handle dense face detection with wide range of scales.

From the relative literatures, all SSD, RetinaFace, and InsightFace are able to handle multi-scale dense object detection. However, RetinaFace uses a technique Feature Pyramid Network (FPN) which is better than the technique used in SSD. InsightFace also used FPN, but its model was designed for face recognition. This means under the same model complexity, InsightFace needs to learn both face detection and recognition tasks while RetinaFace just need to focus on learning the face detection task, which is more suitable for this project.

As a result, I choose RetinaFace for this project. Since the Github link for RetinaFace is not valid anymore in the provided guideline, I searched another for RetinaFace: https://github.com/biubug6/Pytorch_Retinaface

###Implementation details
Here are some implementation details I made:
1. Modularize everything into functions and imports
2. Utilized "argparse" for argument parsing
3. Removed every unecessary imports and files from the original Github clone what are not relevant to this project
4. Create a dataclass to define model settings
    `class ModelSettings:`
5. Create a metric collector class to collect and calculate metrics
    `class MetricCollector:`

###Metric
In the provided validation dataset, although we do not know what are the exact locations of faces, we know how many faces are there. If we assume there is no location discrepency from our predictions to ground truths, for each image, we can obtain the number of true positives, false positives, and false negatives based on the number of predicted faces and ground truth faces. Then, we could calculate the overall precision, recall, and F1-score.

Some of images contain no face but some of them contain many faces. Since the overall precision, recall, and F1-score might be biased to the images with many faces, I also calculated two per-image metrics: false positive per image and false negative per image. This two metrics give us ideas about how many wrong detections (false positive) and miss detections (false negative) could happen in average for each image.