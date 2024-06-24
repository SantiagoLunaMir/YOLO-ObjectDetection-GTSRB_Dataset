## Using YOLO Structure for Object Detection
Using the 43 classes from the dataset GTSRB, I trained the YOLO model, the issue is that there were 11 epochs, I will try to do more epochs.

### Confusion Matrix

![confusion_matrix_normalized](https://github.com/SantiagoLunaMir/YOLO-ObjectDetection-GTSRB_Dataset/assets/111355326/81bd1d20-e887-4836-83a1-57bf67acae0d)

As we could see in the image there are only two classes that are well representated, the sugestions is give to the model more epochs (121) will be my decision. But take notes from the posible overfiting.

### Examples from Detections

These are some examples from the batcht that the model produces:

![val_batch2_labels](https://github.com/SantiagoLunaMir/YOLO-ObjectDetection-GTSRB_Dataset/assets/111355326/3334b034-2fd4-4657-bab8-109d96f20e3e)

### Do you want to use the model?

The best model is in runs/detect/train10/weights/best.pt
At the same time if you want to use the the images and labels to train your own model, are added in this repository in zip formaat, unzip them for your personal use.
is suggest do a data aumentation, because there are only 11 images per class.

#### Load the best model.
```python
from ultralytics import YOLO
bestModel = YOLO("runs/detect/train10/weights/best.pt")
