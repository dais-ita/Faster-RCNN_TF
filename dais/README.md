# Dais Object Detector

To use the object detector, first instantiate an object
```Python
	dais_detctor = DaisDetector(model_path)
```

where `model_path` is the path of the pre-trained model weights.

Then call `detect` method passign the file name.
```Python
	result = dais_detctor.detect(img_path)
```

where `img_path` is expected to be under the `data` subdirectory.

The result is a dictionary with key is the object type, and the value is a list of tuples representing detected objects.
The first item of the tuple is the bounding box and the second item is the score.