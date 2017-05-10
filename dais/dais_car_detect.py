from dais_detector import DaisDetector
import os

model_path = "./VGGnet_fast_rcnn_iter_70000.ckpt"
img_path = os.path.join("..","data","demo","000456.jpg")

print(img_path)
print(os.path.exists(img_path))

dais_detctor = DaisDetector(model_path)

result = dais_detctor.detect(img_path)

print(result)