import gzip
f = gzip.open('/home/suparna/work/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 5

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

import matplotlib.pyplot as plt
image = np.asarray(data[2]).squeeze()
plt.figure()
plt.imshow(image)
plt.show()


f = gzip.open('/home/suparna/work/train-labels-idx1-ubyte.gz','r')
f.read(8)
for i in range(0,50):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)
