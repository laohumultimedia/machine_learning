import numpy as np
import cv2

data = np.load('video-decoded.npy')

while True:
  for frame in data:
    frame = frame.reshape([180, 240, 3])
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cv2.destroyAllWindows()
