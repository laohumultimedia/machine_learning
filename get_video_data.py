import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')
frames = []

print "[+] Loading Video"
index = 0
while cap.isOpened():
	ret, frame = cap.read()
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frames.append(frame)
	pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
	index += 1
	print str(pos_frame)+" frames"
	if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
		# If the number of captured frames is equal to the total number of frames,
		# we stop
		break

print('read :%i frames' % (index))
print "[+] Writing to file"
np.save("video.npy", frames)
