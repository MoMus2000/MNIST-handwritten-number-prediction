from tensorflow import keras
model = keras.models.load_model('/Users/a./Desktop/course/mnist Model')

# import numpy as np
from skimage import transform
# def load(filename):
#    np_image = Image.open(filename)
#    np_image = np.array(np_image).astype('float32')/255
#    np_image = transform.resize(np_image, (28, 28, 1))
#    np_image = np.expand_dims(np_image, axis=0)
#    return np_image
import cv2
import numpy as np

drawing = False

def draw(event , x, y ,flags , params):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img1,(x,y),10,(255,0,0),-1)

img1 = np.zeros((512,512,1),np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image",draw)

while(1):
    cv2.imshow("image",img1)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        break
    elif k ==  ord('p'):
        np_image = np.array(img1).astype('float32')/255.0
        np_image = transform.resize(np_image, (28, 28, 1))
        np_image = np.expand_dims(np_image, axis=0)
        pr = model.predict_classes(np_image)
        print("Predicted as ")
        print(pr[0])
    elif k == ord('c'):
        img1 = np.zeros((512,512,1),np.uint8)

cv2.destroyAllWindows()
# image = load('/Users/a./Desktop/1.jpg')
# print(model.predict_classes(image))
