import cv2
import time


def camera(file_name='image.jpg'):
    # open camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('test_camera')

    # if cap.isOpened():
    start = time.time()
    while True:
        print("Camera opened.")

        # set dimensions
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        print("Dimensions set.")

        # take frame
        ret, frame = cap.read()
        print("Frame taken.")
        #print("ret:", ret, "frame:", frame)

        # write frame to file
        cv2.imshow('test_camera', frame)
        #k=0
        #while k<1000000000:
            #k=k+1S
        #time.sleep(10)
        if time.time() - start > 10:
            #resizing frame before saving
            frame = cv2.resize(frame, (200, 200))
            cv2.imwrite(file_name, frame)
            print("Wrote the image.")
            break
        
        k = cv2.waitKey(1)
        if k%256==27:
            print("Escape hit.")
            break

    # release camera
    cap.release()
    cv2.destroyAllWindows()

    return Image.open(file_name)


# PREDICT IMAGE

def predict_image(model):
    # Convert to a batch of 1
    # xb = to_device(img.unsqueeze(0), device)
    image = camera()
    # Get predictions from model
    pred = model(image)
    # Pick index with highest probability
    _, preds  = torch.max(pred, dim=1)
    # Retrieve the class label
    mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    return mapping[preds[0].item()]