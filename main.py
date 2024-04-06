import sys
import cv2
import mediapipe
import numpy
import autopy
import pydirectinput as p1 

cap = cv2.VideoCapture(0) 

# Initializing mediapipe
initHand = mediapipe.solutions.hands  
# Object of mediapipe with "arguments for the hands module"
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mediapipe.solutions.drawing_utils  
# Object to draw the connections between each finger index
wScr, hScr = autopy.screen.size()  
# Outputs the high and width of the screen (1920 x 1080)
pX, pY = 0, 0  # Previous x and y location
cX, cY = 0, 0  # Current x and y location

def handLandmarks(colorImg):
    landmarkList = []  
    landmarkPositions = mainHand.process(colorImg)  
    landmarkCheck = landmarkPositions.multi_hand_landmarks 
    if landmarkCheck:  
        for hand in landmarkCheck:  
            for index, landmark in enumerate(
                    hand.landmark):  
                draw.draw_landmarks(img, hand,
                                    initHand.HAND_CONNECTIONS)  
                h, w, c = img.shape  
                centerX, centerY = int(landmark.x * w), int(
                    landmark.y * h)  
                landmarkList.append([index, centerX, centerY])  
    return landmarkList

def fingers(landmarks):
    fingerTips = []  
    tipIds = [4, 8, 12, 16, 20]  
    # Check if thumb is up 
    if landmarks[tipIds[0]][1] > landmarks[tipIds[0] - 1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)
    # Check if fingers are up except the thumb
    for id in range(1, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][2]:  
            fingerTips.append(1)
        else:
            fingerTips.append(0)
    return fingerTips


# Define smoothing factor
smoothing_factor = 0.6

while True:
    check, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmList = handLandmarks(imgRGB)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        finger = fingers(lmList)

        if finger[1] == 1 and finger[2] == 0 and finger[4] == 0:
            x3 = numpy.interp(x1, (75, 640 - 75), (0, wScr))
            y3 = numpy.interp(y1, (75, 480 - 75), (0, hScr))

            # Smooth mouse movement
            cX = pX + (x3 - pX) * smoothing_factor
            cY = pY + (y3 - pY) * smoothing_factor

            autopy.mouse.move(wScr - cX, cY)
            pX, pY = cX, cY

        if finger[1] == 0 and finger[0] == 1 : 
             p1.click(button = 'left')
        
        if sum(finger) == 5:
             p1.keyDown("right")
             p1.keyUp("left")
        
        elif sum(finger) == 0:
             p1.keyDown("left")
             p1.keyUp("right")
        elif finger[1] == 1 and finger[2] == 1 and finger[3] == 1:
             p1.press("space")
        elif finger[1]==1:
             p1.keyUp("right")
             p1.keyUp("left")

    else:
        # Lift up all key presses when no hands are detected
        p1.keyUp("right")
        p1.keyUp("left")

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

try:
    cap.release()
except:
    sys.exit()
cv2.destroyAllWindows()
