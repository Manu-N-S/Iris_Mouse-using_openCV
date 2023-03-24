import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pyautogui
scroll = False
cAct = ""
pAct = ""
sCot=0
rbCot=0
lbCot=0
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# load face detection model
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)

cap = cv.VideoCapture(0)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

cursor_speed = 10
screen_width, screen_height = pyautogui.size()

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return reRatio,leRatio 



with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
        img_h, img_w = frame.shape[:2]
        #################################
        results = mp_face.process(rgb_frame)

        if not results.detections:
            print('No faces detected.')
        else:
            for detection in results.detections: # iterate over each detection and draw on image
                mp_drawing.draw_detection(frame, detection)



        #####################################
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame,results,False)
            reRatio,leRatio = blinkRatio(frame,mesh_coords,RIGHT_EYE,LEFT_EYE)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            A= mesh_points[4]
            # cv.circle(frame,A, int(l_radius), (0, 255, 0), 1, cv.LINE_AA)

            if reRatio>5.5 and leRatio>5.5:
                cAct = "scroll"
                if(cAct == pAct):
                    sCot = sCot+1
                    if(sCot>=5):
                        sCot = 0
                        print("SCROLL SWITCH ")
                        if scroll:
                            print("-- OFF")
                            scroll = False
                        else:
                            print("-- ON")
                            scroll = True

            

            if not scroll:
                if reRatio >5.5 and leRatio<5.5:
                    cAct = "rblink"
                    if(cAct == pAct):
                        rbCot = rbCot+1
                        if(rbCot>3):
                            rbCot=0
                            print("RIGHT BLINK")
                            pyautogui.click(button='right')
                            cv.circle(frame,(240,320),5,(255,0,0),3)
                #CEF_COUNTER +=1
                     
                if leRatio >5.5 and reRatio<5.5:
                    cAct = "lblink"
                    if(cAct == pAct):
                        lbCot = lbCot+1
                        if(lbCot>3):
                            lbCot=0
                            print("LEFT BLINK")
                            pyautogui.click(button='left')
                            cv.circle(frame,(240,320),5,(0,0,255),3)  
            
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (255, 0, 0), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 0), 1, cv.LINE_AA)

            cursor_xo = int((center_left[0] + center_right[0]) / 2)
            cursor_yo = int((center_left[1] + center_right[1]) / 2)

            print(cursor_xo,cursor_yo)

            cursor_x = int(cursor_xo * screen_width / frame.shape[1])
            cursor_y = int(cursor_yo * screen_height / frame.shape[0])
            nose_y = int(A[1]*screen_height/frame.shape[0])

            if scroll:
                if cursor_yo < 210:
                    print("SCROLLING UP....")
                    pyautogui.scroll(40)
                if cursor_yo >210 and cursor_yo<223:
                    print("NO SCROLL.....")
                if cursor_yo > 223:
                    print("SCROLLING DOWN....")
                    pyautogui.scroll(-40)
            
            try :
                pyautogui.moveTo((cursor_x-280)*1.5, (cursor_y-350)*4, duration=0.1)
            except:
                print("OUT OF FRAME !!")
            pAct = cAct
        cv.imshow("img", frame)
        key = cv.waitKey(1)
        if key ==ord("q"):
            break
cap.release()
cv.destroyAllWindows()