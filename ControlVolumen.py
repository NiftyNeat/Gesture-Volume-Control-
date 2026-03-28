import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from comtypes.client import CreateObject
import comtypes
print("Imports basicos OK")

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator, EDataFlow, ERole
print("Pycaw OK")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
print("MediaPipe OK")

# ====================== CONTROL DE VOLUMEN ======================
comtypes.CoInitialize()

enumerator = CreateObject(
    "{BCDE0395-E52F-467C-8E3D-C4579291692E}",
    interface=IMMDeviceEnumerator
)
endpoint = enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender.value, ERole.eMultimedia.value)
interface = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
print("Volumen OK")

# ====================== CONFIGURACIÓN DE CÁMARA ======================
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
print("Camara OK")

# ====================== MEDIA PIPE TASKS - HAND LANDMARKER ======================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)
print("Detector OK - Abriendo camara...")

while cam.isOpened():
    success, image = cam.read()
    if not success:
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks_list in detection_result.hand_landmarks:
            lmList = []
            for id, landmark in enumerate(hand_landmarks_list):
                h, w, _ = image.shape
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) > 8:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                cv2.circle(image, (x1, y1), 12, (255, 255, 255), cv2.FILLED)
                cv2.circle(image, (x2, y2), 12, (255, 255, 255), cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                length = math.hypot(x2 - x1, y2 - y1)

                if length < 50:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                vol = np.interp(length, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                volBar = np.interp(length, [50, 220], [400, 150])
                volPer = np.interp(length, [50, 220], [0, 100])

                cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(image, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 255, 255), 3)

    cv2.imshow('Control de Volumen con Mano', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
