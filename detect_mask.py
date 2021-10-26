from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
import argparse
import time
import os
import time
import winsound

# 전체 모듈(얼굴인식 - 마스크인식 - 큐알코드인식 - 온도체크(미완료) - 출입문개방(미완료))
def detect_mask_video():
  # 얼굴 인식 후 프레임 저장
  def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
      (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
      faces = np.array(faces, dtype="float32")
      preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

  # 안면 인식 모듈을 통한 얼굴 인식
  try: 
    ifMask = False
    while True:
      ap = argparse.ArgumentParser()
      ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
      ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
      ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
      args = vars(ap.parse_args())

      # 얼굴 인식 모듈 로딩
      print("[INFO] loading face detector model...")
      prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
      weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
      faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

      # 마스크 인식 모듈 로딩
      print("[INFO] loading face mask detector model...")
      maskNet = load_model(args["model"])

      # 인식을 위한 캠 작동
      print("[INFO] starting video stream...")
      vs = VideoStream(src=0).start()
      time.sleep(1)
      first_check = time.time()
      last_check = time.time()
      # 인식 중
      while True:
        # 캠 크기 설정 (700 * 400)
        frame = vs.read()
        frame = imutils.resize(frame, width=700, height=400)

        # 마스크 인식 여부 확인
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
          (startX, startY, endX, endY) = box
          (mask, withoutMask) = pred
          
          if mask > 0.995:
            label = "Mask"
            color = (0, 255, 0)
            last_check = time.time()
          else:
            label = "No Mask"
            color = (0, 0, 255)
            first_check = time.time()
          
          # 마스크 착용 여부를 표시할 Label 설정
          label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
          cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
          cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # 마스크 착용 여부를 프레임 위에 표시
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # q 버튼을 누르면 종료
        if key == ord("q"): 
          break
        # 마스크 착용 확인이 2초이상 지속될 시에 착용 완료로 넘어감
        if last_check - first_check > 2:
          ifMask = True
          break
        # 마스크 미착용이 3초이상 지속될 시에 미착용으로 넘어감
        elif first_check - last_check > 3:
          ifMask = False
          break
        if ifMask is False:
          winsound.PlaySound("withoutMask.wav", winsound.SND_ASYNC)
        else:
          # 큐알코드 모듈 
          winsound.PlaySound("withMask.wav", winsound.SND_ASYNC) # TODO: 큐알코드 체크 알림음 변경
          import pyzbar.pyzbar as pyzbar
          cap = cv2.VideoCapture(0)

          try: 
            f = open('qrcode_data.txt', 'r', encoding='utf8')
            data_list = f.readlines()
          except FileNotFoundError:
            pass

          isQrCheck = False
          check_code_time = time.time()
          while isQrCheck is False:
            success, frame = cap.read()
            
            if success:
              cv2.imshow('cam', frame) 
              for code in pyzbar.decode(frame):
                my_code = code.data.decode('utf-8')
                print('인식 성공', my_code)
                winsound.PlaySound('./qrbarcode_beep.mp3', winsound.SND_FILENAME)
                for i in data_list:
                  if my_code == i.rsplit('\n')[0]:
                    cap.release()
                    cv2.destroyAllWindows()
                    winsound.PlaySound("withMask.wav", winsound.SND_ASYNC)
                    isQrCheck = True
                if isQrCheck == False and (time.time() - check_code_time) > 10:
                  isQrCheck = True
                  # TODO: 등록된 회원 정보가 아닐 때 break 후 등록된 회원정보가 아니라는 음성
                  # 시간이 어느정도 흘렀을 때 큐알코드가 보이지 않으면 자동으로 다시시작
              cv2.putText(frame, "QR Check", (300, 125 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
              cv2.rectangle(frame, (225, 125), (425, 325), (0, 255, 0), 2)    
              cv2.imshow('cam', frame) 
              if success:
                cv2.imshow('cam', frame)
                key = cv2.waitKey(1)
                if key == 27:
                  break

          cap.release()
  except :
    pass

while True:
  detect_mask_video()