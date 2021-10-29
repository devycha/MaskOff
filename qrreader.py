import pyzbar.pyzbar as pyzbar
import cv2

def qr_check(data_list):
  cap = cv2.VideoCapture(0)
  print("[INFO] qr check is ready")
  check = False
  while (check == False):
    success, frame = cap.read()  
    if success:
      cv2.imshow("Frame", frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      
      for code in pyzbar.decode(frame):
        my_code = code.data.decode('utf-8')
        print('인식 성공', my_code)
        for i in range(len(data_list)):
          if my_code == data_list[i].rsplit('\n')[0]:
            check = True
          if i == len(data_list)-1:
            return False
  
  return check

# print(qr_check(open('qrcode_data.txt', 'r', encoding='utf8').readlines()))