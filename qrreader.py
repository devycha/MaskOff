import cv2
import pyzbar.pyzbar as pyzbar
import winsound

cap = cv2.VideoCapture(0)

try: 
  f = open('qrcode_data.txt', 'r', encoding='utf8')
  data_list = f.readlines()
except FileNotFoundError:
  pass


while True:
  success, frame = cap.read()
  
  if success:
    for code in pyzbar.decode(frame):
      my_code = code.data.decode('utf-8')
      print('인식 성공', my_code)
      winsound.PlaySound('./qrbarcode_beep.mp3', winsound.SND_FILENAME)
      for i in data_list:
        if my_code == i.rsplit('\n')[0]:
          print('환영합니다')
          break
        
    cv2.imshow('cam', frame)
    if success:
      cv2.imshow('cam', frame)
      
      key = cv2.waitKey(1)
      if key == 27:
        break

cap.release()
cv2.destroyAllWindows()