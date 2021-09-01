from PIL import Image
import qrcode

depart = input('학과: ')
stuNum = input('학번: ')
name = input('이름: ')
phoneNum = input('전화번호: ')

info = [depart, stuNum, name, phoneNum]
img = qrcode.make(",".join(info))
img.save("{}qr.png".format(stuNum[4:]))