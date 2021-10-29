import RPi.GPIO as GPIO
from time import sleep

def move():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(12, GPIO.OUT)
    GPIO.setwarnings(False)
    p = GPIO.PWM(12, 50)

    p.start(0)
    p.ChangeDutyCycle(9)
    sleep(2)
    p.ChangeDutyCycle(6)
    sleep(2)