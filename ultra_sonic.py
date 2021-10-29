import RPi.GPIO as GPIO
import time

#GPIO Mode (BOARD/BCM)
GPIO.setmode(GPIO.BCM)

#set GPIO pins
GPIO_TRIGGER=18
GPIO_ECHO=24

#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
GPIO.setup(GPIO_ECHO,GPIO.IN)

def distance():
    #set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER,True)
    
    #set Trigger after 0.01 ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER,False)
    
    StartTime=time.time()
    StopTime=time.time()
    
    #save starttime
    while GPIO.input(GPIO_ECHO)==0:
        StartTime=time.time()
        
    #save time of arrival
    while GPIO.input(GPIO_ECHO)==1:
        StopTime=time.time()
        
    #time difference between start and arrival
    TimeElapsed=StopTime-StartTime
    #multiplay with the sonic speed (34300 cm/s)
    #and divide by 2. because there amd back
    distance=(TimeElapsed*34300)/2
    
    return distance

# from gpiozero import DistanceSensor
# ultrasonic = DistanceSensor(echo=24, trigger=18)
# while True:
#     print(ultrasonic.distance)

while True:
    print(distance())
    time.sleep(0.1)