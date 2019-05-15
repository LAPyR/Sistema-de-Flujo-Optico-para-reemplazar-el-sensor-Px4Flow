import numpy as np
import cv2 as cv
import cv2
import serial
import smbus
import time

#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B #CLKSEL
SMPLRT_DIV   = 0x19 #SMPLRT-DIV
CONFIG       = 0x1A #DLPF-CFG 1A
GYRO_CONFIG  = 0x1B #FS-SEL
INT_ENABLE   = 0x38 #DATA-RDY-EN
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47
arduino = serial.Serial('/dev/ttyACM0', baudrate = 9600, timeout = 1)
track_points = 0 #number of points in the actual frame
distpromx    = 0
distpromy    = 0
dpx          = 0
dpy          = 0
posx         = 0 ##x position
posy         = 0 #y position
valx         = 0
valy         = 0
bild         = 0
a            = 1
vb           = 1
Xpos         = 0
Ypos         = 0
memoriax     = 0
memoriay     = 0
Xdif         = 0
Xdif1        = 0
Ydif1        = 0
Ydif         = 0
m            = True
Movx         = 0
Movy         = 0
grad2rad     = 0.000266316
stime        = 0
ftime        = 0
ttime        = 0
radx         = 0
rady         = 0
radz         = 0
Fx           = 919.581738
Fy           = 920.767041
sumqual      = 0

def MPU_Init():
	#write to sample rate register
	bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)

	#Write to power management register
	bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

	#Write to Configuration register
	bus.write_byte_data(Device_Address, CONFIG, 5)

	#Write to Gyro configuration register
	bus.write_byte_data(Device_Address, GYRO_CONFIG,8)

	#Write to interrupt enable register
	bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
	#Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)

        #concatenate higher and lower value
        value = ((high << 8) | low)

        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value

bus = smbus.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address

MPU_Init()
################   1280*720

#cap=cv2.VideoCapture("vid77.avi")
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM),width=(int)680, height=(int)420,format=(string)I420,framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw,format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(old_gray)
old_gray = equ
#old_gray = cv.cvtColor(old_gray, cv.COLOR_BGR2GRAY)

p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
stime  = time.clock()
while(1):
    ftime=(time.clock())-stime
    stime  = time.clock()
    ttime+=ftime*1000000
	#Read Gyroscope raw valu
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    gyro_z = read_raw_data(GYRO_ZOUT_H)

    Gx     = gyro_x * grad2rad * ftime * 10000 #mrad
    Gy     = -gyro_y * grad2rad * ftime * 10000 #mrad
    Gz     = gyro_z * grad2rad * ftime * 10000 #mrad

    if abs(Gx) > 2:
       radx   += Gx
    if abs(Gy) > 2:
       rady   += Gy
    if abs(Gz) > 2:
       radz   += Gz

    ret,frame = cap.read()
    print (ret)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(frame_gray)
    frame_gray = equ
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    track_points = len(good_new)

    if track_points <= 10: ########
       old_gray = frame_gray
       p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
       mask = np.zeros_like(old_frame)
       dpx+=(-distpromx - Gy/Fx) * 15000 / Fx
       dpy+=(distpromy - Gx/Fy) * 15000 / Fy

    else:
       diferencia = p1 - p0
       Xpos=0
       Ypos=0
       for p in range(0,track_points-1):
          valx= diferencia[p]
          valx= valx[0]
          Xpos+= valx[0] ##first x value from the matrix
          Ypos+= valx[1] ##first y value from the matrix
       distpromx = Xpos / track_points
       distpromy = Ypos / track_points
       dpx+=(-distpromx - Gy/Fx) * 15000 / Fx
       dpy+=(distpromy - Gx/Fy) * 15000  / Fy


       for i,(new,old) in enumerate(zip(good_new,good_old)):
           a,b = new.ravel()
           c,d = old.ravel()
           mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)

           frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)


       #img = cv.add(frame,mask)

       #cv.imshow('Video',frame) ###################################
       # Now update the previous frame and previous points
       old_gray = frame_gray.copy()
       p0 = good_new.reshape(-1,1,2)
    bild+=1

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    sumqual += track_points * 5.1


    #bus.write_byte(0x42,0x16)
    #time.sleep(0.01)
    #gd = bus.read_i2c_block_data(0x42,0x14,2)

    comando = 0
    b_disp = arduino.inWaiting()

    if b_disp > 0:
      comando = arduino.read(1)
      # print comando
      arduino.reset_input_buffer()

    if comando == '1':
     print ('Enviando')
    # print "Optical", dpx
     if dpx <0:
      dpx=65536+dpx
     if dpy <0:
      dpy=65536+dpy

     if (radx < 0):
        radx = 65535+radx
     if (rady < 0):
        rady = 65535+rady
     if (radz < 0):
        radz = 65535+radz
     sumqual = 200
     qual = sumqual/(bild+1)
     qual = 200
     if qual > 255:
        qual = 255

     print ("########################")

     arduino.write(chr(bild & 0xFF))
     arduino.write(chr(bild >> 8))
     arduino.write(chr(int(dpx) & 0xFF))
     arduino.write(chr(int(dpx) >> 8))
     arduino.write(chr(int(dpy) & 0xFF))#5
     arduino.write(chr(int(dpy) >>8))
     arduino.write(chr(int(radx) & 0xFF))
     arduino.write(chr(int(radx) >> 8))
     arduino.write(chr(int(rady) & 0xFF))
     arduino.write(chr(int(rady) >> 8))#10
     arduino.write(chr(int(radz) & 0xFF))
     arduino.write(chr(int(radz) >> 8))
     arduino.write(chr(int(ttime) & 0xFF))
     arduino.write(chr((int(ttime)>>8) & 0xFF))
     arduino.write(chr((int(ttime)>>16) & 0xFF))#15
     arduino.write(chr(int(ttime)>>24))
     arduino.write(chr(int(qual)))
     bild=0
     dpx=0
     dpy=0
     ttime = 0
     radx = 0
     rady = 0
     radz = 0
     sumqual = 0

arduino.close()

k = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()
