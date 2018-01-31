import cv2
import serial
import win32com.client


def scan():
    # scan for available ports. return a list of tuples (num, name)
    available = []
    for i in range(256):
        try:
            s = serial.Serial(i)
            print(s.portstr)
            available.append((i, s.portstr))
            s.close()
        except:
            pass
    return available


print "Found ports:"
for n, s in scan(): print "(%d) %s" % (n, s)


def display_current_serial_numbers():
    wmi = win32com.client.GetObject("winmgmts:")

    for usb in wmi.InstancesOf("Win32_USBHub"):
        print usb.PNPDeviceID


def retrieve_port_number(serial_number):
    pass


class Camera(cv2.VideoCapture):
    def __init__(self, serial_number):
        super(Camera, self).__init__(retrieve_port_number(serial_number))
        pass


if __name__ == '__main__':
    display_current_serial_numbers()
