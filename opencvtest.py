import cv2
import urllib2
import urllib


def send_message(video, faceName):
    data = urllib.urlencode({"viedoName": video, "faceName": faceName})
    request = urllib2.Request("http://192.168.1.19:8080/dali/vedio/face", data)
    response = urllib2.urlopen(request)
    file = response.read()
    if response.code != 200:
        return 'error code'+response.code
    else:
        return file
