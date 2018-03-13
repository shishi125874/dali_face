import urllib
import urllib2
import base64


def send_message(video, faceName, image_output_name, ip):
    # f = open(image_output_name, 'rb')
    # ls_f = base64.b64encode(f.read())
    print('sending message to %s' % ip)
    data = urllib.urlencode(
        {"videoName": video, "faceName": faceName, 'cappFace': image_output_name})
    print(video, faceName)
    path_data = "http://" + ip + ":8080/dali/video/face"
    request = urllib2.Request(path_data, data)
    response = urllib2.urlopen(request)
    file = response.read()
    if response.code != 200:
        return 'error code' + response.code
    else:
        return file
    print("send over")


if __name__ == '__main__':
    video = '/resources/b73465bcd0cb4597835b0a03837fcd32/823153@20180102132914@20180102132723_HDA00N_0001.mp4'
    faceName = 'mfc'
    f = open('/home/images/mfc.jpg', 'rb')
    ls_f = base64.b64encode(f.read())
    print type(ls_f)
    item = []
    item.append(ls_f)
    ip = '192.168.1.111'
    send_message(video, faceName, item, ip)
