import os

path = '/home/images'
count = 001
for file in os.listdir(path):
    os.rename(os.path.join(path, file), os.path.join(path, str('%03d'%count) + ".jpg"))
    count += 1

