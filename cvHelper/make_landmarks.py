landmarks = []
for i in range(2172):
    index = i // 4 + 1
    if i % 4 == 0:
        landmarks.append('x' + str(index))
    elif i % 4 == 1:
        landmarks.append('y' + str(index))
    elif i % 4 == 2:
        landmarks.append('z' + str(index))
    else:
        landmarks.append('v' + str(index))

print(landmarks)