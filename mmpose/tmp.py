import math

x1, y1 = 0,0
x2, y2 = 0,2
temp = math.atan2((y2 - y1), (x2 - x1))
print(temp)
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = 2,2
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = 2,0
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = 2,-2
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = 0,-2
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = -2,-2
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = -2,0
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)

x1, y1 = 0,0
x2, y2 = -2,2
temp = math.atan2((y2 - y1), (x2 - x1))
angle = temp / math.pi * 180
print(angle)
