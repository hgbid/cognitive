import matplotlib.pyplot as plt
import pandas as pd
import matplotlib



def findClosest(arr, n, target):
    if (target <= arr[0]):
        return 0
    if (target >= arr[n - 1]):
        return n - 1
    i = 0
    j = n
    mid = 0
    while i < j:
        mid = (i + j) // 2
        if arr[mid] == target:
            return mid
        if (target < arr[mid]):
            if (mid > 0 and target > arr[mid - 1]):
                return getClosest(arr[mid - 1], arr[mid], mid - 1, mid, target)
            j = mid
        else:
            if (mid < n - 1 and target < arr[mid + 1]):
                return getClosest(arr[mid], arr[mid + 1], mid, mid + 1, target)
            i = mid + 1
    return mid


def getClosest(val1, val2, index1, index2, target):
    if target - val1 >= val2 - target:
        return index2
    else:
        return index1

DRAW_DIFF = True
DRAW_DIFF = False

ANGELS = ["LeftShoulderAngle", "LeftElbowAngle", "LeftWristAngle",
          "RightShoulderAngle", "RightElbowAngle", "RightWristAngle"]
FILES = ["pro2D.xlsx", "computer2D.xlsx", "tablet2D.xlsx","cellphone2D.xlsx",
        "pro2Dxz.xlsx", "computer2Dxz.xlsx", "tablet2Dxz.xlsx","cellphone2Dxz.xlsx",
        "pro3D.xlsx", "computer3D.xlsx", "tablet3D.xlsx","cellphone3D.xlsx",
        "tabletWhatsApp2D.xlsx"]

LIM_ALL = [-1, 100]
LIM_STATIC = [0, 10]
LINE_WIDTH = 1

# Choose data
angle = ANGELS[1]
fstFile = "data\\"+FILES[2]
sndFile = "data\\"+FILES[3]
trdFile ="data\\"+FILES[0]
frtFile = 0#"data\\"+FILES[0]

var1 = pd.read_excel(fstFile)
var2 = pd.read_excel(sndFile)
if trdFile:
    var3 = pd.read_excel(trdFile)
if frtFile:
    var4 = pd.read_excel(frtFile)

matplotlib.use('TkAgg')
fig, ax = plt.subplots()
ax.set_ylabel('Angle [deg]')
ax.set_xlabel('Time [sec]')


ax.plot(var1['T (sec)'], var1[angle], label=fstFile[5:-5], linewidth=LINE_WIDTH)
ax.plot(var2['T (sec)'], var2[angle], label=sndFile[5:-5], linewidth=LINE_WIDTH)
if trdFile:
        ax.plot(var3['T (sec)'], var3[angle], label=trdFile[5:-5], linewidth=LINE_WIDTH)
if frtFile:
        ax.plot(var4['T (sec)'], var4[angle], label=frtFile[5:-5], linewidth=LINE_WIDTH)


if DRAW_DIFF == True: # |var2|<|var1|, diff = var2 - var1
    rel = []
    for index in range(len(var1['T (sec)'])):
        time_val = var1['T (sec)'][index]
        closest_time = findClosest(var2['T (sec)'], len(var2[angle]), time_val)
        rel_val = (var1[angle][index] - var2[angle][closest_time])
        rel.append(abs(rel_val))
    ax.plot(var1['T (sec)'], rel, linewidth=LINE_WIDTH, label="difference " + sndFile[5:-5]+" - "+fstFile[5:-5])

plt.xlim(LIM_ALL)
ax.set_title(angle)
ax.legend()

plt.show()
