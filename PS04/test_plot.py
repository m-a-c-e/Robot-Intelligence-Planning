import matplotlib.pyplot as plt

corners = [(3, 7), (7, 7), (7, 3), (3, 3)]

plt.plot([1], [1], label='S', marker='X', markersize=10, color='g')
plt.plot(9, 9, label='G', marker='X', markersize=10, color='g')


for i in range(0, len(corners)):
    corner1 = corners[i]
    if i == len(corners) - 1:
        corner2 = corners[0]
        plt.plot(corner1, corner2, color='r')
    else:
        corner2 = corners[i + 1]
        plt.plot(corner1, corner2, color='r')

plt.autoscale()
plt.show()


