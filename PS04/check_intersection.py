

corners = [(3, 7), (7, 7), (7, 3), (3, 3)]

pt1 = (3, 7)
pt2 = (3, 3)

line = LineString([pt1, pt2])
other = LineString([corners[2], corners[3]])
print(line.intersects(other))