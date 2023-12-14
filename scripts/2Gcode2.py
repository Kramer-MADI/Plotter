import cv2
import numpy as np
import ezdxf
import matplotlib.pyplot as plt
import math

# Load the image
image = cv2.imread('C:\\Users\\krame\\Pictures\\thumbs.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a 2D array
image_2d = np.array(gray_image)

# Create a new DXF drawing
dwg = ezdxf.new(dxfversion='R2010')

# Add a new layer to the drawing
dwg.layers.new(name='LINES', dxfattribs={'color': 7})

# Get the modelspace
msp = dwg.modelspace()

# Iterate over the 2D array
for i in range(image_2d.shape[0]):
    for j in range(image_2d.shape[1]):
        # If the pixel's value is below a certain threshold, add a point to the DXF drawing
        if image_2d[i, j] < 128:
            msp.add_point((j, i))

# Save the DXF drawing to a file
dwg.saveas(r'C:\Users\krame\progging\Local\dfx\dfx_image.dfx')


# Load the DXF file
dwg = ezdxf.readfile(r'C:\Users\krame\progging\Local\dfx\dfx_image.dfx')

# Get the modelspace
msp = dwg.modelspace()

# Initialize a list of points
points = []

# Iterate over all entities in the modelspace
for e in msp:
    # If the entity is a point, add it to the list of points
    if e.dxftype() == 'POINT':
        x, y, _ = e.dxf.location  # Ignore the z-coordinate
        points.append((x, y))

# Check if points list is not empty
if points:
    # Separate the list of points into two lists: one for the x-coordinates and one for the y-coordinates
    x, y = zip(*points)

    # Use matplotlib's scatter function to display the points
    plt.scatter(x, y, s=1)
    plt.gca().invert_yaxis()  # Invert y axis to match image orientation
    plt.show()
else:
    print("No points found in the DXF file.")

def points_to_gcode(points, threshold):
    # Initialize the G-code commands list
    gcode = ['G90', 'G21', 'G00 Z5']
    
    # Initialize the previous point
    prev_point = points[0]
    
    # Get the maximum y-coordinate and minimum x-coordinate
    max_y = np.max(points[:, 1])
    min_x = np.min(points[:, 0])
    
    for point in points:
        # Shift the x-coordinate and invert the y-coordinate
        x, y = point
        x -= min_x
        y = max_y - y
        
        # Calculate the distance from the previous point
        dist = distance(prev_point, point)
        
        # If the distance is greater than the threshold, lift the tool
        if dist > threshold:
            gcode.append('G00 Z1')
        
        # Move to the point's coordinates
        gcode.append(f'G01 X{x} Y{y}')
        
        # Update the previous point
        prev_point = point
    
    # Lift the tool after all points have been processed
    gcode.append('G00 Z5')
    
    return '\n'.join(gcode)


# Open a file in write mode to sort points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def sort_points(points):
    for i in range(len(points) - 1):
        min_distance = float('inf')
        min_index = i + 1
        for j in range(i + 1, len(points)):
            dist = distance(points[i], points[j])
            if dist < min_distance:
                min_distance = dist
                min_index = j
        points[i + 1], points[min_index] = points[min_index], points[i + 1]
    return points

# Sort the points
points = sort_points(points)

# Define the distance threshold
threshold = 3  # Adjust this value as needed

# Convert the points to a numpy array
points_array = np.array(points)

# Convert the points to G-code
gcode = points_to_gcode(points_array, threshold)

# Open a file in write mode
with open(r'C:\Users\krame\plots2\output.ngc', 'w') as f:
    # Write the G-code to the file
    f.write(gcode)

print("G-code has been written to output.ngc")