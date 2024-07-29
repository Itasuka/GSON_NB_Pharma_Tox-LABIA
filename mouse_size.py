# Import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def main(img = "images_for_openCV/open_field_3.png", l = 13):
    """
    Calculate the width of the objects on the picture based on the leftmost object length

    The 2 parameters are :
    - img the path to the image to analyse
    - l the length of the leftmost object
    
    Return a tuple containing this 4 variables representing the first object dimension :
    - dimA the length in centimeters
    - dA the length in pixels
    - dimB the width in centimeters
    - dB the width in pixels
    """
    """
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    ap.add_argument("-w", "--width", type=float, required=True,
        help="width of the object in the image (in centimeters)")
    args = vars(ap.parse_args())
    """
    # Load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)

    # Apply a Gaussian blur to the image
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    edged = cv2.Canny(gray, 30, 150)  # Adjusted Canny parameters for better edge detection
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Display the intermediate steps
    #cv2.imshow("Gray", gray)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Draw all contours found
    all_contours = image.copy()
    cv2.drawContours(all_contours, cnts, -1, (0, 255, 0), 2)
    #cv2.imshow("All Contours", all_contours)
    #cv2.waitKey(0)

    # Initialize the pixels per metric variable
    pixelsPerMetric = None

    # Loop over the contours individually
    for c in cnts:
        # If the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 9000:  # Adjust this value if needed
            continue

        # Compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv4() else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order
        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

        # Loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Unpack the ordered bounding box, then compute the midpoint between the top-left and top-right coordinates,
        # followed by the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # Compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # If the pixels per metric has not been initialized, compute it as the ratio of pixels to supplied metric
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / l

        # Compute the size of the object in centimeters
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Determine which dimension is length and which is width
        length = max(dimA, dimB)
        width = min(dimA, dimB)

        # Draw the object sizes on the image
        cv2.putText(image, "{:.1f}cm".format(length), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, "{:.1f}cm".format(width), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Print the dimensions in pixels
        print("Dimensions in pixels: {:.1f} px x {:.1f} px".format(dA, dB))

        # Show the output image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

    # Close all OpenCV windows
    #cv2.destroyAllWindows()
    return (dimA, dA, dimB, dB)

main()