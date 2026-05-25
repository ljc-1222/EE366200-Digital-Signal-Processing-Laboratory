import cv2
import tqdm
import os
from calcEnergy import calcEnergy
from findOptSeam import findOptSeam
from reduceImageByIndexArray import reduceImageByIndexArray

def seamCarvingContentAmplification(image):
    """
    Enlarge image with scaling, then reduce by seam carving
    """
    # input size
    h_input, w_input, ch = image.shape
    # Enlarge image with standard scaling
    enlarged_image = cv2.resize(image, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
    h_enlarge, w_enlarge, ch = enlarged_image.shape

    ############### YOUR CODE HERE ###############
    for i in tqdm.tqdm(range (w_enlarge - w_input), desc="Reducing horizontal seams", leave=False):
        energy = calcEnergy(enlarged_image)
        optSeamIndexArray, seamE = findOptSeam(energy, 0)
        enlarged_image = reduceImageByIndexArray(enlarged_image, optSeamIndexArray, 0)
        
    for i in tqdm.tqdm(range (h_enlarge - h_input), desc="Reducing vertical seams", leave=False):
        energy = calcEnergy(enlarged_image)
        optSeamIndexArray, seamE = findOptSeam(energy, 1)
        enlarged_image = reduceImageByIndexArray(enlarged_image, optSeamIndexArray, 1)
        
    output = enlarged_image
    ############### YOUR CODE ENDS ###############

    return output

if __name__ == "__main__":
    import cv2
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    output = seamCarvingContentAmplification(I)
    cv2.imshow("Original", I)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
