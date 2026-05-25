from calcEnergy import calcEnergy
from findOptSeam import findOptSeam
from reduceImageByIndexArray import reduceImageByIndexArray

import tqdm
import os

def seamCarvingReduce(image, reduceSize, seamDirection):
    """
    Reduce image by removing 'reduceSize' seams
    """
    output = image.copy()
    for k in tqdm.tqdm(range(reduceSize), desc="Reducing seams", leave=False):
        energy = calcEnergy(output)
        optSeamIndexArray, seamE = findOptSeam(energy, seamDirection)
        output = reduceImageByIndexArray(output, optSeamIndexArray, seamDirection)
    return output

if __name__ == "__main__":
    import cv2
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    output = seamCarvingReduce(I, int(I.shape[1]/2), 0)
    cv2.imshow("Original", I)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
