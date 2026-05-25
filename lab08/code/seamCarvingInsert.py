import numpy as np
from calcEnergy import calcEnergy
from findOptSeam import findOptSeam
from reduceImageByIndexArray import reduceImageByIndexArray
from enlargeImageByIndexArray import enlargeImageByIndexArray
import tqdm
import os

def seamCarvingInsert(image, insertsize):
    # Duplicate image
    image_duplicate = image.copy()
    h, w, ch = image.shape
    # Create a container to record the seam index
    seamIndex = np.zeros((h, insertsize), dtype=np.uint32)

    # Find seams to remove
    ############### YOUR CODE HERE ###############
    for i in tqdm.tqdm(range(insertsize), desc="Removing seams", leave=False):
        energy = calcEnergy(image_duplicate)
        idx, _ = findOptSeam(energy, 0)
        seamIndex[:, i] = idx
        image_duplicate = reduceImageByIndexArray(image_duplicate, idx, 0)
    ############### YOUR CODE ENDS ###############
    
    # Update seam indices
    ############### YOUR CODE HERE ###############
    for c_ref in tqdm.tqdm(range(insertsize - 2, -1, -1), desc="Updating seam indices", leave=False):
        for c in range(c_ref + 1, insertsize):
            mask = seamIndex[:, c] >= seamIndex[:, c_ref]
            seamIndex[mask, c] += 1
    ############### YOUR CODE ENDS ###############

    # Insert the seam back into the original image.
    # Copy the original image to insert it into the seam.
    output = image.copy()
    ############### YOUR CODE HERE ###############
    for c_ref in tqdm.tqdm(range(insertsize), desc="Inserting seams", leave=False):
        output = enlargeImageByIndexArray(output, seamIndex[:, c_ref])
        for c in range(c_ref + 1, insertsize):
            mask = seamIndex[:, c] >= seamIndex[:, c_ref]
            seamIndex[mask, c] += 1
    ############### YOUR CODE ENDS ###############
    return output

if __name__ == "__main__":
    import cv2
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    output = seamCarvingInsert(I, int(I.shape[1]/2))
    cv2.imshow("Original", I)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
