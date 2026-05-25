import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calcEnergy(I):
    """
    Sum up the energy for each channel
    """
    I = I.astype(np.float64)
    dx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64)
    dy = dx.T  # vertical gradient filter

    ############### YOUR CODE HERE ###############
    R, G, B = I[:,:,0], I[:,:,1], I[:,:,2]

    dIrx = cv2.filter2D(R, -1, dx, borderType=cv2.BORDER_REFLECT_101)
    dIry = cv2.filter2D(R, -1, dy, borderType=cv2.BORDER_REFLECT_101)

    dIgx = cv2.filter2D(G, -1, dx, borderType=cv2.BORDER_REFLECT_101)
    dIgy = cv2.filter2D(G, -1, dy, borderType=cv2.BORDER_REFLECT_101)

    dIbx = cv2.filter2D(B, -1, dx, borderType=cv2.BORDER_REFLECT_101)
    dIby = cv2.filter2D(B, -1, dy, borderType=cv2.BORDER_REFLECT_101)

    energy = np.abs(dIrx) + np.abs(dIry) + np.abs(dIgx) + np.abs(dIgy) + np.abs(dIbx) + np.abs(dIby)

    ############### YOUR CODE ENDS ###############
    return energy.astype(np.float64)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    energy = calcEnergy(I)
    print(energy.shape)
    print(energy)

    plt.imshow(energy, cmap='viridis')
    plt.colorbar(label='Energy Level')
    plt.axis('off')
    plt.show()
