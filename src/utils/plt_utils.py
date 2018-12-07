import matplotlib.pyplot as plt


# helper to show 2 images next to each other
def visualize_result(original, result, gray=False, show=True):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=30)
    if gray:
        ax2.imshow(result, cmap='gray')
    else:
        ax2.imshow(result)
    ax2.set_title('Result Image', fontsize=30)
    if show:
        plt.show()


def visualize_result4(img1, img2, img3, img4):
    f, (ax1, ax2) = plt.subplots(2, 2, figsize=(20, 10))
    ax1[0].imshow(img1[0])
    ax1[0].set_title(img1[1], fontsize=30)
    ax2[0].imshow(img2[0])
    ax2[0].set_title(img2[1], fontsize=30)
    ax1[1].imshow(img3[0])
    ax1[1].set_title(img3[1], fontsize=30)
    ax2[1].imshow(img4[0])
    ax2[1].set_title(img4[1], fontsize=30)
    plt.show()

