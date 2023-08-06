import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import imageio

def augment_data(images, labels):
    """
    Augment image data using various augmentation techniques.

    Args:
        images (List[numpy.ndarray]): List of input images.
        labels (List[int]): List of corresponding labels.

    Returns:
        List[numpy.ndarray], List[int]: Augmented images and their corresponding labels.
    """
    # Convert the list of images to an array
    images_arr = np.array(images)

    # Augmentation pipeline
    augmentation_seq = iaa.Sequential([
        iaa.Fliplr(0.5),                  # Horizontal flip with a 50% chance
        iaa.Affine(rotate=(-45, 45)),     # Random rotation between -45 to 45 degrees
        iaa.Multiply((0.7, 1.3))          # Brightness adjustment
    ])

    # Apply augmentation to the images
    augmented_images = augmentation_seq(images=images_arr)

    # Convert the augmented images array back to a list
    augmented_images_list = list(augmented_images)

    # Convert the list of labels to an array
    labels_arr = np.array(labels)

    # Return the augmented images and labels as lists
    return augmented_images_list, labels_arr.tolist()

# Example usage
if __name__ == "__main__":
    # Load your image data and corresponding labels
    # Replace the sample_images and sample_labels with your actual data
    sample_images = [imageio.imread('path/to/image1.jpg'), imageio.imread('path/to/image2.jpg')]
    sample_labels = [0, 1]

    # Augment the data
    augmented_images, augmented_labels = augment_data(sample_images, sample_labels)

    # Visualize the original and augmented images
    for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")

    for i, (aug_image, aug_label) in enumerate(zip(augmented_images, augmented_labels)):
        plt.subplot(2, 2, i + 3)
        plt.imshow(aug_image)
        plt.title(f"Augmented - Label: {aug_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
