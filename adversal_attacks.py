import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for inference.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def main():
    # Load pre-trained VGG16 model
    model = VGG16(weights='imagenet')

    # Load and preprocess a sample image
    image_path = 'path/to/your/image.jpg'  # Replace with the path to your image
    original_image = load_and_preprocess_image(image_path)

    # Get the true label of the original image
    original_image = tf.expand_dims(original_image, axis=0)
    true_label = decode_predictions(model.predict(original_image), top=1)[0][0][1]
    print(f"True Label: {true_label}")

    # Define the FGSM attack
    epsilon = 0.01  # Strength of the attack
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        gradient = tape.gradient(loss, input_image)
        perturbation = epsilon * tf.sign(gradient)
        return perturbation

    # Create adversarial image
    input_label = tf.one_hot([224], 1000)  # Use any target label (e.g., label 224)
    perturbation = create_adversarial_pattern(original_image, input_label)
    adversarial_image = original_image + perturbation

    # Get the label of the adversarial image
    adversarial_label = decode_predictions(model.predict(adversarial_image), top=1)[0][0][1]
    print(f"Adversarial Label: {adversarial_label}")

if __name__ == "__main__":
    main()
