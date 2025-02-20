import cv2
import tensorflow as tf
import numpy as np
import os


def _sigmoid(x):
    return 1. / ((1. + np.exp(-x)) + np.finfo(np.float32).eps)


def use_tflite_model(name, img_path):
    # check model type
    parts = name.split(".")
    if parts[-1] != "tflite":
        return None

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=name)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    image = cv2.imread(img_path)
    img_width, img_height = input_details[0]['shape'][1], input_details[0]['shape'][2]
    input_data = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    input_data = np.array(input_data, dtype=np.uint8)  # Typ anpassen
    input_data = np.expand_dims(input_data, axis=0)  # Batch-Dimension hinzufügen

    # Fit the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    raw_circle = interpreter.get_tensor(output_details[0]['index'])[0]  # (y_pred, x, y)
    raw_classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects

    # Werte korrigieren und normalisieren
    circle = _sigmoid(raw_circle)
    classes = _sigmoid(raw_classes)

    print("Circle Data:", circle)  # Debug-Ausgabe
    print("Classes: ", classes)  # Debug-Ausgabe

    # Werte aus dem Modell extrahieren
    y_pred, x, y = circle

    # Skalierung auf die Originalbildgröße
    x = int(x * img_width)
    y = int(y * img_height)

    obj_threshold = 0.5
    binary_y_pred = int(y_pred > obj_threshold)

    if binary_y_pred == 1:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    resize_factor = 5

    img_cv2 = cv2.resize(image, (resize_factor * img_width, resize_factor * img_height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.drawMarker(img_cv2, (x * resize_factor, y * resize_factor), color, cv2.MARKER_CROSS, resize_factor * 2,
                   max(2, resize_factor))

    # Create output folder if it does not exist
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Save output image in the output folder
    output_path = os.path.join(output_folder, os.path.basename(img_path)[:-4] + "_out.png")
    cv2.imwrite(output_path, img_cv2)


def main():
    model_name = "final_model.tflite"
    image_folder = "imgs"

    # Iterate through all files in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Ensure the file is an image (optional)
        if image_name.lower().endswith('.png'):
            print(f"Processing: {image_path}")
            use_tflite_model(model_name, image_path)


if __name__ == "__main__":
    main()
