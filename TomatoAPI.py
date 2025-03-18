from flask import Flask, request, jsonify
from ResNetFunctions import load_model, test_model_with_image, image_transforms, get_categories
from PIL import Image
import io

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    # 'image' es la clave con la que se envía el archivo en el formulario
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen"}), 400

    file = request.files['image']
    
    # convertir la imagen a un objeto de imagen aceptable para el modelo ResNet
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"No se pudo abrir la imagen ->  {e}"}), 400
    
    try:
        index_result  = test_model_with_image(model, image, transform_val)
        category = categories[index_result]
    except Exception as e:
        return jsonify({"error": f"No se pudo clasificar la imagen -> {e}"}), 500
    
    return jsonify({"category": category})

if __name__ == '__main__':
    
    model = load_model()
    transform_train, transform_val = image_transforms()
    categories = get_categories()
    
    app.run(debug=True)
