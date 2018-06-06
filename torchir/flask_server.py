from base64 import standard_b64encode
from io import BytesIO
from flask import Flask, render_template, request
from PIL import Image
from model_resnet import predict_category
from model_homebrew import (
    get_network,
    CLASSES, 
    PREPROCESSING_TRANSFORM, 
    SIZE_TRANSFORM
)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    image = request.files['image']

    img = SIZE_TRANSFORM(Image.open(image).convert(mode='RGB'))
    category_num = predict_category(PREPROCESSING_TRANSFORM(img), net)

    category_name = CLASSES[category_num]

    # reencode the resized image before sending back
    byte_stream = BytesIO()
    img.save(byte_stream, 'png', optimize=True)
    byte_stream.seek(0)

    context = {
        'response_string': category_name,
        'image_type': image.headers['Content-Type'],
        'image_data': str(standard_b64encode(byte_stream.read()))[2:-1]
        }
    return render_template('result.html', **context)


print("Loading CNN, please stand by.")
net = get_network()
print("CNN loaded, proceeding.")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
