from flask import Flask, render_template, request

from transform import sketch2fashion

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    image = request.files['image']
    input_image = 'input.' + image.filename.split('.')[-1]
    image.save('static/' + input_image)

    output_image = 'output.' + image.filename.split('.')[-1]
    sketch2fashion(input_image, output_image)

    return render_template('result.html')

if __name__ == '__main__':
    app.run()
