from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from net.net import net
from torchvision.transforms import functional as F
from PIL import Image

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
model = net().cuda()
model.load_state_dict(torch.load("weights/epoch_3000.pth", map_location="cpu"))
model = model.cuda()
print('Pre-trained model is loaded.')

def process_image(file_path):
    preprocess = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor()
    ])

    input_image = Image.open(file_path).convert("RGB")
    input_image = preprocess(input_image)
    input_image = input_image.unsqueeze(0).cuda()

    L, R, X = model(input_image)
    I = torch.pow(L, 0.2) * R
    return I

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        output_image = process_image(file_path)
        output_image = F.to_pil_image(output_image.squeeze(0).cpu(), 'RGB')
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        output_image.save(output_image_path)

        return jsonify({'image': f'output/{filename}'}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/static/output/<filename>')
def send_output_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
