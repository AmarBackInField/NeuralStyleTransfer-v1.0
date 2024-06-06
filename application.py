from torchvision import transforms, models
from PIL import Image
import torch
import io
from main import run_style_transfer
from flask import Flask, request, jsonify,send_file, render_template,Response


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    content_file = request.files['content']
    style_file = request.files['style']
    
    content_path = 'content.jpg'
    style_path = 'style.jpg'
    
    content_file.save(content_path)
    style_file.save(style_path)
    
    best_img, best_loss, imgs = run_style_transfer(content_path, style_path)

    result_image = Image.fromarray(best_img)
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)