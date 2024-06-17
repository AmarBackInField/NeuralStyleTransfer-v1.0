import logging
from torchvision import transforms, models
from PIL import Image
import torch
import io
from main import run_style_transfer
from flask import Flask, request, send_file, render_template, Response

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    try:
        content_file = request.files['content']
        style_file = request.files['style']
        
        content_path = 'content.jpg'
        style_path = 'style.jpg'
        
        # Save the uploaded files
        content_file.save(content_path)
        style_file.save(style_path)
        
        logging.info("Files saved successfully.")
        
        # Run the style transfer
        best_img, best_loss, imgs = run_style_transfer(content_path, style_path)
        
        logging.info("Style transfer completed.")
        
        # Convert the result to an image
        result_image = Image.fromarray(best_img)
        img_io = io.BytesIO()
        result_image.save(img_io, 'JPEG')
        img_io.seek(0)
        
        logging.info("Result image prepared for sending.")
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        logging.error("Error during style transfer: %s", str(e))
        return Response("Error during style transfer", status=500)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
