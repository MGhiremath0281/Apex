from flask import Flask, render_template, request, redirect, url_for
import os

# Import functions from feature folders
from landing.upload import process_landing_upload
from landing.live import process_landing_live
from bird.upload import process_bird_upload
from bird.live import process_bird_live
from fire.upload import process_fire_upload
from fire.live import process_fire_live

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    feature = request.form.get('feature')
    if file.filename == '':
        return "No selected file"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Call appropriate function
    if feature == 'landing':
        process_landing_upload(file_path)
    elif feature == 'bird':
        process_bird_upload(file_path)
    elif feature == 'fire':
        process_fire_upload(file_path)
    else:
        return "Invalid feature selected"

    return redirect(url_for('result'))

@app.route('/result')
def result():
    return render_template('result.html', img_url='static/output.jpg')

@app.route('/live/<feature>')
def live(feature):
    if feature == 'landing':
        process_landing_live()
    elif feature == 'bird':
        process_bird_live()
    elif feature == 'fire':
        process_fire_live()
    else:
        return "Invalid feature selected"
    return "Live stream ended. <a href='/'>Go back</a>"

if __name__ == '__main__':
    app.run(debug=True)
