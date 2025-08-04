from flask import Flask, request, jsonify, render_template, send_from_directory
from diffusers import StableDiffusionPipeline 
import torch
import base64
import pickle
from io import BytesIO
from moviepy import ImageSequenceClip

flask_app = Flask(__name__, template_folder='Templates/')

# Load your model (adjust based on your chosen model)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe.to("cuda") # Uncomment if using GPU

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route('/generate-video', methods=['POST'])
def generate_video():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Replace with your actual model inference
    images = pipe(prompt).images
    for i,img in enumerate(images):
        img.save(f"static/videos/image_{i}.png")
    
    image_path = ["static/videos/image_0.png", "static/videos/image_1.png", "static/videos/image_2.png",
              "static/videos/image_3.png", "static/videos/image_4.png", "static/videos/image_5.png"]
    
    video_clip = ImageSequenceClip(image_path, fps=1)
    video_clip.write_videofile("static/videos/my_stitched_video.mp4", codec="libx264")
    return send_from_directory('static/videos', 'my_stitched_video.mp4')
    



    # Placeholder for demonstration
    #from PIL import Image
    #image = Image.new('RGB', (256, 256), color = 'red') # dummy image

    #buffered = BytesIO()
    #image.save(buffered, format="PNG")
    #img_str = base64.b64encode(buffered.getvalue()).decode()


if __name__ == '__main__':
    flask_app.run(debug=True)