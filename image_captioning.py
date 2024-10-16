from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

# Upload the file
uploaded = files.upload()

# Open and display the image
for fn in uploaded.keys():
    img = Image.open(fn)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()

!pip install transformers
!pip install torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image (assuming it was uploaded as shown above)
for fn in uploaded.keys():
    image = Image.open(fn)

# Preprocess the image
inputs = processor(image, return_tensors="pt")

# Generate a caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Generated caption:", caption)

