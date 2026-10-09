from torch import true_divide
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# load pre trained processor and model
processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

#  set path
img_path = 'images/pandacub.jpg'

# convert to RGB
image = Image.open(img_path).convert('RGB')

# set text prompt
text = 'this is a image of: '

# The return_tensors argument is set to "pt" to return PyTorch tensors.
inputs = processor(images=image, text=text, return_tensors='pt')

# generate caption with 50 token max
outputs = model.generate(**inputs, max_length=50)

# decode output token
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)