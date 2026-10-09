import gradio
import numpy
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# load pre trained processor and model
processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

def caption_image(input_image:numpy.ndarray):
    # convert numpy array to PIL image and convert to RGB
    pil_image = Image.fromarray(input_image).convert('RGB')

    # text prompt
    text = "image description: "

    # process image + text input
    inputs = processor(images=pil_image, text=text, return_tensors='pt')

    # generate caption with 50 token max
    outputs = model.generate(**inputs, max_length=50)

    # decode output token
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iface = gradio.Interface(
    fn=caption_image,
    inputs=gradio.Image(),
    outputs="text",
    title='Image Captioning',
    description='Input an image and get a caption'
)

    
iface.launch()