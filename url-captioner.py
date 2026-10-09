from time import time
from concurrent.futures import process
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# load processor and model
# load pre trained processor and model
processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# URL to scrape
url = "https://en.wikipedia.org/wiki/Giant_panda"

# request
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
print("Status: ", response.status_code, "HTML size: ", len(response.text))

# create BeautifulSoup object to parse HTML
soup = BeautifulSoup(response.text, "html.parser")
img_elements=soup.find_all("img")
print(f"Found {len(img_elements)} image(s)")

with open("captions.txt", "w", encoding="utf-8") as caption_file:
    for index, img_elements in enumerate(img_elements, start=1):
        # try different attributes
        img_url = img_elements.get("src") or img_elements.get("data-src")
        if not img_url and img_elements.has_attr("srcset"):
            img_url = img_elements["srcset"].split()[0]
        if not img_url:
            continue
            
        #  skip svg
        if img_url.endswith(".svg") or ".svg" in img_url:
            continue

        #  fix relative urls
        if img_url.startswith("//"):
            img_url="https:"+img_url
        elif img_url.startswith("/"):
            img_url="https://en.wikipedia.org"+img_url
        elif not img_url.startswith("http"):
            continue
        
        print(img_url)
        try:
            r = requests.get(img_url, timeout=10, headers=headers)
            raw_image = Image.open(BytesIO(r.content))
            
            # skip small image
            if raw_image.size[0]*raw_image.size[1] < 200:
                continue

            raw_image = raw_image.convert("RGB")

            # process the image captioning
            text = "image description"
            inputs = processor(images=raw_image, text=text, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{img_url}: {caption}\n")
            print(f"[{index}] Caption saved")
        except OSError:
            # skip all images
            continue
        except Exception as e:
            print(f"Failed to caption image {index}: {e}")
            continue


