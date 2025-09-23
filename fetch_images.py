import os
import requests
from PIL import Image
from io import BytesIO

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")


def _fetch_data_from_api(url):
    res = requests.get(url, timeout=100)
    res.raise_for_status()
    return res.json()


def _download_image_from_url(img_url, save_path):
    response = requests.get(img_url, timeout=100)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
            return True
    return False


def _fetch_image_unsplash(keyword, save_path, access_key=UNSPLASH_ACCESS_KEY):
    url = (
        f"https://api.unsplash.com/photos/random?query={keyword}&client_id={access_key}"
    )
    try:
        data = _fetch_data_from_api(url)
        img_url = data.get("urls", {}).get("regular")
        found = _download_image_from_url(img_url, save_path)
        if not found:
            print(f"Did not find Unsplash image for {keyword}")
        else:
            print(f"Found Unsplash image for {keyword}")
        return found
    except Exception as e:
        print(f"Failed to fetch Unsplash image for {keyword}: {e}")
        return False
    

def _get_pixabay_images(keyword, api_key=PIXABAY_API_KEY, amount=1) -> list[Image.Image]:
    url = (
        f"https://pixabay.com/api/?key={api_key}&q={keyword}&image_type=photo&per_page={amount}"
    )
    try:
        data = _fetch_data_from_api(url)
        img_urls = [hit.get("webformatURL") for hit in data.get("hits", [])]
        print(f"Found {len(img_urls)} Pixabay images for {keyword}")
        # Download images using async requests
        

        
        # return [Image.open(BytesIO(requests.get(img_url, stream=True, timeout=1000).content)) for img_url in img_urls]
    except Exception as e:
        print(f"Failed to fetch Pixabay image for {keyword}: {e}")
        return []


def _save_image_pixabay_to_file(keyword, save_path, api_key=PIXABAY_API_KEY, amount=1):
    url = (
        f"https://pixabay.com/api/?key={api_key}&q={keyword}&image_type=photo&per_page={amount}"
    )
    try:
        data = _fetch_data_from_api(url)
        img_urls = [hit.get("webformatURL") for hit in data.get("hits", [])]
        for img_url in img_urls:
            _download_image_from_url(img_url, save_path)
        return len(img_urls)
    except Exception as e:
        print(f"Failed to fetch Pixabay image for {keyword}: {e}")
        return 0

def download_images_by_category(data, path="data/images") -> dict[str, list[str]]:
    categories = data.keys()
    os.makedirs(path, exist_ok=True)
    image_paths = {cat: [] for cat in categories}
    for cat in categories:
        for datapoint in data[cat]:
            img_filename = f"{path}/{datapoint.replace(' ', '_')}.jpg"
            if not os.path.exists(img_filename):
                success = _save_image_pixabay_to_file(datapoint, img_filename) > 0
                if not success:
                    success = _fetch_image_unsplash(datapoint, img_filename)
                    if not success:
                        continue
            if os.path.exists(img_filename):
                image_paths[cat].append(img_filename)
    return image_paths

def download_image_using_amount(keyword: str, amount:int=1, path="data/images") -> list[str]:
    os.makedirs(path, exist_ok=True)
    image_paths = []
    for i in range(amount):
        img_filename = f"{path}/{keyword.replace(' ', '_')}_{i}.jpg"
        if not os.path.exists(img_filename):
            found_count = _save_image_pixabay_to_file(keyword, img_filename, amount=amount)
            print(f"Fetched {found_count} images from Pixabay for {keyword}")
        if os.path.exists(img_filename):
            image_paths.append(img_filename)
    return image_paths

def download_image_by_list_using_amount(keywords: list[str], amount:int=1, path="data/images") -> list[str]:
    os.makedirs(path, exist_ok=True)
    image_paths = []
    for keyword in keywords:
        for i in range(amount):
            img_filename = f"{path}/{keyword.replace(' ', '_')}_{i}.jpg"
            if not os.path.exists(img_filename):
                found_count = _save_image_pixabay_to_file(keyword, img_filename, amount=amount)
                print(f"Fetched {found_count} images from Pixabay for {keyword}")
            if os.path.exists(img_filename):
                image_paths.append(img_filename)
    return image_paths

def generate_and_save_images_from_colors(colors:list[str], path="data/images") -> list[str]:
    os.makedirs(path, exist_ok=True)
    image_paths = []
    for color in colors:
        img_filename = f"{path}/{color.replace(' ', '_')}.jpg"
        if not os.path.exists(img_filename):
            img = generate_single_image_for_color(color)
            img.save(img_filename)
            print(f"Generated image for color {color}")
        image_paths.append(img_filename)
    return image_paths

def generate_single_image_for_color(color:str) -> Image.Image:
    img = Image.new("RGB", (100, 100), color)
    return img