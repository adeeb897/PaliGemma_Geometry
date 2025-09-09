import os
import requests

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


def _fetch_image_pixabay(keyword, save_path, api_key=PIXABAY_API_KEY):
    url = (
        f"https://pixabay.com/api/?key={api_key}&q={keyword}&image_type=photo&per_page=3"
    )
    try:
        data = _fetch_data_from_api(url)
        img_url = data.get("hits", [])[0].get("webformatURL")
        found = _download_image_from_url(img_url, save_path)
        if not found:
            print(f"Did not find Pixabay image for {keyword}")
        else:
            print(f"Found Pixabay image for {keyword}")
        return found
    except Exception as e:
        print(f"Failed to fetch Pixabay image for {keyword}: {e}")
        return False

def fetch_all_images_for(data, path="data/images") -> dict[str, list[str]]:
    categories = data.keys()
    os.makedirs(path, exist_ok=True)
    image_paths = {cat: [] for cat in categories}
    for cat in categories:
        for datapoint in data[cat]:
            img_filename = f"{path}/{datapoint.replace(' ', '_')}.jpg"
            if not os.path.exists(img_filename):
                success = _fetch_image_pixabay(datapoint, img_filename)
                if not success:
                    success = _fetch_image_unsplash(datapoint, img_filename)
                    if not success:
                        continue
            if os.path.exists(img_filename):
                image_paths[cat].append(img_filename)
    return image_paths
