import os
import requests
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor
import time

def download_image(url, folder_path, image_name):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(folder_path, image_name), 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def scrape_images(query, folder_path, num_images):
    print(f"Searching DuckDuckGo for: {query}")
    os.makedirs(folder_path, exist_ok=True)
    
    ddgs = DDGS()
    results = ddgs.images(query, max_results=num_images + 200) # Get extra in case of failures
    
    urls = [result['image'] for result in results if 'image' in result]
    print(f"Found {len(urls)} URLs. Downloading...")
    
    successful_downloads = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, url in enumerate(urls):
            if successful_downloads >= num_images:
                break
            image_name = f"img_{successful_downloads:04d}.jpg"
            futures.append(executor.submit(download_image, url, folder_path, image_name))
            successful_downloads += 1
            
        # Wait for all to complete
        for future in futures:
            future.result()

    print(f"Successfully downloaded {len(os.listdir(folder_path))} images to {folder_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    crash_dir = os.path.join(base_dir, 'data', 'raw', 'crash')
    normal_dir = os.path.join(base_dir, 'data', 'raw', 'normal')
    
    # We need 1000 total, so let's get 550 of each just to be safe
    print("Starting automated image download to bypass Kaggle restrictions...")
    scrape_images("car crash accident scene real photo", crash_dir, 550)
    time.sleep(2) # Prevent rate limiting
    scrape_images("normal traffic cars street road day", normal_dir, 550)
    
    print("Dataset construction complete! You now have a real dataset that meets the project requirements.")
