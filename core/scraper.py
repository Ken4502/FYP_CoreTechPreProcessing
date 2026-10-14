import urllib.request
import urllib.parse
import json
import io
import ssl
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional

# Context for SSL connections
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def search_wikimedia_commons(query: str, limit: int = 8) -> List[Dict[str, str]]:
    """
    Searches Wikimedia Commons open repository for botanical and plant disease images.
    Filters out non-image formats (PDF, SVG, TIF) and retrieves responsive thumbnails.
    """
    clean_query = query.strip()
    encoded = urllib.parse.quote(clean_query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrsearch={encoded}&gsrnamespace=6"
        f"&gsrlimit={limit * 2}&prop=imageinfo&iiprop=url|mime|thumburl&iiurlwidth=600"
    )

    req = urllib.request.Request(url, headers=HEADERS)
    results = []
    try:
        with urllib.request.urlopen(req, timeout=9, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                title = pdata.get('title', '').replace('File:', '').replace('_', ' ')
                # Remove extension from title for clean display
                clean_title = title.rsplit('.', 1)[0]
                image_info = pdata.get('imageinfo', [{}])[0]
                mime = image_info.get('mime', '')

                if mime in ['image/jpeg', 'image/png', 'image/webp']:
                    thumb = image_info.get('thumburl', image_info.get('url'))
                    full = image_info.get('url')
                    if thumb and full:
                        results.append({
                            'id': f"wiki_{pid}",
                            'title': clean_title[:45],
                            'thumb_url': thumb,
                            'full_url': full,
                            'source': 'Wikimedia Commons'
                        })
                if len(results) >= limit:
                    break
    except Exception as e:
        print(f"Wikimedia search notice: {e}")

    return results

def search_inaturalist_flora(query: str, limit: int = 6) -> List[Dict[str, str]]:
    """
    Searches iNaturalist global biodiversity database for verified plant observations.
    """
    clean_query = query.strip()
    encoded = urllib.parse.quote(clean_query)
    url = f"https://api.inaturalist.org/v1/observations?taxon_name={encoded}&photos=true&per_page={limit}&order_by=votes"

    req = urllib.request.Request(url, headers=HEADERS)
    results = []
    try:
        with urllib.request.urlopen(req, timeout=9, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for item in data.get('results', []):
                taxon = item.get('taxon', {})
                common = taxon.get('preferred_common_name') or taxon.get('name') or "Plant Observation"
                photos = item.get('photos', [])
                if photos:
                    p = photos[0]
                    u = p.get('url', '')
                    if u:
                        med_url = u.replace('square', 'medium')
                        orig_url = u.replace('square', 'original')
                        results.append({
                            'id': f"inat_{item.get('id')}",
                            'title': common[:45],
                            'thumb_url': med_url,
                            'full_url': orig_url,
                            'source': 'iNaturalist Flora'
                        })
    except Exception as e:
        print(f"iNaturalist search notice: {e}")

    return results

def search_plant_images_online(query: str, limit: int = 8) -> List[Dict[str, str]]:
    """
    Multi-source plant image search across Wikimedia Botanical and iNaturalist.
    """
    if not query or len(query.strip()) < 2:
        return []

    # Search Wikimedia first
    results = search_wikimedia_commons(query, limit=limit)
    
    # If fewer than limit, augment with iNaturalist
    if len(results) < limit:
        needed = limit - len(results)
        inat_results = search_inaturalist_flora(query, limit=needed)
        results.extend(inat_results)

    return results[:limit]

def fetch_image_from_url(url: str, max_dimension: int = 1024) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Downloads an image from a URL, validates format, downsamples high-res to max_dimension,
    and returns (img_rgb, error_message).
    """
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12, context=_ssl_context) as resp:
            data = resp.read()

        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = pil_img.size

        # Downsample gently if excessively large to avoid CPU matting bottlenecks
        if max(w, h) > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return np.array(pil_img), None
    except Exception as e:
        return None, f"Failed to download image from web: {str(e)}"
