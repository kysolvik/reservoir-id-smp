"""Use SICAR Python package to download hydrographies for all Brasil states

Prerequisites:
$ pip install git+https://github.com/urbanogilson/SICAR
$ sudo apt install tesseract-ocr

Note:
    Some issues with rate throttling mean we have to use a special
    custom_download_polygon method. See this discussion for more info:
        https://github.com/urbanogilson/SICAR/issues/32

"""
from types import MethodType
from types import MethodType
from pathlib import Path
import os
import time
from tqdm import tqdm
import httpx
from urllib.parse import urlencode
from SICAR.exceptions import (
    UrlNotOkException,
    FailedToDownloadPolygonException,
)
from collections import deque
from SICAR import Sicar, State, Polygon

# Create Sicar instance
car = Sicar()

def custom_download_polygon(
    self,
    state,
    polygon,
    captcha: str,
    folder: str,
    chunk_size: int = 1024,
    max_retries: int = 200,
    retry_delay: int = 1,
    min_speed_threshold: int = 100,  # Minimum speed in bytes per second
    speed_check_interval: int = 10,  # Number of chunks to average over for speed check
) -> Path:
    query = urlencode({"idEstado": state.value, "tipoBase": polygon.value, "ReCaptcha": captcha})
    path = Path(os.path.join(folder, f"{state.value}_{polygon.value}")).with_suffix(".zip")
    headers = {}

    # Check if a partial file already exists and set Range header
    if path.exists():
        current_size = path.stat().st_size
        headers["Range"] = f"bytes={current_size}-"
    else:
        current_size = 0

    retries = 0
    while retries < max_retries:
        try:
            with self._session.stream("GET", f"{self._DOWNLOAD_BASE}?{query}", headers=headers) as response:
                if response.status_code not in (httpx.codes.OK, httpx.codes.PARTIAL_CONTENT):
                    raise UrlNotOkException(f"{self._DOWNLOAD_BASE}?{query}")

                content_length = int(response.headers.get("Content-Length", 0))
                total_size = content_length + current_size

                content_type = response.headers.get("Content-Type", "")
                if content_length == 0 or not content_type.startswith("application/zip"):
                    raise FailedToDownloadPolygonException()

                # Resume or start a new download
                mode = "ab" if current_size > 0 else "wb"
                with open(path, mode) as fd:
                    with tqdm(
                        initial=current_size,
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=f"Downloading polygon '{polygon.value}' for state '{state.value}'",
                    ) as progress_bar:
                        speed_history = deque(maxlen=speed_check_interval)
                        start_time = time.time()
                        
                        for chunk in response.iter_bytes(chunk_size=chunk_size):
                            fd.write(chunk)
                            progress_bar.update(len(chunk))
                            
                            # Update the end time after each chunk
                            end_time = time.time()
                            
                            # Track the download speed every few chunks based on the interval
                            if len(speed_history) < speed_check_interval:
                                speed = len(chunk) / (end_time - start_time) / 10
                                speed_history.append(speed)
                            else:
                                avg_speed = sum(speed_history) / len(speed_history)
                                if avg_speed < min_speed_threshold:
                                    raise httpx.ReadTimeout(f"Average download speed too low: {avg_speed:.2f} bytes/sec")

                                start_time = end_time
                                speed_history.clear()

                print("Download completed successfully.")
                return path  # Exit if download is complete

        except (httpx.RequestError, httpx.ReadTimeout) as e:
            retries += 1  # Increment retries before printing
            print(f"Retry {retries}/{max_retries} after error: {e}")
            time.sleep(retry_delay)

    # If we exit the loop, the download failed after all retries
    raise FailedToDownloadPolygonException()

car._download_polygon = MethodType(custom_download_polygon, car)

# Download a state
for s in State:
    if s.name =='BA':
        if not os.path.isfile('SICAR/{0}/{0}_HIDROGRAFIA.zip'.format(s.name)):
            car.download_state(s, Polygon.HYDROGRAPHY, folder='SICAR/{}'.format(s.name))
