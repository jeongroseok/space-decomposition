from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import ffmpeg
import numpy as np


def _parse_timedelta(value: str) -> timedelta:
    """
    hh:mm:ss.s 형식의 문자열로부터 timedelta 파싱
    """
    arr = np.array(value.split(':')).astype('float')
    return timedelta(hours=arr[0], minutes=arr[1], seconds=arr[2])


def create_clip(filename: str,
                fps: int,
                starttime: str,
                duration: str,
                resolution: Optional[str] = None) -> np.ndarray:
    """
    crop: top left corner, size
    """
    # 비디오 정보 추출
    video_info = ffmpeg.probe(filename)['streams'][0]
    if resolution is None:
        resolution = f"{video_info['width']}x{video_info['height']}"

    # 출력 해상도
    width, height = np.array(resolution.split('x')).astype('int')

    # GPU 가속 및 기타 옵션
    config = {
        # "hwaccel_output_format": "cuda",
        # "c:v": "hevc_cuvid",
    }

    # 영상 시작 시간 및 길이
    config["ss"] = starttime
    config["t"] = duration

    # 영상 추출
    stream = ffmpeg.input(filename, **config)
    stream = stream.filter('fps', fps=fps, round='up')
    stream = stream.filter('scale', size=resolution)

    stream = stream.output('pipe:', format='rawvideo', pix_fmt='rgb24')
    frames, err = stream.run(capture_stdout=True, capture_stderr=True)

    # ndarray로 변환
    frames = (np.frombuffer(frames, np.uint8).reshape([-1, height, width, 3]))
    return frames
