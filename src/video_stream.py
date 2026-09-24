import threading
import time

import cv2


class ThreadedVideoStream:
    """Continuously reads frames from a video source on a background thread
    so inference never blocks on camera I/O."""

    def __init__(self, source=0, reconnect_delay: float = 0.5):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self._lock = threading.Lock()
        self._frame = None
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> "ThreadedVideoStream":
        self._thread.start()
        return self

    def _update(self):
        while not self._stopped:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(self.reconnect_delay)
                self.capture.open(self.source)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1)
        self.capture.release()
