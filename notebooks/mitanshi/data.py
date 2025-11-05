import requests, cv2, numpy as np

url = "http://192.168.1.107:5001/video/video_benign_Pt-num-000_vid-num-000.mp4"
r = requests.get(url, stream=True)

for chunk in r.iter_content(chunk_size=1024*200):
    frame = np.frombuffer(chunk, np.uint8)
    img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imshow("stream", img)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

cv2.destroyAllWindows()