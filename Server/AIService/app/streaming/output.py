import cv2

# NumPy 배열을 바로 보내면 전송시 크기가 커지므로 JPEG로 압축 후 전송
def encode_jpeg(frame, quality: int = 80):
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

    if not success:
        return None
    # HTTP 통신을 진행할 땐 binary 데이터가 필요하므로 tobytes()를 사용하여 bytes 객체로 변환
    return encoded_frame.tobytes()