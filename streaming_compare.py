import cv2
import numpy as np
import time
from datetime import datetime

# === 붉은색 부품 감지 함수 ===
def preprocess_image(image):
    """이미지를 전처리하여 붉은색 부품 개수를 계산"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 붉은색 범위 정의
    lower_red1 = np.array([0, 50, 50])  # 어두운 빨강
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # 밝은 빨강
    upper_red2 = np.array([180, 255, 255])

    # 붉은색 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 마스크를 부드럽게 처리 (노이즈 제거)
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 붉은색 부품 개수 계산
    shunt_count = sum(1 for contour in contours if cv2.contourArea(contour) > 300)

    # 감지된 영역 시각화
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # 작은 노이즈 제거
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 초록색 박스

    return image, shunt_count

# === 메인 함수 ===
def main():
    # === 1. 정답 이미지에서 붉은색 부품 감지 ===
    correct_image_path = "image/correct.jpg"  # 정답 이미지 경로
    reference_image = cv2.imread(correct_image_path)

    if reference_image is None:
        print(f"정답 이미지를 로드할 수 없습니다: {correct_image_path}")
        return

    # 정답 이미지 처리
    reference_processed, reference_count = preprocess_image(reference_image)

    # 창 크기 설정
    fixed_width, fixed_height = 800, 600
    resized_reference = cv2.resize(reference_processed, (fixed_width, fixed_height))

    # 정답 이미지에 개수 표시
    cv2.putText(resized_reference, f"Shunts: {reference_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === 2. 실시간 스트리밍으로 붉은색 부품 개수 비교 ===
    stream_url = "http://192.168.0.4:8080/video"  # 스트리밍 URL
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("스트리밍을 열 수 없습니다. URL을 확인하세요.")
        return

    # PASS 상태 지속 시간 추적 변수
    pass_start_time = None

    while True:
        # 실시간 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("스트리밍이 중단되었습니다.")
            break

        # 현재 이미지에서 붉은색 부품 감지
        processed_frame, test_count = preprocess_image(frame)

        # 실시간 이미지 크기 조정
        resized_frame = cv2.resize(processed_frame, (fixed_width, fixed_height))

        # 부품 개수 비교
        is_correct = (reference_count == test_count)
        message = f"PASS: {test_count} shunts" if is_correct else f"FAIL: {test_count} vs {reference_count}"
        color = (0, 255, 0) if is_correct else (0, 0, 255)

        # PASS 상태 유지 시간 확인
        if is_correct:
            if pass_start_time is None:  # PASS 시작 시점 기록
                pass_start_time = time.time()
            elif time.time() - pass_start_time >= 2:  # PASS 상태가 2초 이상 지속되면
                # 현재 시간을 파일 이름으로 캡처 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pass_image/pass_capture_{timestamp}.jpg"
                cv2.imwrite(filename, resized_frame)
                print(f"PASS 상태에서 이미지 저장: {filename}")
                break  # 스트리밍 종료
        else:
            pass_start_time = None  # FAIL 상태면 시간 초기화

        # 결과 메시지 표시
        cv2.putText(resized_frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 정답 이미지와 실시간 이미지를 동시에 표시
        cv2.namedWindow("Reference Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reference Image", fixed_width, fixed_height)
        cv2.resizeWindow("Live Stream", fixed_width, fixed_height)
        cv2.imshow("Reference Image", resized_reference)  # 정답 이미지
        cv2.imshow("Live Stream", resized_frame)  # 실시간 스트리밍 이미지

        # 'q' 키를 누르면 수동 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 스트리밍 종료
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
