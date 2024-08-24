1. 모터 제어
testMotor -> 모터 안됨 -> RPi.GPIO 모듈이라서 -> gpiozero 모듈에서 기본적인 버튼 동작 확인 -> testB -> gpioTest -> 최종 모터 제어 -> testM


2. 웹캠 확인
test_cam


3. 블루투스 확인
test_BT


4. 키 입력 확인
블루투스해봤는데 안잡힘 -> 그래서 키보드로 수동 조작 -> test_K -> 이미지 전처리 테스트 -> test_L


5. 텐서플로우 작동 확인
텐서플로우 모델 확인 -> test_TF_CVT 


6. 데이터 수집
주행하면서 데이터 저장하는 코드 -> test_K4


7. 모델 확인
우리가 만든 모델로 돌아가는지 확인 -> test_A

8. 객체 인식 확인
coco데이터셋으로 확인 -> testCV

9. 카메라 두 대 연결 확인
test_two_cam

10. 주행 + 객체 인식 확인
test, test_F

11. 모터 변경 후 실행
좌우 변경됨 -> test_LR

12. 담배 모델 적용
test_camera ~ test_LR9

13. 데이터 수집 방식 변경 (조명반영 RGB 환경 스레쉬홀드)
test_RGB

14. 최종 코드
test_F_F1.py