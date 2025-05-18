# Stable Diffusion
##  Stable Diffusion 기반 이미지 생성에 따른 학습 데이터 효율화 연구
생성형 AI인 **Stable Diffusion**을 활용해 인공 데이터를 생성하고, 이를 실제 데이터와 조합하여 학습 성능을 실험하는 내용

---

# CONTENTS
1. 연구 개요 (Overview)
2. 사용 기술 및 모델 (Tech Stack)
3. 데이터 구성 (Dataset)
4. 실험 설정 및 방법 (Experiment)
5. 실험 결과 (Results)
6. 결론 및 향후 연구 (Conclusion)
7. 이미지 예시 (Samples
8. 참고 및 지원 (Reference)

--- 
  
# 연구 개요 (Overview)

<div align="center" style="height: 100vh;">
  <img src="https://github.com/user-attachments/assets/1f464f34-d808-4af7-aa1a-e0dd688c45a8" alt="Train 1" width="500" height="400" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/259c3f0c-db17-4fb2-ba70-4a2a94451c12" alt="Test 1" width="500" height="400" style="margin-left: 10px;">
</div>

- 실제 이미지 수집의 한계를 보완하기 위해, Stable Diffusion으로 생성한 인공 이미지를 학습 데이터에 적용
- 인공 데이터의 비율을 조절하며 ResNet18 기반 분류 모델의 성능을 비교

---

# 학습 파라미터 및 모델 

- Stable Diffusion (Diffusion 기반 이미지 생성)
- ResNet18 (사전 학습된 CNN 모델)
- Optimizer: Adam  
- Learning Rate: 0.0001  
- Epoch: 25  
- Batch Size: 32  
- Loss: Cross Entropy Loss

---

# 데이터 구성 (Dataset)
- 총 클래스: 6개 (곰, 개, 고양이, 거북이, 사슴, 독수리)
- 데이터 구성:
  - 학습: 1348장
  - 검증: 162장
  - 테스트: 162장
- 데이터 비율 구성:
  - 실제 100%
  - 인공:실제 = 5:5, 6:4, 7:3, 8:2

---

## 실험 설정 및 방법 (Experiment)
- 프롬프트 예시:  
  `"A photorealistic, highly detailed 8K photograph of a turtle in various settings and poses in the sea, shot with a Nikon Z7 II, 70-200mm lens, natural lighting, sharp focus, ultra-high-definition"`
- 사전 학습된 ResNet18을 사용해 비율별로 데이터셋을 학습하고 동일한 테스트셋으로 평가

<div align="center" style="height: 100vh;">
  <img src="https://github.com/user-attachments/assets/39aa0ee8-c225-49ee-9993-af9e21773269" alt="Train 1" width="500" height="400" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/b6204029-34f8-4e19-8a77-deb952ef0683" alt="Test 1" width="500" height="400" style="margin-left: 10px;">
</div>

---

## 실험 결과 (Results)
본 연구에서는 실제 이미지와 Stable Diffusion으로 생성된 인공 이미지의 비율을 조절하여 총 5개의 데이터셋 구성

:arrow_right: ResNet18 모델로 학습한 뒤 동일한 테스트셋을 사용하여 성능을 비교 진행


| 구성      | 정확도                           |
| ------- | ----------------------------- |
| 실제 100% | ███████████████████████ 98.15 |
| 6:4     | ████████████████████░░ 97.55  |
| 5:5     | █████████████████░░░░ 95.06   |
| 7:3     | ████████████████░░░░ 94.87    |
| 8:2     | ███████████████░░░░░ 94.20    |


<div align="center" style="height: 100vh;">
  <img src="https://github.com/user-attachments/assets/b4bf0de6-e050-43c2-92bc-e87042aa182a" alt="Train 1" width="500" height="400" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/9fdc8f72-ff1b-4c4c-ad0d-7b88071a0266" alt="Test 1" width="500" height="400" style="margin-left: 10px;">
</div>

- 가장 높은 정확도는 실제 이미지 100% 데이터셋에서 기록

- 6:4 구성의 혼합 데이터셋에서는 실제 데이터에 가까운 성능을 유지하면서도 인공 데이터를 효과적으로 활용

- 5:5, 7:3, 8:2 구성에서는 인공 이미지 비중이 높아질수록 정확도가 다소 감소

1️⃣ **인공 이미지 비율이 늘어날수록 정확도는 감소했으나, 성능은 일정 수준 유지**

2️⃣ **데이터 부족 상황에서 인공 이미지가 일정 부분 대체 역할 가능**

3️⃣ **인공 이미지의 비율이 과도해질 경우, 모델의 일반화 성능이 저하 가능성 O**

4️⃣ **실제 이미지와 인공 이미지 간의 균형 있는 조합이 중요**


---

## 결론 및 향후 연구 (Conclusion)
- Stable Diffusion 기반 생성 이미지는 실제 이미지 데이터를 보완할 수 있는 가능성을 확인하였다.
- 실제 이미지 비율이 낮아도 성능을 유지해, 생성형 AI의 데이터 보완 가능성을 시사하였다.


![image](https://github.com/user-attachments/assets/7258e3df-a6e0-40d7-8f88-142030608d30)

---


## 참고 및 지원 (Reference)
- 본 연구는 **2024년 과학기술정보통신부 및 정보통신기획평가원의 SW중심대학사업**의 지원을 받아 수행되었습니다. (2024-0-00047)
- Stable Diffusion 관련 논문:  
  [Diffusion Models in Vision: A Survey](https://doi.org/10.1109/TPAMI.2023.3268241)








