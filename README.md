<img alt="image" src="https://github.com/akimcse/akimcse/assets/63237214/4417ab2f-7ef3-4d0e-b551-1e0cc42efb62">

</br></br>

> 부모님이 오늘 어떤 음식을 얼마나 드셨지? 필요한 영양소는 골고루 섭취하신 걸까?
</br>

위와 같은 고민을 한번이라도 해본 적이 있으신가요? 바쁜 일상 속에 부모님이 드신 식단까지 관리하기란 쉽지 않습니다. 그래서 저희는 `CareSpoon` 을 생각해냈습니다.

`CareSpoon` 은 사용자가 식사 전 본인의 식단을 찍기만 하면 어떤 음식을 먹었는지, 얼마나 먹었는지, 그 안에 영양소는 어느정도 섭취했는지를 기록합니다. 보호자는 사용자의 식사 현황과 영양 정보를 확인할 수 있습니다. 손쉽게 영양성분 섭취 정보를 기록하고, 그래프를 통해 한눈에 파악할 수 있습니다.

우리는 다음 방법으로 시니어의 스마트한 건강 관리를 제공하고자 합니다.

- 식단 사진 촬영 후 영양성분(탄수화물, 단백질, 지방, 칼로리) 정보 기록
- 영양소 섭취 현황 그래프 제공
- 식사를 제 때 섭취하지 않았을 때 `Viewer`에게 알림

</br>

## Tech Stack

|         Frontend         |             Server           |               AI              |       
| :----------------------: | :---------------------------: | :---------------------------: |
| ![Android](https://img.shields.io/badge/Android-3DDC84?style=flat-square&logo=Android&logoColor=white) ![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=flat-square&logo=Kotlin&logoColor=white) | ![Java](https://img.shields.io/badge/Java-003B57?style=flat-square&logo=java&logoColor=white) ![SpringBoot](https://img.shields.io/badge/Springboot-6DB33F?style=flat-square&logo=Springboot&logoColor=white) ![GoogleCloud](https://img.shields.io/badge/GoogleCloud-4285F4?style=flat-square&logo=GoogleCloud&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=Mysql&logoColor=white) | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white) ![YOLOv5](https://img.shields.io/badge/YOLO-00FFFF?style=flat-square&logo=YOLO&logoColor=white)  |

</br>


## Features
<table>
  <tr>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/e480aa9b-7005-479a-aab8-c627c9a84d39"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/9348ca7c-5468-4df9-b909-8ee384163119"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/68dad12b-38b5-4d2b-b1a4-7961aba96642"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/10eb9c4e-4ae4-4877-a89a-8d5833630bfa"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/bad0f05c-de8f-44f7-9f60-2300bb3dcff6"></td>
  </tr>
  <tr>
    <td align="center"><b>스플래시</b></td>
    <td align="center"><b>구글 로그인</b></td>
    <td align="center"><b>구글 로그인 2</b></td>
    <td align="center"><b>회원 가입</b></td>
    <td align="center"><b>회원 가입 2</b></td>
  </tr>
</table>

구글 계정을 연동하여 간편하게 로그인이 가능하다. 이후 기초대사량 계산을 위한 간단한 신체 정보를 기입하면 회원가입을 마칠 수 있다.
</br></br>

<table>
  <tr>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/32b8fd59-04fa-4749-bd33-b3bf0776ef6b"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/5e2f444e-03fb-408a-8cf1-07ec915eb161"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/defa901b-ffd0-4020-9544-9b795a8b1a22"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/42993e9e-378c-4ea9-9f47-6230369cb05e"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/779ad13c-c59c-49f2-b671-1c280a77aea4"></td>
  </tr>
  <tr>
    <td align="center"><b>홈</b></td>
    <td align="center"><b>홈2</b></td>
    <td align="center"><b>식단 세부 정보</b></td>
    <td align="center"><b>일별 통계</b></td>
    <td align="center"><b>월별 통계</b></td>
  </tr>
</table>

홈 화면은 시니어 사용자를 고려하여 크고 복잡하지 않은 요소들로 구성되어 있다. </br>
빈 카드를 눌러 식단을 촬영하면 AI가 자동으로 사진 속 음식 정보를 분석하여 영양 정보를 기룩해준다. </br>
식단 카드를 클릭하면 해당 식단에 대한 상세 영양 분석 정보를 확인할 수 있다. </br>
영양 통계 메뉴로 진입하면 AI가 분석한 데이터를 바탕으로 일별, 월별 통계 그래프를 제공한다.</br>
</br></br>

<table>
  <tr>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/0ef43cb1-d04d-400d-b1e7-cd79233c7b60"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/c0924522-29e8-4597-b451-c00096429463"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/a03a16a3-8d97-4f82-a901-23b191ece2c0"></td>
    <td><img width="200" src="https://github.com/akimcse/akimcse/assets/63237214/fab5951b-57af-478f-94a3-c42ee8747aec"></td>
    <td><img width="200" src="https://github.com/CareSpoon/.github/assets/63237214/f26beb15-477a-4f5e-a5b8-130d2e9b1573"></td>
  </tr>
  <tr>
    <td align="center"><b>식단 기록</b></td>
    <td align="center"><b>친구 관리</b></td>
    <td align="center"><b>친구 추가</b></td>
    <td align="center"><b>설정</b></td>
    <td align="center"><b>정보 변경</b></td>
  </tr>
</table>
 </br>
 식단 기록 메뉴로 진입하면 주간의 아침, 점심, 저녁 식단 기록을 확인할 수 있다. 마찬가지로 식단 카드를 클릭하면 해당 식단에 대한 상세 화면으로 진입한다. </br> 
친구와 공유하기 메뉴로 진입하면 현재 내 정보를 공유하고 있는 친구 목록을 볼 수 있고, 사용자가 원한다면 언제든 친구 관계를 끊을 수 있다. </br>
공유하기 화면 오른쪽 상단의 + 버튼을 눌러 친구 추가 화면으로 진입하면 고유 코드를 검색하여 새 유저를 친구로 등록할 수 있다.</br>
설정 화면 상단의 알림 토글 버튼을 통해 식단 촬영 시간에 대한 알림을 온/오프 할 수 있다. </br>
회원가입 시 입력한 신체정보는 권장 섭취량의 계산에 쓰이므로, 신체 정보가 바뀔 시 언제든 자유롭게 변경할 수 있다. </br>
</br></br>

</br>

## Project Architecture
![image](https://github.com/CareSpoon/.github/assets/79795051/8a2c4354-07ce-49de-918e-b417a31bfa28)

</br>

## AI
### Requirement
- Python: 3.7
```
pip install -r requirements.txt
```
### AI model
<table class="tg">
<tbody>
  <tr>
    <td><b>Model</b></td>
    <td>YOLOv5 custom dataset 사용하여 학습</td>
  </tr>
<tr>
    <td><b>Serving</b></td>
    <td>FastAPI 사용하여 AI 모델 서빙</td>
  </tr>
<tr>
    <td><b>Dataset</b></td>
<td>AI HUB 음식 이미지 및 영양정보 텍스트 전처리 후 사용</td>
</tr>
 <tr>
    <td><b>Train</b></td>
<td>final_best.pt<br/>100 epoch</td>
</tr>
<tr>
    <td><b>Accuracy</b></td>
    <td>Class: 한식 10종<br/>mAP@50: 0.98<br/>mAP@50-95: 0.86</td>
</tr>
<tr>
    <td><b>Result</b></td>
    <td><img width="400" src="https://github.com/CareSpoon/.github/assets/79077316/212f749d-0603-435d-9170-f418621e8a8f"></td>
</tr>
</tbody>
</table>
</br>

</br>
