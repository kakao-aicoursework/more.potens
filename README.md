# Kakao Developers helper BOT

## 챗봇 동작 알고리즘 절차
1. 챗봇 사용자의 질문 분석
2. LLM Chaining

## 2023-09-19
- 가이드를 통째로 넣고 사용법을 물어보자
  - SequentialChain을 사용했을 때 답변이 나오지 않았음

## 2023-09-20
- UI 수정
- 버튼 클릭 시 함수가 2번씩 실행됨. 이유는?

## 2023-09-21
- Embedding 도입

## 2023-09-22
- 카카오톡 소셜 API, 카카오톡 채널, 카카오싱크에 관련된 질문을 구분하는 routing 도입
  - 제대로 동작하지 않는듯. 어디가 문제일지

## 2023-09-25
- History 도입
  - 파일이 비어 있을 경우 오류 발생. 초기 히스토리는 어떻게 만들어야 하는가?