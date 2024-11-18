import pandas as pd
import streamlit as st

# Streamlit 제목
st.title("여러 열의 텍스트 데이터 확인 도구")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type="csv")

if uploaded_file:
    # CSV 데이터 읽기
    data = pd.read_csv(uploaded_file)

    # 열 선택
    st.subheader("텍스트가 포함된 열 선택")
    selected_columns = st.multiselect(
        "텍스트를 확인할 열을 선택하세요:", data.columns, help="긴 텍스트가 포함된 열을 다중 선택하세요."
    )

    if selected_columns:
        # 행 번호 선택
        st.subheader("행 번호 선택")
        row_index = st.number_input(
            "보고 싶은 행의 번호를 입력하세요 (0부터 시작)", min_value=0, max_value=len(data) - 1, step=1
        )

        # 선택된 열의 데이터를 행별로 표시
        st.subheader(f"선택한 행 {int(row_index)}의 데이터")
        for column in selected_columns:
            st.text_area(f"{column} (열)", data[column].iloc[int(row_index)], height=200)
