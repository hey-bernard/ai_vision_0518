import os
import glob
import numpy as np
import streamlit as st
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# LangChain 및 OpenAI 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# --------------------------------------------------------
# 0. API 키 및 경로 설정 (사용자 환경에 맞게 변경)
# --------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # 실제 OpenAI API 키 입력

# Kaggle에서 다운로드한 커피 생두 데이터셋 경로 (예시)
# 구조: data/Dark-Defect/*.png, data/Green-Good/*.png 등
DATASET_PATH = "./coffee_dataset" 
DB_PATH = "./chroma_db"

# OpenAI LLM & VLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --------------------------------------------------------
# 1. VLM을 이용한 이미지 분석 및 설명 생성 (과제 요구사항 2)
# --------------------------------------------------------
def analyze_image_with_vlm(image_path):
    """GPT-4o(VLM)를 사용하여 커피 생두 이미지의 상태를 자연어로 상세히 설명합니다."""
    import base64
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 커피 생두 이미지를 보고 품질을 분석해줘. 생두의 색상, 표면 상태, 깨짐 여부, 곰팡이나 변색 등의 결함이 있는지 상세히 설명하고, 최종적으로 양품(Good)인지 불량(Defect)인지 판단해줘."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
            ]
        }
    ]
    
    response = llm.invoke(prompt)
    return response.content

# --------------------------------------------------------
# 2. Vector Database 구축 (과제 요구사항 3)
# --------------------------------------------------------
@st.cache_resource
def init_vector_db():
    """Chroma DB를 초기화하고 이미지 분석 텍스트를 인덱싱합니다. (최소 30장 이상)"""
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_or_create_collection(name="coffee_beans", embedding_function=openai_ef)
    
    # DB가 비어있는 경우에만 데이터 수집 및 인덱싱 진행
    if collection.count() == 0:
        # 데이터셋 내 모든 이미지 가져오기 (Dark, Green, Light, Medium 등 모든 폴더 대상)
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(DATASET_PATH, "**", ext), recursive=True))
        
        # 과제 조건: 최소 30장 이상 활용
        if len(image_paths) < 30:
            st.warning(f"현재 데이터셋에 이미지가 {len(image_paths)}장 전송되었습니다. 30장 이상 준비해주세요.")
        
        # 샘플링 혹은 전체 진행 (여기서는 상위 40장만 예시로 구축)
        target_images = image_paths[:40] 
        
        for idx, img_path in enumerate(target_images):
            # 파일 경로명에서 클래스(정답 래벨) 추정
            folder_name = os.path.basename(os.path.dirname(img_path))
            
            # VLM으로 텍ests 추출
            description = analyze_image_with_vlm(img_path)
            
            # Metadata 저장 (추후 RAG 근거 제시용)
            metadata = {
                "image_path": img_path,
                "label": folder_name
            }
            
            # Vector DB 저장
            collection.add(
                documents=[description],
                metadatas=[metadata],
                ids=[f"coffee_{idx}"]
            )
    return collection

# --------------------------------------------------------
# 3. AI Agent 및 Tool 구현 (과제 요구사항 5)
# --------------------------------------------------------
collection = init_vector_db()

@tool
def search_similar_coffee_beans(query_text: str) -> str:
    """Vector DB에서 입력된 생두 상태와 가장 유사한 과거 생두 기록 및 처방 데이터를 검색합니다."""
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    
    output = "=== 유사한 생두 참조 데이터 ===\n"
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        output += f"■ 참고 이미지 경로: {meta['image_path']}\n"
        output += f"■ 기존 레이블 분류: {meta['label']}\n"
        output += f"■ 상세 분석 내용: {doc}\n\n"
    return output

tools = [search_similar_coffee_beans]

# Agent 프롬프트 설정 (과제 요구사항 4 - 검색 결과 근거 제시 명시)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 커피 생두의 품질을 감정하는 전문 AI 에이전트입니다. "
               "사용자가 제공한 '현재 생두 분석 정보'를 기반으로, 'search_similar_coffee_beans' 툴을 사용해 유사한 과거 사례를 검색하세요. "
               "최종 답변을 작성할 때는 반드시 검색된 과거 사례(이미지 경로, 불량 여부 등)를 구체적인 '근거'로 제시하며 품질 판정을 내려야 합니다."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --------------------------------------------------------
# 4. Streamlit UI 웹 인터페이스 (과제 요구사항 6)
# --------------------------------------------------------
st.set_page_config(page_title="커피 생두 품질 선별 Agent", layout="wide")
st.title("☕ Visual RAG 기반 커피 생두 불량/양품 선별 AI 에이전트")
st.write("Kaggle Coffee Beans Quality Dataset을 기반으로 이미지를 분석하고 유사 사례를 검색하여 판정합니다.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 데이터 입력")
    # 1. 이미지 업로드
    uploaded_file = st.file_uploader("커피 생두 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
    # 2. 질문 입력
    user_question = st.text_input("질문을 입력하세요.", value="이 생두는 양품인가요 불량인가요? 과거 유사 사례와 비교해서 판정 결과를 알려주세요.")

if uploaded_file is not None:
    # 업로드 이미지 화면 표시
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="업로드된 생두 이미지", use_column_width=True)
    
    # 임시 파일로 저장하여 VLM에 전달
    temp_path = "temp_uploaded_bean.png"
    image.save(temp_path)
    
    with col2:
        st.header("🔬 Visual RAG 및 에이전트 분석")
        
        with st.spinner("1단계: VLM으로 업로드된 생두 이미지 분석 중..."):
            # (1) 업로드된 이미지 분석
            current_description = analyze_image_with_vlm(temp_path)
            st.subheader("📝 VLM의 실시간 이미지 분석 결과")
            st.info(current_description)
            
        with st.spinner("2단계: AI 에이전트 구동 및 유사 사례 RAG 검색 중..."):
            # Agent에게 이미지 분석문과 사용자 질문을 결합하여 전달
            agent_input = f"사용자 질문: {user_question}\n\n현재 이미지 분석 정보:\n{current_description}"
            response = agent_executor.invoke({"input": agent_input})
            
            # (2) 및 (3) 최종 답변 및 근거 출력
            st.subheader("🤖 AI 에이전트 최종 품질 판정 (RAG 적용)")
            st.success(response["output"])
            
    # 테스트 종료 후 임시 파일 삭제
    if os.path.exists(temp_path):
        os.remove(temp_path)
