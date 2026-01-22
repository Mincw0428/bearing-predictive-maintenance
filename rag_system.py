from pinecone import Pinecone
import google.generativeai as genai
import time

# ==========================================
# 🔑 API 키 입력 (2개 다 필요합니다!)
# ==========================================
GROQ_API_KEY = "GROQ_API_KEY"
PINECONE_API_KEY = "PINECONE_API_KEY"

genai.configure(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스 연결 (이름이 사이트와 똑같아야 함)
index_name = "bearing-manual" 
index = pc.Index(index_name)

# 1. 매뉴얼 로드 및 클라우드 DB 업로드
def load_manual_to_db():
    try:
        with open("manual.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        # 문단 나누기
        chunks = [c for c in text.split("\n\n") if c.strip()]
        
        print(f"☁️ 클라우드(Pinecone)에 {len(chunks)}개 데이터 업로드를 시작합니다...")
        
        vectors = []
        for i, chunk in enumerate(chunks):
            # 구글 모델(768차원)로 임베딩
            embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )['embedding']
            
            # Pinecone 포맷에 맞게 포장
            vectors.append({
                "id": f"vec_{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            })
            
        # 업로드 (Upsert)
        index.upsert(vectors=vectors)
        print("✅ 업로드 완료! Pinecone 대시보드에서 Record Count가 올라갔는지 확인해보세요.")
        time.sleep(2) # 서버 반영 대기
        
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")

# 2. 검색 함수
def query_manual(query_text, n_results=1):
    # 질문도 똑같은 768차원으로 변환
    query_vec = genai.embed_content(
        model="models/text-embedding-004",
        content=query_text,
        task_type="retrieval_query"
    )['embedding']
    
    # Pinecone에서 비슷한 내용 찾기
    res = index.query(vector=query_vec, top_k=n_results, include_metadata=True)
    
    if res['matches']:
        return [match['metadata']['text'] for match in res['matches']]
    return ["관련 매뉴얼 없음"]

if __name__ == "__main__":
    load_manual_to_db()
    # 테스트
    print("\n[검색 테스트] 질문: '첨도 높음'")
    print(query_manual("첨도 높음"))