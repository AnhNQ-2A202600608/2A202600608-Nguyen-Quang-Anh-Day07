from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            context = "No relevant context found."
        else:
            context = "\n\n".join([r["content"] for r in results])
            
        prompt = (
            f"Bạn là chuyên gia tư vấn Luật Giao thông Việt Nam.\n"
            f"Dựa vào thông tin trong phần [Context] dưới đây, hãy trả lời [Câu hỏi] một cách ngắn gọn, chính xác.\n"
            f"Nếu [Context] không chứa thông tin để trả lời, HÃY TRẢ LỜI RÕ: 'Dữ liệu không đề cập đến vấn đề này', TUYỆT ĐỐI KHÔNG tự bịa ra thông tin.\n\n"
            f"[Context]:\n{context}\n\n"
            f"[Câu hỏi]: {question}"
        )
        return self.llm_fn(prompt)
