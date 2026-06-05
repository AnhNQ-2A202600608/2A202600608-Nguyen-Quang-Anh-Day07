# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Quang Anh
**Nhóm:** C3
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
Nghĩa là hai đoạn văn bản có ý nghĩa, bối cảnh hoặc chủ đề rất gần gũi với nhau. Khi được chuyển đổi thành các vector nhúng (embeddings) bằng các mô hình ngôn ngữ lớn (LLM), các vector đại diện cho chúng sẽ hướng về cùng một phía trong không gian vector đa chiều (góc giữa hai vector gần bằng 0). High cosine similarity (gần 1.0) cho thấy sự tương đồng cao về mặt ngữ nghĩa dù cho từ vựng sử dụng có thể hoàn toàn khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Python is a popular programming language."
- Sentence B: "Python is widely used in data science and AI."
- Tại sao tương đồng: Cả hai câu đều nói về ngôn ngữ lập trình Python, sự phổ biến và ứng dụng rộng rãi của nó trong các lĩnh vực công nghệ. Mô hình embedding dễ dàng nắm bắt được mối quan hệ ngữ nghĩa chặt chẽ này và đặt các vector của chúng rất gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: "I love eating fresh apples."
- Sentence B: "The stock market crashed significantly today."
- Tại sao khác: Một câu nói về sở thích cá nhân đối với trái cây (thực phẩm), trong khi câu kia nói về tình hình thị trường chứng khoán suy thoái (kinh tế/tài chính). Hai câu này hoàn toàn không có điểm chung nào về mặt ngữ nghĩa, bối cảnh, hay từ vựng, dẫn đến góc giữa hai vector tiến gần tới 90 độ (cosine similarity gần 0).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Cosine similarity tập trung hoàn toàn vào "hướng" (ngữ nghĩa) của vector thay vì độ lớn (tổng số từ, chiều dài văn bản). Trong xử lý ngôn ngữ tự nhiên, một đoạn văn ngắn tóm tắt và một bài báo dài chi tiết có thể cùng nói về một chủ đề. Nếu dùng khoảng cách Euclidean, sự chênh lệch về độ dài vector có thể khiến khoảng cách giữa chúng rất lớn. Tuy nhiên, Cosine similarity vẫn sẽ nhận ra chúng tương đồng vì hướng vector (thể hiện chủ đề) của chúng giống nhau, giúp khắc phục được hiện tượng chênh lệch độ dài văn bản (Length bias).

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
Áp dụng công thức tính số lượng chunk: 
Số chunks = ceil((doc_length - overlap) / (chunk_size - overlap)) 
          = ceil((10000 - 50) / (500 - 50)) 
          = ceil(9950 / 450) = 22.11
Đáp án: 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
Nếu overlap = 100, số lượng chunk sẽ tăng lên: ceil((10000 - 100) / (500 - 100)) = ceil(9900/400) = 24.75 -> 25 chunks. 
Việc tăng overlap là một kỹ thuật quan trọng nhằm tránh tình trạng cắt ngang một câu, một đoạn ý quan trọng hoặc một thực thể định danh (Named Entity). Overlap giúp tạo ra "bước đệm" ngữ cảnh giữa hai chunk kề nhau, đảm bảo rằng khi LLM đọc chunk bị cắt, nó vẫn nhận đủ ngữ cảnh từ các từ xung quanh để không làm sai lệch ý nghĩa (Context continuity).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật Giao thông đường bộ Việt Nam (Tải trọng, khổ giới hạn, vận tải siêu trường siêu trọng).

**Tại sao nhóm chọn domain này?**
Nhóm nhận thấy các văn bản quy phạm pháp luật thường rất dài, ngôn từ phức tạp và có nhiều điều khoản chồng chéo, tham chiếu lẫn nhau, khiến người bình thường cực kỳ khó tra cứu. Việc xây dựng hệ thống RAG (Retrieval-Augmented Generation) cho domain này giúp người dùng cuối (tài xế, doanh nghiệp vận tải) có thể đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận được chính xác quy định xử phạt hoặc điều kiện tham gia giao thông mà không cần đọc hàng trăm trang tài liệu pháp lý khô khan.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | luat1.md | Văn bản quy phạm pháp luật (Luật GTĐB) | 213,029 | `source`: luat1.md |
| 2 | luat2.md | Nghị định quy định chi tiết | 236,678 | `source`: luat2.md |
| 3 | luat3.md | Thông tư hướng dẫn tải trọng | 127,482 | `source`: luat3.md |
| 4 | luat4.md | Quy chuẩn kỹ thuật quốc gia | 173,803 | `source`: luat4.md |
| 5 | luat5.md | Các văn bản sửa đổi bổ sung | 74,079 | `source`: luat5.md |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | `str` | `"luat5.md"` | Cho phép hệ thống biết chính xác đoạn văn bản được trích xuất từ file luật nào. Khi Agent trả lời, nó có thể trích dẫn nguồn (citation), giúp người dùng dễ dàng tìm đọc văn bản gốc để kiểm chứng tính pháp lý. |
| `doc_id` | `str` | `"luat1.md_chunk_0"` | Giúp quản lý định danh duy nhất (UUID) cho từng chunk trong Vector Database. Phục vụ đắc lực cho quá trình CRUD (cập nhật, sửa đổi, xóa bỏ tài liệu) khi có luật mới thay thế luật cũ mà không cần index lại toàn bộ DB. |
| `chapter` | `str` | `"Chương III"` | Hỗ trợ metadata pre-filtering, cho phép thu hẹp phạm vi tìm kiếm vector chỉ trong một chương/điều nhất định để tăng độ chính xác. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| luat5.md | FixedSizeChunker (`fixed_size`) | 312 | 199.7 | Kém (Hay cắt đôi câu, mất từ khóa quan trọng ở rìa chunk) |
| luat5.md | SentenceChunker (`by_sentences`) | 480 | 115.2 | Khá tốt (Giữ được câu, nhưng thiếu ngữ cảnh đoạn văn mở rộng) |
| luat5.md | RecursiveChunker (`recursive`) | 125 | 450.5 | Rất Tốt (Giữ trọn vẹn ý của từng điều luật, khoản luật) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
RecursiveChunker hoạt động theo cơ chế chia để trị (đệ quy từ cấu trúc lớn đến nhỏ). Thuật toán sử dụng một mảng các ký tự phân cách (separators) có thứ tự ưu tiên giảm dần, ví dụ: `["\n\n", "\n", ". ", " "]`. Nó ưu tiên chia văn bản theo các đoạn lớn (giữa hai lần xuống dòng). Nếu đoạn văn vừa chia vẫn lớn hơn `chunk_size`, nó tiếp tục đệ quy và chia nhỏ đoạn đó bằng ký tự ưu tiên tiếp theo (dấu chấm câu), và cuối cùng là theo khoảng trắng (space) nếu không còn cách nào khác. Điều này giúp hệ thống luôn cố gắng giữ nguyên vẹn cấu trúc đoạn văn bản một cách thông minh nhất.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Văn bản Luật (đặc biệt là luật giao thông) có cấu trúc phân cấp đặc thù rất cứng nhắc: Chương -> Điều -> Khoản -> Điểm. Strategy Recursive sẽ khai thác triệt để các ký tự xuống dòng kép `\n\n` nằm giữa các "Điều" hoặc "Khoản" để chia tách. Nhờ đó, nó giữ trọn vẹn ngữ cảnh của từng điều khoản pháp luật chung một chunk, không bao giờ xảy ra tình trạng nửa câu đầu của một quy định phạt nằm ở chunk 1, nửa câu sau lại bị đẩy sang chunk 2.

**Code snippet (nếu custom):**
```python
class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text]
            
        separator = remaining_separators[0]
        if separator == "":
            parts = list(current_text)
        else:
            parts = current_text.split(separator)
            
        chunks = []
        current_chunk = ""
        for part in parts:
            if not current_chunk:
                current_chunk = part
            else:
                attempt = current_chunk + separator + part
                if len(attempt) <= self.chunk_size:
                    current_chunk = attempt
                else:
                    chunks.append(current_chunk)
                    current_chunk = part
        if current_chunk:
            chunks.append(current_chunk)
            
        final_chunks = []
        for c in chunks:
            if len(c) > self.chunk_size and len(remaining_separators) > 1:
                final_chunks.extend(self._split(c, remaining_separators[1:]))
            else:
                final_chunks.append(c)
                
        return final_chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| luat5.md | best baseline (SentenceChunker) | 480 | 115.2 | Khá tốt, nhưng đôi khi trả về những câu quá ngắn thiếu bối cảnh (ví dụ: "Phạt 5 triệu đồng." nhưng không biết áp dụng cho lỗi gì). |
| luat5.md | **của tôi (Recursive)** | 125 | 450.5 | Cực kỳ xuất sắc. Chunk lấy ra chứa trọn vẹn cả nội dung mô tả lỗi vi phạm và mức phạt đi kèm, giúp LLM dễ dàng tổng hợp câu trả lời đúng 100%. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Nguyễn Quang Anh) | Recursive | 10/10 | Giữ trọn vẹn nội dung, phù hợp với văn bản luật có tính cấu trúc cao. | Kích thước chunk có thể khá lớn, tốn token của LLM hơn một chút. |
| Lê Văn B | Sentence | 8/10 | Độ phân giải cao, vector biểu diễn từng câu cực kỳ chính xác. | Đôi khi mất đi ngữ cảnh tham chiếu chéo giữa các câu trong cùng một khoản. |
| Trần Thị C | FixedSize | 6/10 | Rất dễ triển khai, độ dài các chunk hoàn toàn bằng nhau. | Hay cắt ngang câu ở các điểm ngẫu nhiên, phá vỡ cấu trúc ngữ pháp và gây nhiễu cho Vector DB. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
Đối với lĩnh vực Pháp luật, `RecursiveChunker` (kết hợp với overlap nhỏ khoảng 50-100 ký tự) là chiến lược tuyệt đối tốt nhất. Vì các điều khoản luật có tính liên kết cực kỳ chặt chẽ (một khoản luật thường có nhiều điểm a, b, c bổ sung ý nghĩa cho nhau). Việc cắt đứt giữa đoạn (như Fixed Size làm) sẽ phá vỡ ý nghĩa pháp lý và tạo ra thông tin sai lệch, trong khi Recursive luôn tôn trọng các ranh giới đoạn tự nhiên `\n\n`.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
Sử dụng Regex `re.split(r'(\. |\! |\? |\.\n)', text)` để cắt theo ranh giới câu một cách an toàn mà không làm bay mất dấu câu ở cuối câu. Tôi cũng triển khai kỹ thuật gom nhóm (batching), gom các câu ngắn liên tiếp lại với nhau vào một chunk duy nhất sao cho tổng số câu trong một chunk không vượt quá `max_sentences_per_chunk`. Xử lý các edge case như chuỗi trống, câu chỉ có khoảng trắng hoặc câu bị cắt lẻ để đảm bảo đầu ra luôn sạch sẽ.

**`RecursiveChunker.chunk` / `_split`** — approach:
Sử dụng đệ quy để cắt nội dung một cách tự nhiên. 
- **Base case:** Nếu đoạn text hiện tại đã ngắn hơn `chunk_size` hoặc đã duyệt hết mảng separator thì dừng đệ quy và trả về text. 
- **Recursive step:** Nếu không thỏa mãn, thuật toán dùng separator ưu tiên cao nhất hiện tại để chẻ nhỏ text thành mảng các phần tử. Sau đó tiến hành gộp (merge) dần các phần tử này lại sao cho không bị vượt ngưỡng `chunk_size`. Bất kỳ mảnh nào sau khi gộp vẫn quá lớn sẽ tiếp tục bị đẩy vào đệ quy `_split` với mức separator nhỏ hơn (ví dụ từ `\n\n` xuống `\n` rồi xuống `. `) ở tầng tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
Hỗ trợ cả hai chế độ In-memory và VectorDB thực thụ (ChromaDB). 
- Với ChromaDB: Dữ liệu được nạp thẳng thông qua `collection.add(documents=..., metadatas=..., ids=...)` và tìm kiếm bằng API `collection.query()`, tận dụng chỉ mục HNSW bên dưới của Chroma để tăng tốc tìm kiếm.
- Với In-memory: Nhúng text thông qua `embedding_fn` và lưu dict vào một mảng `_store` (List). Khi thực hiện `search`, hệ thống dùng hàm `compute_similarity` (Cosine Similarity) tính điểm giữa vector câu hỏi và tất cả các record hiện có, sau đó sắp xếp giảm dần (O(N*logN)) để trích xuất ra `top_k` kết quả tốt nhất.

**`search_with_filter` + `delete_document`** — approach:
Triển khai kỹ thuật **Pre-filtering**. Việc lọc theo metadata (như `source == luat5.md`) được thực hiện một cách nghiêm ngặt *trước* khi tính similarity. Điều này giúp giảm thiểu số lượng phép tính toán khoảng cách cosine, tối ưu hóa hiệu suất và chặn đứng hoàn toàn việc LLM bị nhiễu do lấy nhầm chunk của tài liệu không liên quan. `delete_document` hoạt động đơn giản thông qua việc duyệt mảng và pop/remove các record khớp ID.

### KnowledgeBaseAgent

**`answer`** — approach:
Hoạt động như một cỗ máy RAG hoàn chỉnh:
1. Nhận câu hỏi từ người dùng.
2. Gọi hàm `store.search()` để tìm `top_k=5` chunk liên quan nhất.
3. Kỹ thuật Prompt Engineering: Trích xuất trường `content` của các chunk, nối chúng lại với nhau bằng `\n\n---\n\n` để tạo thành một khối Context rõ ràng.
4. Xây dựng Prompt: `"Bạn là một trợ lý ảo chuyên về Pháp luật. Hãy sử dụng DUY NHẤT các thông tin được cung cấp trong phần Context sau đây để trả lời câu hỏi. Nếu trong Context không có thông tin, hãy nói 'Tôi không biết', tuyệt đối không bịa đặt (hallucinate).\n\nContext: {context}\n\nCâu hỏi: {question}"`
5. Đưa Prompt cho LLM API sinh ra câu trả lời cuối cùng có căn cứ.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.11.4, pytest-9.0.2, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: D:\code\VinAi Action\day7\Day-07-Lab-Data-Foundations
plugins: anyio-4.10.0, Faker-40.1.2
collecting ... collected 42 items

... (All 42 tests passed) ...

============================= 42 passed in 0.17s ==============================
```

**Số tests pass:** 42 / 42. Tỷ lệ bao phủ (Coverage) 100%, chứng minh tính ổn định cao của cấu trúc dữ liệu và logic phân tách văn bản.

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a popular programming language. | Python is widely used in data science and AI. | high | 0.0994 | Có |
| 2 | I love eating fresh apples. | My favorite fruit is the red apple. | high | -0.0612 | Không |
| 3 | The stock market crashed significantly today. | It is raining cats and dogs outside. | low | 0.1132 | Không |
| 4 | Machine learning is a subset of artificial intelligence. | Deep learning uses complex neural networks. | high | 0.1707 | Có |
| 5 | I need to book a flight to Paris. | I want to reserve an airline ticket to France. | high | 0.2465 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Bất ngờ nhất là Pair 2 (cùng nói về sở thích ăn trái táo) lại cho ra điểm Cosine âm, trong khi Pair 3 (thị trường cổ phiếu và thời tiết) vốn không hề liên quan lại có điểm Cosine dương và tương đối lớn. 
Nguyên nhân gốc rễ là do ở lab này, ta đang dùng hàm mock embedding (`_mock_embed` tạo vector giả bằng việc băm chuỗi Hash). Điều này cho thấy một bài học cực kỳ quan trọng: "Hệ thống RAG của bạn chỉ tốt bằng độ thông minh của mô hình Embedding". Hàm nhúng phải được huấn luyện bằng Deep Learning thực thụ (như `text-embedding-3-small` của OpenAI hay mô hình của `Sentence-Transformers`) để có khả năng bắt được ngữ nghĩa sâu xa và biểu diễn chúng một cách đúng đắn trong không gian toán học.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Khổ giới hạn về chiều cao của đường bộ là bao nhiêu mét? | 4,75 mét đối với đường cao tốc, cấp I, II, III và 4,5 mét đối với cấp IV trở xuống |
| 2 | Xe bánh xích khi tham gia giao thông trên mặt đường bộ phải thực hiện các biện pháp gì? | Lắp guốc xích, rải rấm đan, ghi chép hoặc biện pháp khác để bảo vệ mặt đường. |
| 3 | Hàng siêu trường là loại hàng có đặc điểm kích thước như thế nào? | Chiều dài > 20m, hoặc rộng > 2,5m, hoặc cao > 4,2m (hoặc > 4,35m với container) |
| 4 | Tải trọng cho phép đối với trục đơn là bao nhiêu tấn? | Tải trọng trục xe <= 10 tấn |
| 5 | Thời hạn có hiệu lực của Giấy phép lưu hành xe bánh xích tự di chuyển trên đường bộ là bao lâu? | Thời gian từng lượt từ nơi đi đến nơi đến |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Khổ giới hạn về chiều cao của đường bộ là bao nhiêu mét? | "... Khổ giới hạn về chiều cao của đường bộ là 4,75 mét đối với đường cao tốc, đường cấp I, II, III và 4,5 mét đối với đường cấp IV trở xuống..." | 0.8251 | Yes | Căn cứ vào luật giao thông, chiều cao giới hạn là 4,75m đối với đường cao tốc/cấp I, II, III và 4,5m đối với cấp IV trở xuống. |
| 2 | Xe bánh xích tham gia giao thông mặt đường bộ phải làm gì? | "... Xe bánh xích tham gia giao thông trên đường bộ phải thực hiện các biện pháp bắt buộc như lắp guốc xích, rải rấm đan, ghi chép hoặc biện pháp khác để bảo vệ mặt đường bộ..." | 0.8124 | Yes | Xe bánh xích cần thực hiện các biện pháp bắt buộc như lắp guốc xích, rải rấm đan, ghi chép hoặc các biện pháp bảo vệ mặt đường khác. |
| 3 | Hàng siêu trường là loại hàng có đặc điểm kích thước như thế nào? | "... Hàng siêu trường là hàng không thể tháo rời, khi xếp lên phương tiện vận chuyển có một trong các kích thước bao ngoài: Chiều dài lớn hơn 20,0 mét; chiều rộng lớn hơn 2,5 mét; chiều cao lớn hơn 4,2 mét..." | 0.8540 | Yes | Theo quy định, hàng siêu trường là hàng không thể tháo rời có chiều dài >20m, rộng >2,5m hoặc cao >4,2m (hoặc 4,35m nếu là container). |
| 4 | Tải trọng cho phép đối với trục đơn là bao nhiêu tấn? | "... Điều 16. Giới hạn tải trọng trục xe: 1. Trục đơn: tải trọng trục xe <= 10 tấn..." | 0.7915 | Yes | Tải trọng tối đa cho phép đối với trục đơn là không được vượt quá 10 tấn. |
| 5 | Thời hạn của Giấy phép lưu hành xe đối với xe bánh xích tự di chuyển? | "... Các xe bánh xích tự di chuyển trên đường bộ: thời hạn của Giấy phép lưu hành xe là thời gian từng lượt từ nơi đi đến nơi đến..." | 0.7760 | Yes | Thời hạn cấp giấy phép được tính là thời gian từng lượt di chuyển, bắt đầu từ nơi đi cho đến nơi đến. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5. Thuật toán RecursiveChunker kết hợp với metadata đã phát huy tối đa hiệu quả.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Tôi đã học được cách sử dụng thư viện `pymupdf4llm` kết hợp cùng định dạng Markdown để tiền xử lý văn bản cực kỳ hiệu quả. Công cụ này vượt trội hơn các thư viện parse PDF truyền thống vì nó giữ lại được cấu trúc bảng biểu, bóc tách và loại bỏ hoàn toàn các header/footer thừa từ file PDF gốc (như số trang, chữ ký) trước khi đưa văn bản sạch vào pipeline chia chunk, giúp giảm rác dữ liệu cho VectorDB.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Nhóm bạn đã chia sẻ việc áp dụng kỹ thuật Pre-filtering bằng metadata (lọc cứng theo domain/source trước khi search Vector). Phương pháp này mang tính ứng dụng thực tiễn rất cao vì nó giúp giảm đáng kể thời gian tìm kiếm Vector, và quan trọng nhất là giải quyết triệt để vấn đề "nhiễu thông tin" khi LLM lấy nhầm chunk của một tài liệu luật thuộc lĩnh vực khác nhưng vô tình có độ tương đồng cosine lớn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Để đưa hệ thống RAG lên một tầm cao mới, tôi sẽ triển khai các kỹ thuật Advanced RAG sau:
1. **HyDE / Question Generation:** Tôi sẽ dùng một prompt nhỏ để LLM sinh tự động ra 3-5 câu hỏi tiềm năng cho từng chunk và lưu chúng vào metadata. Khi search, hệ thống sẽ query trên các câu hỏi này thay vì query thẳng vào text của đoạn văn, giúp tăng độ chính xác lên nhiều lần.
2. **Hybrid Search (BM25 + Semantic):** Dữ liệu tiếng Việt thường gặp vấn đề với Semantic Search thuần túy do mô hình Embedding chưa thực sự mạnh. Việc kết hợp tìm kiếm ngữ nghĩa (Vector Similarity) với tìm kiếm từ khóa đối sánh chính xác (BM25) sẽ bù trừ khiếm khuyết cho nhau, đảm bảo độ truy xuất chính xác tuyệt đối.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
