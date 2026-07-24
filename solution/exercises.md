# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Câu trả lời của model khi tăng chỉ số temperature càng cao thì càng sáng tạo. Nếu 0.0 sẽ chỉ đưa ra các thông tin chính xác về hang (thời gian phát hiện, thời gian mở cửa) thì đến 1.5 thông tin được đưa có cả thông tin so sánh (phóng đại): "chứa cả một tòa nhà chọc trời cao 40 tầng".

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Chọn trong khoảng từ 0.5 đến 1, mô hình đưa ra câu trả lời đủ sáng tạo để thu hút người dùng, nhưng cũng đưa ra các thông tin chính xác để người dùng sử dụng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Về chi phí: GPT-4o đắt hơn GPT-4o-mini khoảng 16,7 lần cho cùng một lượng token xử lý (dựa trên tỷ lệ giá 0.010 / 0.0006 đối với token đầu ra).
Trường hợp GPT-4o xứng đáng: Các tác vụ yêu cầu suy luận logic phức tạp, viết code chuyên sâu, phân tích dữ liệu nhiều tầng, hoặc xử lý ngôn ngữ đòi hỏi độ tinh tế cao (như dịch thuật văn học, sáng tạo nội dung sâu sắc).
Trường hợp nên dùng GPT-4o-mini: Các tác vụ có khối lượng lớn nhưng lặp đi lặp lại và đơn giản (ví dụ: phân loại văn bản, trích xuất dữ liệu, chatbot hỏi đáp thông tin cơ bản, tóm tắt đoạn văn ngắn) để tối ưu hóa chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác biệt rõ rệt về hệ thống từ vựng và chiều sâu kiến thức: phản hồi đầu tiên mang tính đại cương, trong khi phản hồi sau sử dụng dày đặc thuật ngữ chuyên ngành (DLT, mã băm, PoW, PoS). Điều này minh chứng rằng system prompt hoạt động như một bộ khung định hình "nhân cách" và ngữ cảnh cho LLM. Nó chi phối trực tiếp cách mô hình chọn lọc từ vựng, cấu trúc câu và mức độ chi tiết để truyền đạt cùng một thông tin sao cho khớp nhất với đối tượng mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số từ thực tế: 128 từ, số token ước lượng (từ / 0.75): 170.7 tokens, số token thực tế (tiktoken): 154 tokens => Độ chênh lệch: Thực tế cao hơn ước lượng 9.8%
Tiếng Việt tốn token hơn tiếng Anh vì có thêm chủ yếu do 3 nguyên nhân:
- Dữ liệu huấn luyện bị lệch: Tokenizer được tối ưu hóa cho tiếng Anh (nhận diện 1 từ dài = 1 token). Tiếng Việt xuất hiện ít hơn nên thường bị "băm" nhỏ thành nhiều mảnh.
- Ký tự có dấu: Hệ thống dấu (á, ế, ợ...) tốn nhiều byte mã hóa (UTF-8) hơn chữ Latinh thường, khiến bộ mã hóa không đọc được trọn chữ mà phải tách thành các token hoặc byte lẻ.
- Ngôn ngữ rời chữ Tiếng Việt viết rời từng âm tiết (ví dụ: "sinh viên"), trong khi tiếng Anh dính liền ("student"). Khoảng trắng nhiều khiến thuật toán phải cắt vụn văn bản nhiều hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming phát huy vai trò quan trọng nhất trong các ứng dụng tương tác trực tiếp với con người (như chatbot, trợ lý ảo) vì nó giảm thiểu độ trễ cảm nhận (Time to First Token), giúp người dùng không bị nản chí khi chờ đợi các phản hồi dài và tạo cảm giác giao tiếp tự nhiên. Ngược lại, non-streaming lại là lựa chọn phù hợp và an toàn hơn cho các tác vụ giao tiếp giữa hệ thống với hệ thống (xử lý hàng loạt, pipeline tự động), hoặc khi cần trích xuất dữ liệu định dạng chuẩn (như JSON, XML) bởi ứng dụng máy tính cần nhận trọn vẹn và xác thực tính toàn vẹn của toàn bộ đầu ra trước khi thực thi các logic tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm tải dần cho máy chủ đang bị nghẽn bằng cách kéo giãn thời gian chờ sau mỗi lần thất bại (ví dụ: 1s, 2s, 4s, 8s...), cho phép hệ thống API có đủ thời gian để phục hồi. Nếu hàng nghìn client cùng retry với một delay cố định (như 1s), hệ thống sẽ gặp hiện tượng "Thundering Herd" (hiệu ứng bầy đàn) — tất cả các request thất bại sẽ đồng loạt gửi lại yêu cầu vào chính xác cùng một thời điểm ở chu kỳ tiếp theo. Điều này tạo ra các đỉnh tải (spike) liên tiếp, khiến máy chủ không bao giờ thoát khỏi trạng thái quá tải và sập hoàn toàn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: Chuyên gia tư vấn và tối ưu hóa lịch trình thông minh.
System Prompt: "Bạn là một chuyên gia tư vấn tuyến đường và lịch trình. Hãy phân tích dữ liệu đầu vào của người dùng để đề xuất lộ trình tối ưu nhất. Luôn trả lời ngắn gọn, phân cấp thông tin bằng danh sách (bullet points) và từ chối trả lời các câu hỏi không liên quan đến lên kế hoạch."
Giải thích lựa chọn từ ngữ:
"ngắn gọn, phân cấp bằng danh sách": Bắt buộc LLM phải định dạng đầu ra có cấu trúc. Điều này giúp dễ đọc trên giao diện người dùng và tránh việc mô hình sinh ra các đoạn văn tường thuật dài dòng lãng phí token.
"từ chối trả lời các câu hỏi không liên quan": Đặt ranh giới (guardrails) để tránh hiện tượng jailbreak hoặc người dùng lạm dụng trợ lý cho các mục đích sai lệch (như làm thơ, giải toán), giúp tiết kiệm chi phí API.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Quản lý context bị giới hạn. Trợ lý hiện tại thao tác bằng cách đẩy nguyên xi lịch sử hội thoại dạng mảng (array) vào API. Khi số lượt chat tăng lên, lượng token sẽ phình to rất nhanh, vừa gây tốn kém chi phí (vì API tính tiền dựa trên tổng token input), vừa dễ dẫn đến việc mô hình "quên" thông tin ở đầu hoặc vượt quá giới hạn Context Window.
Đề xuất cải thiện: Triển khai cơ chế "Tóm tắt bộ nhớ cuốn chiếu" (Rolling Memory Summarization).
Cách triển khai: Thay vì giữ lại toàn bộ nguyên văn tin nhắn, ta thiết lập một ngưỡng (ví dụ: sau mỗi 5 lượt chat), ứng dụng sẽ tự động gọi một luồng API phụ chạy ngầm để yêu cầu LLM tóm tắt các ý chính của 5 lượt này. Bản tóm tắt này sau đó được lưu lại và chèn vào cuối system prompt như một bối cảnh nền, còn mảng tin nhắn gốc sẽ được dọn dẹp để lấy chỗ trống cho các lượt chat mới. Cách này giúp bot nhớ được thông tin dài hạn mà vẫn giữ payload gửi đi luôn nhỏ nhẹ.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
