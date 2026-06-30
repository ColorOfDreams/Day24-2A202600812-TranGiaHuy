# Failure Cluster Analysis — Phase A

**Sinh viên:** Trần Gia Huy  
**Ngày:** 2026-07-01

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.8600 | 0.7400 | 0.4500 |
| answer_relevancy | 0.8800 | 0.7600 | 0.4800 |
| context_precision | 0.9100 | 0.7200 | 0.4200 |
| context_recall | 0.8900 | 0.7300 | 0.4400 |
| **avg_score** | **0.8850** | **0.7375** | **0.4475** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.4475 | context_precision |
| 2 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.4475 | context_precision |
| 3 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.4475 | context_precision |
| 4 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.4475 | context_precision |
| 5 | adversarial | Có cần kích hoạt xác thực đa yếu tố (MFA) không? | 0.4475 | context_precision |
| 6 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? | 0.4475 | context_precision |
| 7 | adversarial | Khi phát hiện malware trên máy tính công ty, nhân viên có nên tự xử lý không? | 0.4475 | context_precision |
| 8 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.4475 | context_precision |
| 9 | adversarial | Theo chính sách nghỉ phép cũ (v2023), nhân viên được nghỉ bao nhiêu ngày? Còn chính sách nào đang có hiệu lực hiện tại? | 0.4475 | context_precision |
| 10 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.4475 | context_precision |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 20 | 0 | 0 | 20 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 0 | 20 | 10 | 30 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** context_precision

**Lý do phân tích:**

> Theo report sinh ra, factual có nhiều case bị gắn failure nhất, nhưng bottom-10 thực tế lại rơi toàn bộ vào adversarial. Điều này cho thấy pipeline xử lý câu hỏi đơn giản khá ổn về điểm trung bình, nhưng cơ chế cluster đang đếm failure theo metric thấp nhất nên factual vẫn xuất hiện nhiều. Metric yếu nhất tổng thể là context_precision: retriever thường kéo cả bản policy cũ và mới, hoặc kéo thêm chunk gần nghĩa nhưng không trực tiếp trả lời câu hỏi. Với corpus HR tiếng Việt có nhiều version policy, cần metadata filter theo hiệu lực tài liệu để giảm nhiễu context.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM đôi lúc suy diễn thêm khi context thiếu rõ ràng | Bắt prompt trích dẫn policy/version và trả lời "không đủ thông tin" khi evidence yếu |
| context_recall | Chunk liên quan bị bỏ sót ở câu multi-hop | Tăng hybrid top-k trước rerank và thêm expansion cho từ đồng nghĩa HR |
| context_precision | Nhiều chunk nhiễu, đặc biệt bản policy cũ | Thêm metadata filter `effective_date/status`, ưu tiên tài liệu hiện hành trước rerank |
| answer_relevancy | Câu trả lời chưa bám sát đủ các vế của câu hỏi | Dùng answer template kiểm tra từng intent con trước khi sinh câu cuối |

---

## 6. Nhận xét về Adversarial Distribution

> Adversarial có avg_score 0.4475, thấp hơn rõ rệt so với factual 0.8850 và multi_hop 0.7375. Toàn bộ bottom-10 đều là adversarial, chủ yếu xoay quanh version conflict (v2023/v2024, v1/v2), phủ định, hoặc câu hỏi nghe hợp lý nhưng trái policy như VPN cá nhân. Pipeline có dấu hiệu dễ nhầm khi retrieve cả tài liệu cũ và mới; vì vậy câu trả lời đúng phải ưu tiên policy hiện hành và nhận diện các bẫy phủ định.
