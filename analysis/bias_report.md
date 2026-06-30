# LLM Judge Bias Report — Phase B

**Sinh viên:** Trần Gia Huy  
**Ngày:** 2026-07-01  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên ít nhất 5 cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nghỉ khi kết hôn | B | Ground truth đầy đủ hơn vì có thêm điều kiện không trừ phép năm |
| 2 | Mua thiết bị 55 triệu | B | Model answer chọn Director sai, ground truth yêu cầu CEO |
| 3 | Thưởng Tết tối thiểu | B | Ground truth đầy đủ hơn về điều kiện 6 tháng và pro-rata |
| 4 | Senior 9 năm thâm niên | B | Ground truth nêu rõ công thức phép và band lương |
| 5 | Hoàn trả khóa học 25 triệu | B | Judge chọn B dù ground truth bị thiếu trong map hiện tại |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | B | B | B | Yes |
| 2 | B | B | B | Yes |
| 3 | B | B | B | Yes |
| 4 | B | B | B | Yes |
| 5 | B | B | B | Yes |
| 6 | A | A | A | Yes |
| 7 | B | A | tie | No |
| 8 | B | B | B | Yes |
| 9 | A | A | A | Yes |
| 10 | B | A | tie | No |

**Position bias rate:** 20% (= 2 case NOT consistent / 10)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 0 | No |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 0 | No |
| 21 | 1 | 0 | No |
| 23 | 1 | 0 | No |
| 29 | 0 | 1 | No |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 1 | No |

**Cohen's κ:** -0.1538  
**Interpretation:** poor agreement

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 1 / 8 cases
- B thắng + B dài hơn A: 5 / 8 cases  
- **Verbosity bias rate:** 75%

**Kết luận:** Judge có xu hướng chọn câu trả lời dài hơn trong phần lớn case decisive. Đây là rủi ro vì câu dài chưa chắc đúng hơn; với HR policy, câu trả lời ngắn nhưng đúng ngưỡng/phê duyệt vẫn phải được ưu tiên hơn câu dài nhưng chứa policy sai hoặc outdated.

---

## 5. Nhận xét chung

> κ chưa đạt ngưỡng 0.6; kết quả -0.1538 cho thấy judge hiện chưa đủ đáng tin để dùng một mình trong production. Position bias rate 20% chưa vượt mức cảnh báo 30%, nhưng hai case tie cho thấy swap-and-average vẫn hữu ích để phát hiện bất ổn. Verbosity bias 75% đáng chú ý hơn, vì judge dễ thưởng cho câu dài thay vì kiểm tra đúng/sai theo policy. Trong production nên dùng judge như một tín hiệu phụ, kết hợp rubric rule-based, kiểm tra citation, và review thủ công với các case có rủi ro cao.
