# Outline tutorial: Định vị tàu hoả bốn mùa với Bag-of-Queries

## Thông tin chung

- Bài toán: Tác vụ 2, Định vị tàu hoả bốn mùa, OLP'AI 2026
- Phương pháp chính: Bag-of-Queries (BoQ)
- Paper tham khảo: *BoQ: A Place is Worth a Bag of Learnable Queries*, CVPR 2024
- Mã nguồn tham khảo: `amaralibey/Bag-of-Queries`
- Dữ liệu thực hành: `data/nordland-part-2020.zip`
- Notebook: `notebooks/BoQ_Nordland_OLPAI2026.ipynb`

Tutorial gồm hai tài liệu. `README.md` giải thích bài toán, dữ liệu, kiến trúc và các quyết định kỹ thuật. Notebook chứa toàn bộ mã nguồn theo thứ tự từ chuẩn bị dữ liệu đến tạo `output.csv`.

## Mục tiêu học tập

Sau khi hoàn thành tutorial, người đọc có thể:

1. Diễn giải Tác vụ 2 dưới dạng bài toán Visual Place Recognition và image retrieval.
2. Chuẩn bị các cặp ảnh summer và winter của Nordland cho Metric Learning.
3. Giải thích luồng dữ liệu qua ResNet-50, các BoQ block và lớp tạo global descriptor.
4. Tự cài đặt kiến trúc BoQ bằng PyTorch và kiểm tra kích thước tensor tại từng bước.
5. Huấn luyện mô hình bằng Multi-Similarity Loss, đánh giá theo ngưỡng vị trí của đề và phân tích các trường hợp truy xuất sai.
6. Trích xuất embeddings, tìm ảnh winter gần nhất và tạo file submission đúng định dạng.

## Kiến thức cần có

Người đọc nên biết Python và các thao tác PyTorch cơ bản, gồm `Dataset`, `DataLoader`, `nn.Module`, forward pass, backward pass và optimizer. Phần lý thuyết sử dụng kiến thức nền về CNN, Transformer, attention, embeddings, cosine similarity và Metric Learning. Tutorial sẽ nhắc lại những khái niệm trực tiếp liên quan đến BoQ.

# Phần I. Bài toán và dữ liệu

## 1. Bối cảnh Visual Place Recognition

- VPR xác định vị trí của ảnh truy vấn bằng cách đối chiếu với một kho ảnh đã biết vị trí.
- Trong Tác vụ 2, query là ảnh summer và database là các frame winter trên cùng đoạn tuyến Nordland.
- Thay đổi mùa, ánh sáng, thời tiết, độ nhoè và chất lượng nén làm cho hai ảnh cùng vị trí có thể khác nhau rõ rệt.
- Đầu ra của mô hình là một global descriptor. Hai ảnh cùng vị trí cần có descriptor gần nhau trong không gian embedding.

Phần mở đầu dùng một vài cặp summer và winter để cho thấy mức thay đổi ngoại hình giữa hai mùa. Các ví dụ phải gồm cả cặp đúng và những frame winter lân cận có hình ảnh gần giống nhau.

## 2. Đầu vào, đầu ra và cách chấm

Với mỗi tập kiểm tra:

- `queries/` chứa ảnh summer cần định vị.
- `database/` chứa các frame winter liên tiếp của cùng đoạn tuyến.
- `query_id` và `winter_id` là tên file sau khi bỏ phần mở rộng `.jpg`.
- Mỗi `query_id` có đúng một dự đoán. Nhiều query được phép trỏ đến cùng một `winter_id`.

Hai ảnh được tính là cùng vị trí khi:

\[
|\operatorname{pos}(w)-\operatorname{pos}(q)| \le 10.
\]

Metric chính của tutorial là `Accuracy@±10`, tức tỷ lệ query có dự đoán Top-1 nằm trong ngưỡng trên. `Recall@5` và `Recall@10` được dùng để phân tích khả năng xếp hạng. Một query thành công tại `K` nếu ít nhất một ảnh đúng xuất hiện trong Top-K.

File submission có hai cột:

```csv
query_id,winter_id
q_7d1a0c,w_03be91
```

Notebook cần kiểm tra các điều kiện sau trước khi lưu file:

- Tên và thứ tự hai cột là `query_id,winter_id`.
- Mỗi query hợp lệ xuất hiện đúng một lần.
- Số dòng dự đoán bằng số ảnh trong `queries/`.
- Mọi `winter_id` đều thuộc `database/` của tập đang xét.
- Không áp đặt tính duy nhất cho `winter_id`.

## 3. Bộ dữ liệu Nordland dùng trong tutorial

File `data/nordland-part-2020.zip` có 1.750 ảnh summer và 1.750 ảnh winter. Hai chuỗi dùng chung dải chỉ số liên tục từ 251 đến 2000. Tutorial xem số thứ tự frame là nhãn `position`; ảnh summer và winter có cùng chỉ số tạo thành một cặp đã căn chỉnh.

Ví dụ:

| Ảnh | Mùa | `position` |
| --- | --- | ---: |
| `summer/images-00251.png` | summer | 251 |
| `winter/images-00251.png` | winter | 251 |
| `summer/images-00252.png` | summer | 252 |
| `winter/images-00252.png` | winter | 252 |

Notebook xác minh số lượng file, dải chỉ số, frame bị thiếu và mức khớp chỉ số giữa hai mùa trước khi tạo metadata.

## 4. Chia train và validation

Hai mùa phải dùng cùng một cách chia để mỗi vị trí chỉ xuất hiện trong một tập.

| Tập | Chỉ số frame | Số cặp ảnh |
| --- | --- | ---: |
| Train | 251 đến 1640 | 1.390 |
| Buffer, không sử dụng | 1641 đến 1650 | 10 |
| Validation | 1651 đến 2000 | 350 |
| Tổng | 251 đến 2000 | 1.750 |

```text
Train                  Buffer             Validation
251 ........ 1640 | 1641 ...... 1650 | 1651 ........ 2000
```

Khoảng cách nhỏ nhất giữa train và validation là 11 frame. Vì ngưỡng đúng của đề là 10 frame, không có train position nào nằm trong tập ground truth hợp lệ của validation. Tutorial vẫn trực quan hoá các frame quanh ranh giới để kiểm tra mức giống nhau về hình ảnh. Có thể chạy thêm thí nghiệm với buffer 25 hoặc 50 frame; đây là phân tích độ nhạy, không phải cấu hình mặc định.

## 5. Positive pairs và false negatives

Baseline gán mỗi `position` cho một place identity:

```text
summer_251 -> label 251
winter_251 -> label 251
```

Mỗi lớp có hai ảnh từ hai mùa. Batch sampler phải đưa cả hai ảnh của một số position vào cùng batch để loss luôn có positive pairs.

Các frame liền kề có thể mô tả gần như cùng một vị trí. Khi mọi position khác nhau đều bị xem là negative, một số cặp âm sẽ là false negatives. Tutorial giải thích giới hạn này và giữ exact-position labels cho pipeline đầu tiên. Phần mở rộng có thể dùng quy tắc theo khoảng cách:

- Positive khi chênh lệch position không quá 10.
- Negative khi chênh lệch position lớn hơn 20.
- Bỏ qua vùng từ 11 đến 20 khi mining pairs.

Quy tắc mở rộng cần miner tùy chỉnh và không nằm trong luồng chạy chính.

## 6. Mô phỏng giao diện của tập kiểm tra

Từ validation, notebook tạo một `mock_test` để thực hành inference và kiểm tra submission:

- Giữ toàn bộ ảnh winter trong `database/`.
- Lấy khoảng một ảnh summer trên mỗi 10 frame cho `queries/`.
- Chuyển ảnh sang RGB và lưu dưới định dạng `.jpg` để khớp giao diện của đề.
- Đổi tên file thành các ID ngẫu nhiên với seed cố định để tên không làm lộ position.
- Lưu mapping position bên ngoài thư mục `mock_test`; phần inference không được đọc mapping này.
- Dùng mapping riêng trong scorer để tính `Accuracy@±10`.

`mock_test` mô phỏng cấu trúc và cách chấm của đề. Nó không đại diện cho độ khó hay phân bố của public test và private test chính thức.

# Phần II. Từ baseline đến Bag-of-Queries

## 7. Baseline retrieval

Trước khi triển khai BoQ, tutorial xây dựng một baseline bằng ResNet-50 có ImageNet pretrained weights:

1. Trích xuất feature map từ backbone.
2. Dùng global average pooling để tạo descriptor.
3. Chuẩn hoá descriptor bằng L2.
4. Xếp hạng ảnh winter bằng cosine similarity.
5. Báo cáo `Accuracy@±10`, `Recall@5` và `Recall@10` trên validation.

Baseline xác nhận pipeline dữ liệu, metric và retrieval hoạt động trước khi thêm BoQ. Các kết quả của BoQ sẽ được so sánh với cùng một tập validation và cùng scorer.

## 8. Kiến trúc BoQ

Phần lý thuyết bám theo Figure 2 và Figure 4 của paper.

### 8.1. Backbone và local features

- Dùng ResNet-50 với ImageNet pretrained weights.
- Bỏ các tầng classification và lấy feature map không gian từ backbone.
- Dùng convolution `3 x 3` để chiếu số kênh về `proj_channels`.
- Flatten hai chiều không gian thành chuỗi local features \(X^0\).

Notebook in shape sau backbone, projection và flatten để người đọc thấy cách tensor chuyển từ \([B,C,H,W]\) sang \([B,HW,D]\).

### 8.2. Một BoQ block

Mỗi block gồm:

1. Transformer Encoder cập nhật local features.
2. Một tập learnable queries \(Q\).
3. Self-attention giữa các queries để chia sẻ ngữ cảnh.
4. Cross-attention, trong đó queries đọc thông tin từ local features.
5. Layer normalization cho đầu ra của queries.

Phần này giải thích rõ query của BoQ là tham số học được, khác với query image trong bài toán retrieval.

### 8.3. BoQ aggregator và global descriptor

Đầu ra query của tất cả BoQ block được nối lại, chiếu tuyến tính, flatten và chuẩn hoá L2. Notebook kiểm tra:

- Số attention heads chia hết kích thước projection.
- Descriptor có shape \([B,\text{output\_dim}]\).
- L2 norm của mỗi descriptor xấp xỉ 1.
- Forward và backward pass không sinh `NaN` hoặc `Inf`.

### 8.4. Attention maps

Figures 3, 5, 6 và 7 của paper được dùng để giải thích sự chuyên hoá của learnable queries. Notebook lấy attention weights từ mô hình, đưa chúng về kích thước ảnh và hiển thị các vùng mà một số query tập trung vào. Phần nhận xét chỉ dựa trên hình trực quan thu được, không gán ý nghĩa cố định cho từng query.

## 9. Metric Learning và Multi-Similarity Loss

Tutorial giải thích ba thành phần:

- Embedding similarity giữa các ảnh trong batch.
- Cách miner chọn positive và negative pairs có ích.
- Cách Multi-Similarity Loss kéo positive pairs lại gần và đẩy negative pairs ra xa.

Notebook sử dụng `MultiSimilarityMiner` và `MultiSimilarityLoss` từ `pytorch-metric-learning`, cùng bộ tham số của mã nguồn tham khảo. Trước khi train toàn bộ mô hình, một batch nhỏ được dùng để xác nhận loss hữu hạn và gradient truyền đến cả backbone đang mở khoá lẫn BoQ aggregator.

# Phần III. Notebook thực hành

## 10. Thiết lập và cấu hình

- Import thư viện và đặt random seed.
- Kiểm tra phiên bản PyTorch, CUDA và tên GPU.
- Khai báo đường dẫn dữ liệu, kích thước ảnh, batch size và tham số BoQ tại một cell cấu hình.
- Có `DEMO_MODE` để chạy một tập nhỏ trong một epoch.
- Chọn các tầng ResNet-50 được freeze hoặc fine-tune.

Notebook phải chạy tuần tự từ đầu. Mọi thư mục đầu ra được tạo bằng code và không phụ thuộc vào trạng thái từ lần chạy trước.

## 11. Đọc, kiểm tra và chia dữ liệu

1. Giải nén hoặc đọc dữ liệu từ `nordland-part-2020.zip`.
2. Trích xuất chỉ số frame từ tên file.
3. Kiểm tra đủ 1.750 ảnh cho mỗi mùa, dải 251 đến 2000 và không có chỉ số lệch giữa hai mùa.
4. Tạo bảng metadata gồm `path`, `season`, `position` và `split`.
5. Áp dụng split train, buffer và validation đã mô tả ở Phần I.
6. Hiển thị một số cặp cùng position và các frame quanh ranh giới 1640 đến 1651.

## 12. Dataset, transforms và batch sampler

- Resize và chuẩn hoá ảnh theo ImageNet statistics.
- Áp dụng augmentation cho train; validation chỉ resize và normalize.
- Dataset trả về ảnh, `position`, mùa và đường dẫn ảnh.
- Sampler chọn nhiều position trong một batch và lấy cả summer lẫn winter của mỗi position.
- Một cell minh hoạ batch và kiểm tra mỗi label xuất hiện đúng hai lần.

## 13. Chạy baseline

- Trích xuất descriptor bằng ResNet-50 và global average pooling.
- Tính cosine similarity giữa summer validation và winter validation.
- Xếp hạng kết quả và tính các metric.
- Lưu bảng kết quả baseline để so sánh với BoQ.

## 14. Cài đặt BoQ

Notebook triển khai lần lượt:

1. `BoQBlock`.
2. `BoQ` aggregator.
3. Wrapper kết nối ResNet-50 với BoQ.
4. Unit checks cho shape, L2 norm, forward pass và backward pass.

Tên lớp và luồng tensor nên gần với mã nguồn tham khảo để người đọc có thể đối chiếu, nhưng phần cài đặt trong notebook phải tự chứa và chạy độc lập.

## 15. Huấn luyện

- Dùng AdamW cho các tham số được phép cập nhật.
- Dùng learning-rate warmup và MultiStepLR theo cấu hình của mã nguồn tham khảo.
- Bật automatic mixed precision khi có CUDA.
- Theo dõi loss và learning rate theo epoch.
- Lưu checkpoint có `Accuracy@±10` tốt nhất trên validation.
- Cho phép tiếp tục train từ checkpoint.

Trong `DEMO_MODE`, mục tiêu là kiểm tra toàn bộ pipeline. Kết quả của chế độ này không được dùng để kết luận về chất lượng mô hình.

## 16. Đánh giá và chọn checkpoint

Với mỗi checkpoint cần đánh giá:

- `Accuracy@±10`, tương đương Recall@1 theo ground truth nhiều positive.
- Recall@5 và Recall@10.
- Phân bố sai số \(|\operatorname{pos}(w)-\operatorname{pos}(q)|\).
- Kết quả so với baseline trên cùng validation split.

Phần phân tích kiểm tra thêm ảnh hưởng của buffer 25 hoặc 50 frame nếu cần. Mọi bảng kết quả phải ghi rõ split, ngưỡng đúng và checkpoint được sử dụng.

## 17. Phân tích định tính

- Hiển thị query summer cùng Top-K ảnh winter và similarity score.
- Tách ví dụ đúng, gần đúng trong ngưỡng và sai ngoài ngưỡng.
- Quan sát các lỗi do cảnh lặp, đường hầm, ảnh tối, nhoè, sương mù hoặc chói sáng khi các hiện tượng này có trong dữ liệu.
- Hiển thị attention maps của một số learnable queries trên các ví dụ đã chọn.

## 18. Inference trên `mock_test`

1. Tạo embeddings cho toàn bộ `database/` theo batch.
2. Tạo embeddings cho `queries/` theo batch.
3. Lưu embeddings cùng ID ảnh để có thể chạy lại retrieval mà không cần forward model lần nữa.
4. Tính cosine similarity và chọn Top-1 `winter_id` cho từng `query_id`.
5. Dùng mapping ẩn của mock test để tính `Accuracy@±10` sau khi inference hoàn tất.

Với database lớn, tutorial giới thiệu FAISS như một lựa chọn thay cho phép nhân ma trận toàn bộ. Luồng mặc định dùng cosine similarity trực tiếp vì mock dataset còn nhỏ.

## 19. Tạo và kiểm tra `output.csv`

Notebook tạo DataFrame có đúng hai cột `query_id,winter_id`, sau đó chạy hàm validator trước khi lưu. Validator báo lỗi khi:

- Thiếu query hoặc có query trùng.
- Có ID không thuộc tập tương ứng.
- Sai tên cột hoặc thừa cột.
- Số dòng khác số ảnh query.

Notebook cũng mô phỏng quy tắc chấm của đề đối với query trùng, query lạ, query bị thiếu và winter ID không hợp lệ.

# Phần IV. Cấu trúc README

`README.md` sử dụng cùng thứ tự khái niệm với notebook:

1. Bài toán VPR và Tác vụ 2.
2. Dữ liệu, nhãn position, ngưỡng ±10 và định dạng submission.
3. Cách mô phỏng train, validation và mock test từ `nordland-part-2020.zip`.
4. Baseline ResNet-50.
5. Kiến trúc BoQ và luồng tensor.
6. Metric Learning và Multi-Similarity Loss.
7. Huấn luyện, đánh giá và phân tích lỗi.
8. Inference và tạo `output.csv`.
9. Hướng dẫn chạy notebook ở `DEMO_MODE` và chế độ đầy đủ.
10. Tài liệu tham khảo và cách trích dẫn paper.

Các kết quả số chỉ được đưa vào README sau khi notebook đã tạo ra chúng. Mỗi bảng kết quả phải ghi cấu hình backbone, kích thước ảnh, split, metric và checkpoint để người đọc có thể đối chiếu.

# Đầu ra của tutorial

Sau khi hoàn thiện, repository có các thành phần sau:

```text
README.md
notebooks/
└── BoQ_Nordland_OLPAI2026.ipynb
```

Notebook tạo thêm các artifact trong quá trình chạy:

```text
artifacts/
├── checkpoints/
├── embeddings/
├── figures/
├── mock_test/
└── output.csv
```
