\section{Giới thiệu}
Hệ thống tự hành thường cần biết vị trí để di chuyển và ra quyết định, nhưng GPS không phải lúc nào cũng dùng được vì tín hiệu có thể yếu hoặc mất. Con người chỉ cần liếc qua vài biển hiệu hay một góc phố là nhận ra mình đang ở khu nào, và máy cũng có thể ước lượng vị trí bằng cách đối chiếu hình ảnh camera với những cảnh đã biết vị trí. Nhờ vậy, xe tự lái định vị được khi vào hầm hoặc đi giữa các tòa nhà cao tầng, robot nhận ra “chỗ này mình đi qua rồi” để hiệu chỉnh bản đồ, drone xác định vùng đang bay, còn kính AR hiện chỉ dẫn đúng chỗ. Một trường hợp đặc thù hơn là tàu hỏa, vì tàu đi lặp lại trên cùng một tuyến nên có thể xây dựng trước kho ảnh tham chiếu dọc đường để so với ảnh từ camera trên tàu và xác định vị trí.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\linewidth]{\projectroot/cross-season-VPR/Figures/muc1_vpr_overview.pdf}
  \caption{Các ứng dụng và quy trình của bài toán VPR.}
\end{figure}

Từ những nhu cầu đó hình thành bài toán Visual Place Recognition (VPR), tức xác định ảnh được chụp ở đâu. Ta có sẵn kho ảnh chụp các địa điểm đã biết, gọi là kho ảnh tham chiếu. Khi có ảnh mới cần xác định nơi chụp, gọi là ảnh truy vấn, hệ thống tìm trong kho ảnh trông giống nó nhất, từ đó suy ra ảnh mới được chụp ở cùng địa điểm với chúng. Tuy nhiên, “giống” ở đây không phải giống về bề ngoài, vì cùng một địa điểm có thể trông khác nhau giữa ngày và đêm, nắng và mưa, trong khi hai nơi khác nhau đôi khi giống nhau đến mức khó phân biệt.