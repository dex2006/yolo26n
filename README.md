# Night MLLM → OpenCV → YOLO26n Agent

流程：

原始圖片
→ MLLM 看圖片
→ JSON 決定 enhancement
→ OpenCV 修圖
→ 可人工調參
→ YOLO26n
→ 看偵測結果
→ 保存實驗紀錄

MLLM 不直接修改圖片。

目前允許：

- none
- gamma
- clahe
- gamma_clahe
- sharpen

所有非幾何影像處理不會改變 bounding box 座標，
因此對應 YOLO label 可以保留。

若未來使用 crop / rotate / perspective 等幾何處理，
必須同步修改 bounding box。
