def generate(maxn, filename):
  for i in range(1, maxn+1):
    print(r"\includegraphics[width=\textwidth, page="+str(i)+r", trim = 15mm 20mm 15mm 20mm]{"+str(filename)+r"}")
  
generate(10, "pdfs/外文翻译YOLOv4原文.pdf")
generate(9, "pdfs/外文翻译YOLOv4译文.pdf")