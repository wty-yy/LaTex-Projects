def generate(maxn, filename):
  for i in range(1, maxn+1):
    # print(r"\includegraphics[width=\textwidth, page="+str(i)+r", trim = 30mm 20mm 30mm 20mm, clip]{"+str(filename)+r"}")
    print(r"\includegraphics[width=\textwidth, page="+str(i)+r", trim = 15mm 20mm 15mm 20mm]{"+str(filename)+r"}")
  
generate(16, "pdfs/师梓豪_2204112376_外文翻译原文.pdf")
generate(16, "pdfs/师梓豪_2204112376_外文翻译译文.pdf")