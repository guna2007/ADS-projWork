import pdfkit
import os

html_file = "Final_Report.html"
pdf_file = "Final_Report.pdf"

if os.path.exists(html_file):
    pdfkit.from_file(html_file, pdf_file)
    print(f"PDF created: {pdf_file}")
else:
    print("HTML file not found")
