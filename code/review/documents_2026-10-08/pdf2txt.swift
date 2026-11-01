import Foundation
import PDFKit
let a = CommandLine.arguments
guard a.count >= 3, let doc = PDFDocument(url: URL(fileURLWithPath: a[1])) else { print("usage: pdf2txt in.pdf out.txt"); exit(1) }
var out = ""
for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i) else { continue }
    out += "\n\n=====PAGE \(i+1)=====\n"
    out += page.string ?? ""
}
try? out.write(toFile: a[2], atomically: true, encoding: .utf8)
print("pages:", doc.pageCount)
