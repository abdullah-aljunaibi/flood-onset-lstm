// Font sizes of the text of a PDF, per page: swift fontsizes in.pdf out.json
// Writes {"pages":[{"page":1,"sizes":{"7.5":1234,...},"min":..,"n_chars":..}], "pagesize":[w,h]}: the sizes are those printed (PDFKit reads the text runs after the page scaling).
import Foundation
import PDFKit
import AppKit
let a = CommandLine.arguments
guard a.count >= 3, let doc = PDFDocument(url: URL(fileURLWithPath: a[1])) else { print("usage: fontsizes in.pdf out.json"); exit(1) }
var pages: [[String: Any]] = []
for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i), let s = page.attributedString else { continue }
    var sizes: [String: Int] = [:]
    var minSize = 1e9
    var total = 0
    s.enumerateAttribute(.font, in: NSRange(location: 0, length: s.length), options: []) { v, r, _ in
        if let f = v as? NSFont {
            let key = String(format: "%.1f", f.pointSize)
            let chars = (s.string as NSString).substring(with: r).filter { !$0.isWhitespace }.count
            if chars > 0 { sizes[key, default: 0] += chars; total += chars; minSize = min(minSize, Double(f.pointSize)) }
        }
    }
    let b = page.bounds(for: .mediaBox)
    pages.append(["page": i + 1, "sizes": sizes, "min": minSize, "n_chars": total, "w": b.width, "h": b.height])
}
let data = try JSONSerialization.data(withJSONObject: ["pages": pages], options: [.prettyPrinted])
try data.write(to: URL(fileURLWithPath: a[2]))
print("pages:", doc.pageCount)
