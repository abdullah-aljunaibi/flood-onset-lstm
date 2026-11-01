// OCR of a picture with Apple's Vision framework (macOS): swift ocr in.png out.json
// Writes [{"text":..., "confidence":..., "x":..., "y":..., "w":..., "h":...}, ...] (box in pixels, origin at the top left), one entry per recognized line of text.
import Foundation
import Vision
import AppKit

let a = CommandLine.arguments
guard a.count >= 3, let img = NSImage(contentsOfFile: a[1]), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("usage: ocr in.png out.json"); exit(1)
}
let W = Double(cg.width), H = Double(cg.height)
var rows: [[String: Any]] = []
let req = VNRecognizeTextRequest { r, e in
    guard let obs = r.results as? [VNRecognizedTextObservation] else { return }
    for o in obs {
        guard let c = o.topCandidates(1).first else { continue }
        let b = o.boundingBox
        rows.append(["text": c.string, "confidence": Double(c.confidence), "x": b.minX * W, "y": (1 - b.maxY) * H, "w": b.width * W, "h": b.height * H])
    }
}
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false
req.minimumTextHeight = 0.004
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try handler.perform([req])
let data = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted])
try data.write(to: URL(fileURLWithPath: a[2]))
print("lines:", rows.count, "image:", Int(W), "x", Int(H))
