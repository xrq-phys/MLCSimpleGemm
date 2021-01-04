import Foundation
import MLCompute

@_cdecl("mlcsgemm_simple_")
func MLCSimpleSGEMM(transA: Bool,
                    transB: Bool,
                    m: Int,
                    n: Int,
                    k: Int,
                    alpha: Float,
                    addrA: UnsafeMutablePointer<Float>,
                    addrB: UnsafeMutablePointer<Float>,
                    addrC: UnsafeMutablePointer<Float>) {
    // Specify shapes.
    var tA : MLCTensor? = nil
    var tB : MLCTensor? = nil
    if (!transA) {
        tA = MLCTensor(shape: [1, m, k], dataType: .float32)
    } else {
        tA = MLCTensor(shape: [1, k, m], dataType: .float32)
    }
    if (!transB) {
        tB = MLCTensor(shape: [1, k, n], dataType: .float32)
    } else {
        tB = MLCTensor(shape: [1, n, k], dataType: .float32)
    }
    let tC = MLCTensor(shape: [1, m, n], dataType: .float32)

    // Create computational graph.
    let iGraph = MLCGraph()
    let tAB = iGraph.node(with: MLCMatMulLayer(descriptor: MLCMatMulDescriptor(alpha: alpha,
                                                                               transposesX: transA,
                                                                               transposesY: transB)!)!,
                          sources: [tA!, tB!])
    iGraph.node(with: MLCArithmeticLayer(operation: .add), sources: [tAB!, tC])

    let iPlan = MLCInferenceGraph(graphObjects: [iGraph])
    iPlan.addInputs(["A": tA!, "B": tB!, "C": tC])
    iPlan.compile(options: .linkGraphs, device: MLCDevice.gpu()!)

    // Prepare inputs.
    let datA = MLCTensorData(bytesNoCopy: addrA, length: m * k * MemoryLayout<Float>.size)
    let datB = MLCTensorData(bytesNoCopy: addrB, length: k * n * MemoryLayout<Float>.size)
    let datC = MLCTensorData(bytesNoCopy: addrC, length: m * n * MemoryLayout<Float>.size)

    // Execute graph.
    iPlan.execute(inputsData: ["A": datA, "B": datB, "C": datC],
                  batchSize: 0,
                  options: .synchronous) { (ans, err, elapsed) in
        #if DEBUG
            print("MLC: Errors: \(String(describing: err))")
            print("MLC: Result: \(String(describing: ans))")
        #endif

        ans!.copyDataFromDeviceMemory(toBytes: addrC,
                                      length: m * n * MemoryLayout<Float>.size,
                                      synchronizeWithDevice: false)

        let arrayC = UnsafeBufferPointer(start: addrC, count: m * n)
        #if DEBUG
            print(Array(arrayC))
        #endif
    }
}


