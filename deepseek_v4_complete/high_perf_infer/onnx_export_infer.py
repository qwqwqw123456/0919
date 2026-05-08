class ONNXExporter:
    @staticmethod
    def export(model, path):
        import torch.onnx
        torch.onnx.export(model, torch.randn(1,3,224,224), path)
