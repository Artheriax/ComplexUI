# /// script
# dependencies = ["torch>=2.2"]
# ///
import torch

conditioning: CONDITIONING

if not isinstance(conditioning, dict):
    raise TypeError("conditioning must be a CONDITIONING wrapper")
cond = conditioning.get("cond")
if not isinstance(cond, torch.Tensor):
    raise TypeError("conditioning wrapper is missing its cond tensor")

result = dict(conditioning)
result["cond"] = torch.zeros_like(cond)
pooled = result.get("pooled")
if isinstance(pooled, torch.Tensor):
    result["pooled"] = torch.zeros_like(pooled)
result["text"] = ""

conditioning = result
return conditioning: CONDITIONING
