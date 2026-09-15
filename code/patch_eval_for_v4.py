"""
Run ON THE POD to teach evaluate_cache.py about yasunet_v4 and yasunet_v4_adv.
Both use the v3 architecture (modes=10, n_refine=6) and the forward signature
model(inp, yp). Idempotent: safe to run more than once.
"""
import sys
path = "/home/ubuntu/TexasDataset/lift550/evaluate_cache.py"  # adjust if needed
content = open(path).read()

# 1. model-loading branch: extend the v2/v3 elif to include v4 + v4_adv
needle = 'elif model_name in ("yasunet_v2", "yasunet_v3", "yasunet_v3fskip"):'
if needle in content and "yasunet_v4" not in content.split(needle,1)[1][:400]:
    add = '''        elif model_name in ("yasunet_v4", "yasunet_v4_adv"):
            from yasunet_v3 import YASUNetV2
            model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
'''
    # insert the new elif right before the closing of the v3fskip branch's last model= line
    anchor = '            from yasunet_v3_fskip import YASUNetV2\n            model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)\n'
    assert anchor in content, "v3fskip anchor not found - check file"
    content = content.replace(anchor, anchor + add, 1)
    # also extend the outer membership test
    content = content.replace(
        'elif model_name in ("yasunet_v2", "yasunet_v3", "yasunet_v3fskip"):',
        'elif model_name in ("yasunet_v2", "yasunet_v3", "yasunet_v3fskip", "yasunet_v4", "yasunet_v4_adv"):',
        1)
    print("Patched model-loading branch for v4 / v4_adv")
else:
    print("Model-loading branch already includes v4 (or v3 branch missing)")

# 2. forward-call: include v4 variants in the model(inp, yp) path
fwd_needle = 'if model_name in ("yasunet", "yasunet_v2", "yasunet_v3", "yasunet_v3fskip"):'
if fwd_needle in content:
    content = content.replace(
        fwd_needle,
        'if model_name in ("yasunet", "yasunet_v2", "yasunet_v3", "yasunet_v3fskip", "yasunet_v4", "yasunet_v4_adv"):',
        1)
    print("Patched forward-call branch for v4 / v4_adv")
else:
    print("WARNING: forward-call needle not found - check the model(inp, yp) line manually")

open(path, "w").write(content)
print("Done. Verify with: grep -n yasunet_v4 evaluate_cache.py")
